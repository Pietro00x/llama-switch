#pragma once

// Model downloader for the Switch app: HTTP/1.1 over libnx sockets + the ssl service.
//
// The body is written to "<destination>.part" and renamed to destination only once it is complete.
// An existing .part file is resumed with a Range request. The helpers in the second half of this
// header are pure functions (no libnx) so they can be compiled and tested on the host.

#include "app.h" // to_lower_ascii, trim (libnx-free)

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>

struct download_progress {
    uint64_t received     = 0;    // bytes in the .part file, including resumed bytes
    uint64_t total        = 0;    // full file size; 0 when the server did not tell
    uint64_t resumed_from = 0;    // bytes that were already on the SD card when the transfer began
    double   speed_bps    = 0.0;  // moving average over the last few seconds (0 = not known yet)
    double   eta_s        = -1.0; // seconds left; < 0 = unknown
    bool     connecting   = false; // true until the response body starts
};

// Called while connecting and while the body is received (also every ~250 ms while the connection
// is stalled). Return false to cancel: download_file then returns DOWNLOAD_RESULT_CANCELLED and keeps
// the .part file so a later call resumes it.
using download_progress_cb = std::function<bool(const download_progress & p)>;

// Rewrites huggingface.co ".../blob/main/..." page links to ".../resolve/main/..." download links.
std::string normalize_hf_url(std::string url);

// "<destination>.part"
inline std::string download_part_path(const std::string & destination) {
    return destination + ".part";
}

// Size of the resumable "<destination>.part" file, 0 if there is none.
uint64_t download_part_size(const std::string & destination);

// Free space of the SD card when path is on "sdmc:/". Returns false if it can't be determined.
bool download_free_space(const std::string & path, uint64_t & free_bytes);

enum download_result {
    DOWNLOAD_RESULT_OK,
    DOWNLOAD_RESULT_FAILED,    // err describes the problem
    DOWNLOAD_RESULT_CANCELLED, // the progress callback returned false
};

// Downloads url (http or https, follows up to 6 redirects) to destination.
// Creates parent directories, resumes an existing .part file, checks the free SD space, and on
// failure keeps the .part file for a later resume (unless its contents turned out to be unusable).
download_result download_file(
    const std::string & url,
    const std::string & destination,
    std::string & err,
    const download_progress_cb & progress_cb);

// Closes the socket/ssl services if download_file initialized them.
void download_exit();

//
// portable helpers (no libnx)
//

// Parses a non-negative decimal number that makes up all of s. Returns false otherwise.
inline bool dl_parse_u64(const std::string & s, uint64_t & out) {
    if (s.empty() || s.size() > 19) {
        return false;
    }
    uint64_t v = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') {
            return false;
        }
        v = v * 10 + (uint64_t) (s[i] - '0');
    }
    out = v;
    return true;
}

// Parses the status line and the header fields of raw (everything before the empty line).
// Header names are lower-cased; for repeated names the last value wins.
inline bool http_parse_headers(
    const std::string & raw,
    int & status_code,
    std::string & reason,
    std::map<std::string, std::string> & headers) {
    status_code = 0;
    reason.clear();
    headers.clear();

    size_t pos = 0;
    bool first = true;
    while (pos <= raw.size()) {
        size_t eol = raw.find('\n', pos);
        if (eol == std::string::npos) {
            eol = raw.size();
        }
        std::string line = raw.substr(pos, eol - pos);
        pos = eol + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (first) {
            first = false;
            // "HTTP/1.1 206 Partial Content"
            if (line.compare(0, 5, "HTTP/") != 0) {
                return false;
            }
            const size_t sp = line.find(' ');
            if (sp == std::string::npos) {
                return false;
            }
            const size_t sp2 = line.find(' ', sp + 1);
            const std::string code = line.substr(sp + 1, sp2 == std::string::npos ? std::string::npos : sp2 - sp - 1);
            uint64_t v = 0;
            if (code.size() != 3 || !dl_parse_u64(code, v) || v < 100) {
                return false;
            }
            status_code = (int) v;
            if (sp2 != std::string::npos) {
                reason = trim(line.substr(sp2 + 1));
            }
            continue;
        }

        if (line.empty()) {
            break;
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        headers[to_lower_ascii(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }
    return status_code != 0;
}

// Parses a Content-Range value: "bytes <first>-<last>/<total>" or "bytes */<total>".
// total is 0 when it is "*". first/last are 0 for the "*/<total>" form (has_range = false).
inline bool http_parse_content_range(
    const std::string & value,
    bool & has_range,
    uint64_t & first,
    uint64_t & last,
    uint64_t & total) {
    has_range = false;
    first = last = total = 0;

    const std::string v = trim(value);
    if (to_lower_ascii(v.substr(0, 6)) != "bytes ") {
        return false;
    }
    const std::string rest  = trim(v.substr(6));
    const size_t      slash = rest.find('/');
    if (slash == std::string::npos) {
        return false;
    }
    const std::string range = trim(rest.substr(0, slash));
    const std::string size  = trim(rest.substr(slash + 1));

    if (size != "*" && !dl_parse_u64(size, total)) {
        return false;
    }
    if (range == "*") {
        return size != "*";
    }
    const size_t dash = range.find('-');
    if (dash == std::string::npos ||
        !dl_parse_u64(range.substr(0, dash), first) ||
        !dl_parse_u64(range.substr(dash + 1), last) ||
        last < first || (total > 0 && last >= total)) {
        return false;
    }
    has_range = true;
    return true;
}

inline const char * http_status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 206: return "Partial Content";
        case 301: return "Moved Permanently";
        case 302: return "Found";
        case 303: return "See Other";
        case 307: return "Temporary Redirect";
        case 308: return "Permanent Redirect";
        case 400: return "Bad Request";
        case 401: return "Unauthorized (gated model? needs a token)";
        case 403: return "Forbidden (gated model or wrong URL)";
        case 404: return "Not Found (check the URL)";
        case 408: return "Request Timeout";
        case 410: return "Gone";
        case 416: return "Range Not Satisfiable";
        case 429: return "Too Many Requests (try later)";
        case 500: return "Internal Server Error";
        case 502: return "Bad Gateway";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        default:  return "";
    }
}

// "HTTP 404 Not Found (check the URL)"; prefers the table text, falls back to the server's reason.
inline std::string http_status_message(int status, const std::string & reason) {
    std::string text = http_status_text(status);
    if (text.empty()) {
        text = reason;
    }
    std::string msg = "HTTP " + std::to_string(status);
    if (!text.empty()) {
        msg += " " + text;
    }
    return msg;
}

// Resuming works with a one-byte overlap: for a .part file of part_size bytes the request asks for
// "bytes=<part_size - 1>-". A complete .part then still gets a 206 (with the total size in
// Content-Range) instead of a 416 that many CDNs send without Content-Range, and the overlapping
// byte is compared with the last byte on disk to detect a changed remote file.
inline uint64_t download_range_start(uint64_t part_size) {
    return part_size > 0 ? part_size - 1 : 0;
}

// "Range: bytes=N-\r\n" or "" when nothing is resumed.
inline std::string download_range_header(uint64_t part_size) {
    if (part_size == 0) {
        return std::string();
    }
    return "Range: bytes=" + std::to_string(download_range_start(part_size)) + "-\r\n";
}

enum download_action {
    DOWNLOAD_ACTION_REDIRECT,   // follow plan.location
    DOWNLOAD_ACTION_RESTART,    // 200: write the body from offset 0 (truncate the .part)
    DOWNLOAD_ACTION_APPEND,     // 206: the body starts at plan.body_start (== part_size - 1, the overlap byte)
    DOWNLOAD_ACTION_COMPLETE,   // 416 with total == part_size: the .part already is the whole file
    DOWNLOAD_ACTION_RETRY_FULL, // the .part can't be resumed: delete it and request the whole file
    DOWNLOAD_ACTION_ERROR,      // plan.err
};

struct download_plan {
    download_action action     = DOWNLOAD_ACTION_ERROR;
    uint64_t        body_start = 0;     // file offset of the first body byte
    uint64_t        body_end   = 0;     // file offset after the last body byte (if has_body_end)
    bool            has_body_end = false; // false: the body ends when the connection closes (or chunked)
    uint64_t        total      = 0;     // full file size, for display; 0 = unknown
    bool            chunked    = false; // body uses chunked transfer-encoding
    std::string     location;           // for DOWNLOAD_ACTION_REDIRECT (unresolved)
    std::string     err;                // for DOWNLOAD_ACTION_ERROR
};

// Decides what to do with a response to a request made with download_range_header(part_size).
inline download_plan download_plan_response(
    int status,
    const std::string & reason,
    const std::map<std::string, std::string> & headers,
    uint64_t part_size) {
    download_plan plan;

    std::map<std::string, std::string>::const_iterator it = headers.find("transfer-encoding");
    if (it != headers.end()) {
        const std::string te = to_lower_ascii(it->second);
        if (te.find("chunked") != std::string::npos) {
            plan.chunked = true;
        } else if (te != "identity") {
            plan.err = "unsupported transfer-encoding: " + it->second;
            return plan;
        }
    }

    // Content-Length is ignored when chunked (RFC 9112 6.3)
    uint64_t content_length = 0;
    bool     has_length     = false;
    it = headers.find("content-length");
    if (!plan.chunked && it != headers.end()) {
        has_length = dl_parse_u64(it->second, content_length);
        if (!has_length) {
            plan.err = "invalid Content-Length: " + it->second;
            return plan;
        }
    }

    bool     has_cr   = false;
    bool     cr_range = false;
    uint64_t cr_first = 0;
    uint64_t cr_last  = 0;
    uint64_t cr_total = 0;
    it = headers.find("content-range");
    if (it != headers.end()) {
        has_cr = http_parse_content_range(it->second, cr_range, cr_first, cr_last, cr_total);
    }

    if (status >= 300 && status < 400 && status != 304) {
        it = headers.find("location");
        if (it == headers.end() || it->second.empty()) {
            plan.err = http_status_message(status, reason) + " without Location header";
            return plan;
        }
        plan.action   = DOWNLOAD_ACTION_REDIRECT;
        plan.location = it->second;
        return plan;
    }

    if (status == 200) {
        plan.action        = DOWNLOAD_ACTION_RESTART;
        plan.body_start    = 0;
        plan.has_body_end  = has_length;
        plan.body_end      = has_length ? content_length : 0;
        plan.total         = plan.body_end;
        return plan;
    }

    if (status == 206) {
        if (!has_cr || !cr_range) {
            plan.err = "206 response without a valid Content-Range";
            return plan;
        }
        if (has_length && content_length != cr_last - cr_first + 1) {
            plan.err = "Content-Length does not match Content-Range";
            return plan;
        }
        if (part_size == 0 && cr_first == 0) {
            // no Range was sent but the server answered with the file from the start anyway
            plan.action       = DOWNLOAD_ACTION_RESTART;
            plan.body_start   = 0;
            plan.has_body_end = !plan.chunked;
            plan.body_end     = cr_last + 1;
            plan.total        = cr_total;
            return plan;
        }
        if (part_size == 0 || cr_first != download_range_start(part_size)) {
            // the server sent a different range than requested: start over without Range
            plan.action = DOWNLOAD_ACTION_RETRY_FULL;
            return plan;
        }
        // http_parse_content_range() guarantees last < total, so total >= part_size here
        plan.action       = DOWNLOAD_ACTION_APPEND;
        plan.body_start   = cr_first;
        plan.has_body_end = !plan.chunked;
        plan.body_end     = cr_last + 1;
        plan.total        = cr_total; // 0 for "bytes a-b/*"
        return plan;
    }

    if (status == 416) {
        if (part_size > 0 && has_cr && cr_total > 0 && cr_total == part_size) {
            plan.action = DOWNLOAD_ACTION_COMPLETE;
            plan.total  = cr_total;
            return plan;
        }
        if (part_size > 0) {
            // .part is larger than the remote file (or the size is unknown): it can't be trusted
            plan.action = DOWNLOAD_ACTION_RETRY_FULL;
            return plan;
        }
    }

    plan.err = http_status_message(status, reason);
    return plan;
}

// Moves the finished .part file over destination. The final path is never opened for writing: an
// existing destination is renamed to "<destination>.old" first (FAT rename() does not replace
// files), deleted only once the new file is in place, and restored if the rename fails.
inline bool download_finalize_part(const std::string & part_path, const std::string & destination, std::string & err) {
    const std::string backup = destination + ".old";

    FILE * probe = std::fopen(destination.c_str(), "rb");
    const bool had_old = probe != nullptr;
    if (probe) {
        std::fclose(probe);
    }

    if (had_old) {
        std::remove(backup.c_str());
        if (std::rename(destination.c_str(), backup.c_str()) != 0) {
            err = "cannot replace " + destination + ": " + std::strerror(errno);
            return false;
        }
    }
    if (std::rename(part_path.c_str(), destination.c_str()) != 0) {
        err = "cannot rename " + part_path + ": " + std::strerror(errno);
        if (had_old) {
            std::rename(backup.c_str(), destination.c_str());
        }
        return false;
    }
    if (had_old) {
        std::remove(backup.c_str());
    }
    return true;
}

// Incremental decoder for "Transfer-Encoding: chunked" bodies.
// dl_chunked_feed() decodes n input bytes into out and returns the number of body bytes produced
// (always <= n, so out may be the same buffer as in). Check d.error / d.done afterwards.
enum dl_chunked_state {
    DL_CHUNKED_STATE_SIZE,
    DL_CHUNKED_STATE_SIZE_EXT,
    DL_CHUNKED_STATE_SIZE_LF,
    DL_CHUNKED_STATE_DATA,
    DL_CHUNKED_STATE_DATA_CR,
    DL_CHUNKED_STATE_DATA_LF,
    DL_CHUNKED_STATE_TRAILER_START,
    DL_CHUNKED_STATE_TRAILER_LINE,
    DL_CHUNKED_STATE_TRAILER_LF,
    DL_CHUNKED_STATE_DONE,
};

struct dl_chunked {
    dl_chunked_state state     = DL_CHUNKED_STATE_SIZE;
    uint64_t         remaining = 0; // bytes left in the current chunk / accumulated size
    int              digits    = 0;
    bool             done      = false;
    bool             error     = false;
};

inline size_t dl_chunked_feed(dl_chunked & d, const char * in, size_t n, char * out) {
    size_t n_out = 0;
    size_t i     = 0;
    while (i < n && !d.done && !d.error) {
        const char ch = in[i];
        switch (d.state) {
            case DL_CHUNKED_STATE_SIZE:
            case DL_CHUNKED_STATE_SIZE_EXT:
                if (d.state == DL_CHUNKED_STATE_SIZE && ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))) {
                    if (d.digits >= 15) {
                        d.error = true;
                        break;
                    }
                    const int v = ch <= '9' ? ch - '0' : (ch | 0x20) - 'a' + 10;
                    d.remaining = d.remaining * 16 + (uint64_t) v;
                    d.digits++;
                } else if (ch == '\r') {
                    d.state = DL_CHUNKED_STATE_SIZE_LF;
                } else if (ch == '\n') {
                    d.state = DL_CHUNKED_STATE_SIZE_LF;
                    continue; // handle the end of the size line below without consuming
                } else if (d.digits > 0 && (ch == ';' || ch == ' ' || ch == '\t')) {
                    d.state = DL_CHUNKED_STATE_SIZE_EXT; // chunk extensions are ignored
                } else if (d.state == DL_CHUNKED_STATE_SIZE) {
                    d.error = true;
                    break;
                }
                i++;
                break;
            case DL_CHUNKED_STATE_SIZE_LF:
                if (ch != '\n' || d.digits == 0) {
                    d.error = true;
                    break;
                }
                i++;
                d.digits = 0;
                d.state  = d.remaining == 0 ? DL_CHUNKED_STATE_TRAILER_START : DL_CHUNKED_STATE_DATA;
                break;
            case DL_CHUNKED_STATE_DATA: {
                size_t take = n - i;
                if ((uint64_t) take > d.remaining) {
                    take = (size_t) d.remaining;
                }
                std::memmove(out + n_out, in + i, take);
                n_out       += take;
                i           += take;
                d.remaining -= take;
                if (d.remaining == 0) {
                    d.state = DL_CHUNKED_STATE_DATA_CR;
                }
                break;
            }
            case DL_CHUNKED_STATE_DATA_CR:
                if (ch == '\r') {
                    d.state = DL_CHUNKED_STATE_DATA_LF;
                    i++;
                } else if (ch == '\n') {
                    d.state = DL_CHUNKED_STATE_SIZE;
                    i++;
                } else {
                    d.error = true;
                }
                break;
            case DL_CHUNKED_STATE_DATA_LF:
                if (ch != '\n') {
                    d.error = true;
                    break;
                }
                d.state = DL_CHUNKED_STATE_SIZE;
                i++;
                break;
            case DL_CHUNKED_STATE_TRAILER_START:
                if (ch == '\r') {
                    d.state = DL_CHUNKED_STATE_TRAILER_LF;
                } else if (ch == '\n') {
                    d.state = DL_CHUNKED_STATE_DONE;
                    d.done  = true;
                } else {
                    d.state = DL_CHUNKED_STATE_TRAILER_LINE;
                }
                i++;
                break;
            case DL_CHUNKED_STATE_TRAILER_LINE:
                if (ch == '\n') {
                    d.state = DL_CHUNKED_STATE_TRAILER_START;
                }
                i++;
                break;
            case DL_CHUNKED_STATE_TRAILER_LF:
                if (ch != '\n') {
                    d.error = true;
                    break;
                }
                d.state = DL_CHUNKED_STATE_DONE;
                d.done  = true;
                i++;
                break;
            case DL_CHUNKED_STATE_DONE:
                d.done = true;
                break;
        }
    }
    return n_out;
}

// Transfer speed as a moving average over a window of samples taken at least 250 ms apart.
constexpr int k_dl_speed_samples = 16; // ~4 s window

struct dl_speed {
    uint64_t t_us[k_dl_speed_samples]  = {};
    uint64_t bytes[k_dl_speed_samples] = {};
    int      count = 0;
    int      head  = 0; // index of the next slot to write
    double   bps   = 0.0;
};

// bytes = bytes transferred so far in this session (monotonic), t_us = monotonic time.
inline void dl_speed_update(dl_speed & s, uint64_t t_us, uint64_t bytes) {
    if (s.count > 0) {
        const int newest = (s.head + k_dl_speed_samples - 1) % k_dl_speed_samples;
        if (t_us < s.t_us[newest] + 250000) {
            return;
        }
    }
    s.t_us[s.head]  = t_us;
    s.bytes[s.head] = bytes;
    s.head = (s.head + 1) % k_dl_speed_samples;
    if (s.count < k_dl_speed_samples) {
        s.count++;
    }
    if (s.count < 2) {
        return;
    }
    const int oldest = s.count < k_dl_speed_samples ? 0 : s.head;
    const int newest = (s.head + k_dl_speed_samples - 1) % k_dl_speed_samples;
    const uint64_t dt = s.t_us[newest] - s.t_us[oldest];
    if (dt > 0) {
        s.bps = (double) (s.bytes[newest] - s.bytes[oldest]) * 1e6 / (double) dt;
    }
}

// Seconds left, or -1 when unknown (no total or no speed yet).
inline double dl_eta_seconds(uint64_t received, uint64_t total, double bps) {
    if (total == 0 || bps <= 0.0) {
        return -1.0;
    }
    if (received >= total) {
        return 0.0;
    }
    return (double) (total - received) / bps;
}

// "1h02m", "3m05s", "42s", "--:--" for unknown
inline std::string dl_fmt_duration(double s) {
    if (s < 0.0 || s > 359999.0) {
        return "--:--";
    }
    const unsigned long t = (unsigned long) (s + 0.5);
    char buf[32];
    if (t >= 3600) {
        std::snprintf(buf, sizeof(buf), "%luh%02lum", t / 3600, (t / 60) % 60);
    } else if (t >= 60) {
        std::snprintf(buf, sizeof(buf), "%lum%02lus", t / 60, t % 60);
    } else {
        std::snprintf(buf, sizeof(buf), "%lus", t);
    }
    return std::string(buf);
}

// "812.3 MB", "1.07 GB"
inline std::string dl_fmt_size(uint64_t bytes) {
    char buf[32];
    if (bytes >= 1024ull * 1024 * 1024) {
        std::snprintf(buf, sizeof(buf), "%.2f GB", (double) bytes / (1024.0 * 1024.0 * 1024.0));
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f MB", (double) bytes / (1024.0 * 1024.0));
    }
    return std::string(buf);
}

// "2.3 MB/s", "512 KB/s", "--" for unknown
inline std::string dl_fmt_speed(double bps) {
    if (bps <= 0.0) {
        return "--";
    }
    char buf[32];
    if (bps >= 1024.0 * 1024.0) {
        std::snprintf(buf, sizeof(buf), "%.1f MB/s", bps / (1024.0 * 1024.0));
    } else {
        std::snprintf(buf, sizeof(buf), "%.0f KB/s", bps / 1024.0);
    }
    return std::string(buf);
}
