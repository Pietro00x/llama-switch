#include "download.h"

#include "app.h"
#include "sysinfo.h"

#include <switch.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static bool g_socket_initialized = false;
static bool g_ssl_initialized    = false;

// socket reads: TLS records are at most 16 KB, so one ssl read never returns more than that anyway
constexpr size_t   k_read_buf_size   = 256 * 1024;
constexpr u32      k_ssl_io_max      = 0x8000;
constexpr size_t   k_tls_record_max  = 16 * 1024;
// stdio buffer for the .part file: SD card writes are much faster in large blocks
constexpr size_t   k_write_buf_size  = 2 * 1024 * 1024;
// how long the connection may deliver nothing before the download fails
constexpr int      k_poll_ms         = 250;
constexpr int      k_stall_limit_ms  = 60 * 1000;
// keep some room on the SD card for the model list, logs, and FAT bookkeeping
constexpr uint64_t k_free_margin     = 16ull * 1024 * 1024;
constexpr int      k_max_redirects   = 6;

std::string normalize_hf_url(std::string url) {
    const std::string blob = "/blob/main/";
    const size_t host_pos = url.find("huggingface.co/");
    const size_t blob_pos = url.find(blob);
    if (host_pos != std::string::npos && blob_pos != std::string::npos) {
        url.replace(blob_pos, blob.size(), "/resolve/main/");
    }
    return url;
}

static bool stat_file_size(const std::string & path, uint64_t & size) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }
    size = (uint64_t) st.st_size;
    return true;
}

uint64_t download_part_size(const std::string & destination) {
    uint64_t size = 0;
    if (!stat_file_size(download_part_path(destination), size)) {
        return 0;
    }
    return size;
}

bool download_free_space(const std::string & path, uint64_t & free_bytes) {
    if (path.rfind("sdmc:/", 0) != 0) {
        return false;
    }
    FsFileSystem * fs = fsdevGetDeviceFileSystem("sdmc");
    if (!fs) {
        return false;
    }
    s64 free_space = 0;
    if (R_FAILED(fsFsGetFreeSpace(fs, "/", &free_space)) || free_space < 0) {
        return false;
    }
    free_bytes = (uint64_t) free_space;
    return true;
}

namespace {

struct parsed_url {
    std::string scheme;
    std::string host;
    std::string port;
    std::string path;
};

bool parse_url(const std::string & raw_url, parsed_url & out, std::string & err) {
    const size_t scheme_pos = raw_url.find("://");
    if (scheme_pos == std::string::npos) {
        err = "invalid URL (missing scheme)";
        return false;
    }

    out.scheme = to_lower_ascii(raw_url.substr(0, scheme_pos));
    if (out.scheme != "https" && out.scheme != "http") {
        err = "unsupported URL scheme: " + out.scheme;
        return false;
    }

    const std::string rest = raw_url.substr(scheme_pos + 3);
    const size_t path_pos = rest.find('/');
    const std::string host_port = path_pos == std::string::npos ? rest : rest.substr(0, path_pos);
    out.path = path_pos == std::string::npos ? "/" : rest.substr(path_pos);

    const size_t colon_pos = host_port.rfind(':');
    if (colon_pos != std::string::npos) {
        out.host = host_port.substr(0, colon_pos);
        out.port = host_port.substr(colon_pos + 1);
    } else {
        out.host = host_port;
        out.port = out.scheme == "https" ? "443" : "80";
    }

    if (out.host.empty()) {
        err = "invalid URL (missing host)";
        return false;
    }

    return true;
}

std::string url_authority(const parsed_url & u) {
    const bool default_port = (u.scheme == "https" && u.port == "443") || (u.scheme == "http" && u.port == "80");
    if (default_port) {
        return u.host;
    }
    return u.host + ":" + u.port;
}

std::string resolve_redirect_url(const parsed_url & base, const std::string & location) {
    if (location.rfind("http://", 0) == 0 || location.rfind("https://", 0) == 0) {
        return location;
    }
    if (location.rfind("//", 0) == 0) {
        return base.scheme + ":" + location;
    }
    if (!location.empty() && location[0] == '/') {
        return base.scheme + "://" + url_authority(base) + location;
    }

    std::string dir = "/";
    size_t slash = base.path.find_last_of('/');
    if (slash != std::string::npos) {
        dir = base.path.substr(0, slash + 1);
    }
    return base.scheme + "://" + url_authority(base) + dir + location;
}

bool ensure_socket_service(std::string & err) {
    if (g_socket_initialized) {
        return true;
    }
    // a larger TCP receive window keeps the Wi-Fi link busy on high-latency CDN connections
    SocketInitConfig cfg = *socketGetDefaultInitConfig();
    cfg.tcp_rx_buf_max_size = std::max<u32>(cfg.tcp_rx_buf_max_size, 0x80000);
    Result rc = socketInitialize(&cfg);
    if (R_FAILED(rc)) {
        rc = socketInitializeDefault();
    }
    if (R_FAILED(rc)) {
        err = "socket service init failed (rc " + rc_to_hex(rc) + ")";
        return false;
    }
    g_socket_initialized = true;
    return true;
}

bool ensure_ssl_service(std::string & err) {
    if (g_ssl_initialized) {
        return true;
    }
    Result rc = sslInitialize(3);
    if (R_FAILED(rc)) {
        err = "ssl service init failed (rc " + rc_to_hex(rc) + ")";
        return false;
    }
    g_ssl_initialized = true;
    return true;
}

// Asks nifm whether the console has an internet connection.
// Only used to explain a network failure, so an unavailable service just yields "unknown".
enum internet_state {
    INTERNET_STATE_UNKNOWN,
    INTERNET_STATE_CONNECTED,
    INTERNET_STATE_OFFLINE,
};

internet_state query_internet_state() {
    if (R_FAILED(nifmInitialize(NifmServiceType_User))) {
        return INTERNET_STATE_UNKNOWN;
    }
    NifmInternetConnectionType   type     = {};
    NifmInternetConnectionStatus status   = {};
    u32                          strength = 0;
    const Result rc = nifmGetInternetConnectionStatus(&type, &strength, &status);
    nifmExit();
    // the call itself fails when there is no network connection at all
    if (R_FAILED(rc) || status != NifmInternetConnectionStatus_Connected) {
        return INTERNET_STATE_OFFLINE;
    }
    return INTERNET_STATE_CONNECTED;
}

std::string offline_hint() {
    if (query_internet_state() == INTERNET_STATE_OFFLINE) {
        return " - no internet connection (check Wi-Fi / airplane mode)";
    }
    return std::string();
}

class net_connection {
public:
    bool connect_to(const parsed_url & u, std::string & err) {
        use_ssl_ = (u.scheme == "https");

        if (!ensure_socket_service(err)) {
            return false;
        }

        struct addrinfo hints {};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        struct addrinfo * results = nullptr;
        const int gai_rc = getaddrinfo(u.host.c_str(), u.port.c_str(), &hints, &results);
        if (gai_rc != 0 || !results) {
            err = "DNS lookup failed for " + u.host;
            if (gai_rc != 0) {
                err += std::string(" (") + gai_strerror(gai_rc) + ")";
            }
            err += offline_hint();
            return false;
        }

        bool connected = false;
        int  last_errno = 0;
        for (struct addrinfo * it = results; it; it = it->ai_next) {
            sock_ = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (sock_ < 0) {
                last_errno = errno;
                continue;
            }
            if (::connect(sock_, it->ai_addr, it->ai_addrlen) == 0) {
                connected = true;
                break;
            }
            last_errno = errno;
            ::close(sock_);
            sock_ = -1;
        }
        freeaddrinfo(results);

        if (!connected || sock_ < 0) {
            err = "cannot connect to " + u.host + ":" + u.port;
            if (last_errno != 0) {
                err += std::string(" (") + std::strerror(last_errno) + ")";
            }
            err += offline_hint();
            return false;
        }

        // a stalled connection eventually fails instead of blocking forever
        timeval tv {};
        tv.tv_sec = 30;
        (void) setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void) setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        if (!use_ssl_) {
            return true;
        }

        if (!ensure_ssl_service(err)) {
            return false;
        }

        Result rc = sslCreateContext(&ssl_ctx_, SslVersion_Auto);
        if (R_FAILED(rc)) {
            err = "TLS: sslCreateContext failed (rc " + rc_to_hex(rc) + ")";
            return false;
        }
        ssl_ctx_open_ = true;

        rc = sslContextCreateConnection(&ssl_ctx_, &ssl_conn_);
        if (R_FAILED(rc)) {
            err = "TLS: sslContextCreateConnection failed (rc " + rc_to_hex(rc) + ")";
            return false;
        }
        ssl_conn_open_ = true;

        (void) sslConnectionSetOption(&ssl_conn_, SslOptionType_DoNotCloseSocket, true);

        // with DoNotCloseSocket no new fd is returned; if one is, close_all() closes it
        ssl_fd_ = socketSslConnectionSetSocketDescriptor(&ssl_conn_, sock_);
        if (ssl_fd_ < 0 && errno != ENOENT) {
            err = "TLS: sslConnectionSetSocketDescriptor failed";
            return false;
        }

        rc = sslConnectionSetHostName(&ssl_conn_, u.host.c_str(), (u32) u.host.size());
        if (R_FAILED(rc)) {
            err = "TLS: sslConnectionSetHostName failed (rc " + rc_to_hex(rc) + ")";
            return false;
        }

        rc = sslConnectionSetVerifyOption(&ssl_conn_, SslVerifyOption_PeerCa | SslVerifyOption_HostName);
        if (R_FAILED(rc)) {
            err = "TLS: sslConnectionSetVerifyOption failed (rc " + rc_to_hex(rc) + ")";
            return false;
        }

        rc = sslConnectionDoHandshake(&ssl_conn_, nullptr, nullptr, nullptr, 0);
        if (R_FAILED(rc)) {
            err = "TLS handshake with " + u.host + " failed (rc " + rc_to_hex(rc) + ")";
            const Result cert_rc = sslConnectionGetVerifyCertError(&ssl_conn_);
            if (R_FAILED(cert_rc)) {
                err += ": certificate not accepted (rc " + rc_to_hex(cert_rc) + "), check the console date/time";
            } else {
                err += offline_hint();
            }
            return false;
        }

        return true;
    }

    bool write_all(const char * data, size_t len, std::string & err) {
        size_t off = 0;
        while (off < len) {
            if (use_ssl_) {
                const u32 chunk = (u32) std::min<size_t>(len - off, k_ssl_io_max);
                u32 sent = 0;
                Result rc = sslConnectionWrite(&ssl_conn_, data + off, chunk, &sent);
                if (R_FAILED(rc)) {
                    err = "TLS write failed (rc " + rc_to_hex(rc) + ")";
                    return false;
                }
                if (sent == 0) {
                    err = "TLS write returned 0";
                    return false;
                }
                off += sent;
                continue;
            }

            ssize_t sent = ::send(sock_, data + off, len - off, 0);
            if (sent < 0) {
                if (errno == EINTR) {
                    continue;
                }
                err = std::string("send failed: ") + std::strerror(errno);
                return false;
            }
            if (sent == 0) {
                err = "send returned 0";
                return false;
            }
            off += (size_t) sent;
        }
        return true;
    }

    // 1 = data (or EOF) can be read without blocking, 0 = nothing arrived within timeout_ms.
    // When polling is not available the result is always 1 and read_some() blocks instead
    // (bounded by SO_RCVTIMEO). While data streams in (the last read filled a whole TLS record
    // or the read buffer) the poll IPC is skipped.
    int wait_readable(int timeout_ms) {
        if (!poll_ok_ || streaming_) {
            return 1;
        }
        if (use_ssl_) {
            s32 pending = 0;
            if (R_SUCCEEDED(sslConnectionPending(&ssl_conn_, &pending)) && pending > 0) {
                return 1;
            }
            u32 events = 0;
            if (R_FAILED(sslConnectionPoll(&ssl_conn_, SslPollEvent_Read, &events, (u32) timeout_ms))) {
                poll_ok_ = false;
                return 1;
            }
            return events != 0 ? 1 : 0;
        }
        pollfd pfd {};
        pfd.fd     = sock_;
        pfd.events = POLLIN;
        const int rc = ::poll(&pfd, 1, timeout_ms);
        if (rc < 0) {
            if (errno == EINTR) {
                return 0;
            }
            poll_ok_ = false;
            return 1;
        }
        return rc > 0 ? 1 : 0;
    }

    ssize_t read_some(char * out, size_t size, std::string & err) {
        if (use_ssl_) {
            u32 read_size = 0;
            Result rc = sslConnectionRead(&ssl_conn_, out, (u32) std::min<size_t>(size, k_ssl_io_max), &read_size);
            if (R_FAILED(rc)) {
                err = "connection lost (TLS read rc " + rc_to_hex(rc) + ")";
                return -1;
            }
            streaming_ = read_size >= k_tls_record_max;
            return (ssize_t) read_size;
        }

        for (;;) {
            ssize_t n = ::recv(sock_, out, size, 0);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0) {
                err = std::string("connection lost (") + std::strerror(errno) + ")";
            }
            streaming_ = n > 0 && (size_t) n == size;
            return n;
        }
    }

    void close_all() {
        if (ssl_conn_open_) {
            sslConnectionClose(&ssl_conn_);
            ssl_conn_open_ = false;
        }
        if (ssl_ctx_open_) {
            sslContextClose(&ssl_ctx_);
            ssl_ctx_open_ = false;
        }
        if (ssl_fd_ >= 0) {
            ::close(ssl_fd_);
            ssl_fd_ = -1;
        }
        if (sock_ >= 0) {
            ::close(sock_);
            sock_ = -1;
        }
    }

    ~net_connection() {
        close_all();
    }

private:
    int sock_ = -1;
    int ssl_fd_ = -1;
    bool use_ssl_ = false;
    bool poll_ok_ = true;
    bool streaming_ = false;
    bool ssl_ctx_open_ = false;
    bool ssl_conn_open_ = false;
    SslContext ssl_ctx_ {};
    SslConnection ssl_conn_ {};
};

std::string write_error(uint64_t file_pos, int e) {
    std::string msg = std::string("SD card write failed: ") + std::strerror(e);
    if (e == ENOSPC) {
        msg = "SD card is full";
    }
    if (file_pos >= 0xFFFFFFFFull) {
        msg += " (FAT32 SD cards can't hold files of 4 GB or more; use exFAT)";
    }
    return msg;
}

enum transfer_result {
    TRANSFER_RESULT_OK,
    TRANSFER_RESULT_FAILED,
    TRANSFER_RESULT_CANCELLED,
    TRANSFER_RESULT_RETRY_FULL, // delete the .part and download the whole file again
};

// Reads the last byte of the .part file (the overlap byte of the resume request).
bool read_last_byte(const std::string & path, uint64_t size, unsigned char & out) {
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        return false;
    }
    bool ok = std::fseek(fp, (long) (size - 1), SEEK_SET) == 0;
    int ch = ok ? std::fgetc(fp) : EOF;
    std::fclose(fp);
    if (ch == EOF) {
        return false;
    }
    out = (unsigned char) ch;
    return true;
}

transfer_result download_transfer(
    const std::string & start_url,
    const std::string & part_path,
    uint64_t part_size,
    std::string & err,
    const download_progress_cb & progress_cb) {
    download_progress prog;
    prog.received     = part_size;
    prog.resumed_from = part_size;
    prog.connecting   = true;

    bool cancelled = false;
    auto report = [&]() {
        if (progress_cb && !progress_cb(prog)) {
            err       = "download cancelled";
            cancelled = true;
            return false;
        }
        return true;
    };
    auto fail = [&]() {
        return cancelled ? TRANSFER_RESULT_CANCELLED : TRANSFER_RESULT_FAILED;
    };

    unsigned char last_byte = 0;
    if (part_size > 0 && !read_last_byte(part_path, part_size, last_byte)) {
        return TRANSFER_RESULT_RETRY_FULL;
    }

    // one buffer for headers and body; the body is decoded in place when chunked
    char * io_buf = (char *) std::malloc(k_read_buf_size);
    if (!io_buf) {
        err = "out of memory (read buffer)";
        return TRANSFER_RESULT_FAILED;
    }
    struct buf_guard {
        char * p;
        ~buf_guard() { std::free(p); }
    } io_guard { io_buf };

    std::string current_url = start_url;
    for (int redirect_count = 0; redirect_count <= k_max_redirects; ++redirect_count) {
        parsed_url u;
        if (!parse_url(current_url, u, err)) {
            return TRANSFER_RESULT_FAILED;
        }

        if (!report()) {
            return fail();
        }

        net_connection conn;
        if (!conn.connect_to(u, err)) {
            return TRANSFER_RESULT_FAILED;
        }

        // the Range header is sent again on every redirect hop (HF redirects to its CDN)
        const std::string request =
            "GET " + u.path + " HTTP/1.1\r\n"
            "Host: " + url_authority(u) + "\r\n"
            "User-Agent: llama-switch/1.0\r\n"
            "Accept: */*\r\n"
            "Accept-Encoding: identity\r\n" +
            download_range_header(part_size) +
            "Connection: close\r\n\r\n";

        if (!conn.write_all(request.c_str(), request.size(), err)) {
            return TRANSFER_RESULT_FAILED;
        }

        // waits for data; keeps calling the progress callback (so B still cancels) while stalled
        // (the stall time is measured on the clock in case a poll returns before its timeout)
        auto read_next = [&](ssize_t & n) {
            uint64_t stall_start = 0;
            for (;;) {
                const uint64_t t_poll = now_us();
                if (conn.wait_readable(k_poll_ms) > 0) {
                    break;
                }
                const uint64_t t = now_us();
                if (t - t_poll < 10000) {
                    svcSleepThread(10 * 1000 * 1000ULL); // don't spin if the poll returned early
                }
                if (stall_start == 0) {
                    stall_start = t_poll;
                }
                const uint64_t stall_us = t - stall_start;
                if (stall_us >= (uint64_t) k_stall_limit_ms * 1000) {
                    err = "connection stalled (no data for 60 s)" + offline_hint();
                    return false;
                }
                if (stall_us >= 2000000) {
                    prog.speed_bps = 0.0;
                    prog.eta_s     = -1.0;
                }
                if (!report()) {
                    return false;
                }
            }
            n = conn.read_some(io_buf, k_read_buf_size, err);
            return n >= 0;
        };

        std::string response_prefix;
        size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            ssize_t n = 0;
            if (!read_next(n)) {
                return fail();
            }
            if (n == 0) {
                err = "connection closed before the response headers";
                return TRANSFER_RESULT_FAILED;
            }
            response_prefix.append(io_buf, (size_t) n);
            header_end = response_prefix.find("\r\n\r\n");
            if (header_end == std::string::npos && response_prefix.size() > 64 * 1024) {
                err = "response headers too large";
                return TRANSFER_RESULT_FAILED;
            }
        }

        int status_code = 0;
        std::string reason;
        std::map<std::string, std::string> headers;
        if (!http_parse_headers(response_prefix.substr(0, header_end + 2), status_code, reason, headers)) {
            err = "invalid HTTP response from " + u.host;
            return TRANSFER_RESULT_FAILED;
        }

        const download_plan plan = download_plan_response(status_code, reason, headers, part_size);
        switch (plan.action) {
            case DOWNLOAD_ACTION_REDIRECT:
                current_url = resolve_redirect_url(u, plan.location);
                continue;
            case DOWNLOAD_ACTION_ERROR:
                err = plan.err;
                return TRANSFER_RESULT_FAILED;
            case DOWNLOAD_ACTION_RETRY_FULL:
                return TRANSFER_RESULT_RETRY_FULL;
            case DOWNLOAD_ACTION_COMPLETE:
                prog.total      = plan.total;
                prog.connecting = false;
                (void) report();
                return TRANSFER_RESULT_OK;
            case DOWNLOAD_ACTION_RESTART:
            case DOWNLOAD_ACTION_APPEND:
                break;
        }

        const bool append = plan.action == DOWNLOAD_ACTION_APPEND;
        if (!append) {
            prog.resumed_from = 0;
        }
        prog.received   = append ? part_size : 0;
        prog.total      = plan.total;
        prog.connecting = false;

        if (plan.has_body_end && plan.body_end > prog.received) {
            uint64_t free_bytes = 0;
            if (download_free_space(part_path, free_bytes)) {
                const uint64_t need      = plan.body_end - prog.received;
                const uint64_t available = free_bytes + (append ? 0 : part_size); // restart truncates the .part
                if (need + k_free_margin > available) {
                    err = "not enough space on the SD card: need " + dl_fmt_size(need + k_free_margin) +
                          ", free " + dl_fmt_size(available);
                    return TRANSFER_RESULT_FAILED;
                }
            }
        }

        FILE * fp = std::fopen(part_path.c_str(), append ? "ab" : "wb");
        if (!fp) {
            err = std::string("cannot open ") + part_path + ": " + std::strerror(errno);
            return TRANSFER_RESULT_FAILED;
        }
        char * write_buf = (char *) std::malloc(k_write_buf_size);
        if (write_buf) {
            std::setvbuf(fp, write_buf, _IOFBF, k_write_buf_size);
        }
        auto close_part = [&]() {
            const bool ok = std::fclose(fp) == 0;
            const int  e  = errno;
            std::free(write_buf);
            errno = e;
            return ok;
        };

        dl_chunked chunked;
        dl_speed   speed;
        uint64_t   body_pos = plan.body_start; // file offset of the next body byte
        dl_speed_update(speed, now_us(), 0);

        // writes n body bytes; returns false on error (err set) or on a changed remote file
        bool changed = false;
        auto consume = [&](char * data, size_t n) {
            if (plan.chunked) {
                n = dl_chunked_feed(chunked, data, n, data);
                if (chunked.error) {
                    err = "invalid chunked encoding from server";
                    return false;
                }
            }
            if (plan.has_body_end && body_pos + n > plan.body_end) {
                n = (size_t) (plan.body_end - body_pos); // ignore anything past the announced size
            }
            // the overlap byte of a resume request must match what is already on disk
            if (append && n > 0 && body_pos < part_size) {
                if ((unsigned char) data[0] != last_byte) {
                    changed = true;
                    return false;
                }
                data++;
                n--;
                body_pos++;
            }
            if (n > 0 && std::fwrite(data, 1, n, fp) != n) {
                err = write_error(std::max<uint64_t>(plan.total, body_pos + n), errno);
                return false;
            }
            body_pos     += n;
            prog.received = body_pos;

            dl_speed_update(speed, now_us(), prog.received - prog.resumed_from);
            prog.speed_bps = speed.bps;
            prog.eta_s     = dl_eta_seconds(prog.received, prog.total, speed.bps);
            // called for every chunk so B cancels quickly; the caller throttles redrawing
            return report();
        };

        auto body_done = [&]() {
            if (plan.chunked) {
                return chunked.done;
            }
            return plan.has_body_end && body_pos >= plan.body_end;
        };

        bool ok = true;
        const size_t body_offset = header_end + 4;
        if (response_prefix.size() > body_offset) {
            ok = consume(&response_prefix[body_offset], response_prefix.size() - body_offset);
        } else {
            ok = report();
        }
        response_prefix.clear();
        response_prefix.shrink_to_fit();

        // without a size (and not chunked) the end of the connection is the end of the file
        while (ok && !body_done()) {
            ssize_t n = 0;
            if (!read_next(n)) {
                ok = false;
                break;
            }
            if (n == 0) {
                break;
            }
            ok = consume(io_buf, (size_t) n);
        }

        const bool closed = close_part();
        if (changed) {
            return TRANSFER_RESULT_RETRY_FULL;
        }
        if (!ok) {
            return fail();
        }
        if (!closed) {
            err = write_error(std::max<uint64_t>(plan.total, body_pos), errno);
            return TRANSFER_RESULT_FAILED;
        }

        const std::string resume_hint = " - start the download again to resume";
        if (plan.chunked && !chunked.done) {
            err = "connection closed early (" + dl_fmt_size(body_pos) + " received)" + resume_hint;
            return TRANSFER_RESULT_FAILED;
        }
        if (plan.has_body_end && body_pos != plan.body_end) {
            err = "connection closed early (" + dl_fmt_size(body_pos) + " of " +
                  dl_fmt_size(plan.total > 0 ? plan.total : plan.body_end) + ")" + resume_hint;
            return TRANSFER_RESULT_FAILED;
        }
        return TRANSFER_RESULT_OK;
    }

    err = "too many redirects";
    return TRANSFER_RESULT_FAILED;
}

} // namespace

download_result download_file(
    const std::string & input_url,
    const std::string & destination,
    std::string & err,
    const download_progress_cb & progress_cb) {
    const std::string url       = normalize_hf_url(input_url);
    const std::string part_path = download_part_path(destination);

    // everything below writes only to part_path; destination is touched by download_finalize_part()
    ensure_parent_dir(destination);

    for (int attempt = 0; attempt < 2; ++attempt) {
        uint64_t part_size = 0;
        if (!stat_file_size(part_path, part_size)) {
            part_size = 0;
        }

        const transfer_result res = download_transfer(url, part_path, part_size, err, progress_cb);
        if (res == TRANSFER_RESULT_CANCELLED) {
            return DOWNLOAD_RESULT_CANCELLED;
        }
        if (res == TRANSFER_RESULT_FAILED) {
            return DOWNLOAD_RESULT_FAILED;
        }
        if (res == TRANSFER_RESULT_RETRY_FULL) {
            // the partial file doesn't match the remote file (or can't be resumed): start over
            std::remove(part_path.c_str());
            if (attempt == 0 && part_size > 0) {
                continue;
            }
            err = "server sent an unexpected range; download could not be resumed";
            return DOWNLOAD_RESULT_FAILED;
        }

        return download_finalize_part(part_path, destination, err) ? DOWNLOAD_RESULT_OK : DOWNLOAD_RESULT_FAILED;
    }

    err = "download could not be resumed";
    return DOWNLOAD_RESULT_FAILED;
}

void download_exit() {
    if (g_ssl_initialized) {
        sslExit();
        g_ssl_initialized = false;
    }
    if (g_socket_initialized) {
        socketExit();
        g_socket_initialized = false;
    }
}
