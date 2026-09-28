#include "download.h"

#include "app.h"
#include "sysinfo.h"

#include <switch.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

static bool g_socket_initialized = false;
static bool g_ssl_initialized    = false;

std::string normalize_hf_url(std::string url) {
    const std::string blob = "/blob/main/";
    const size_t host_pos = url.find("huggingface.co/");
    const size_t blob_pos = url.find(blob);
    if (host_pos != std::string::npos && blob_pos != std::string::npos) {
        url.replace(blob_pos, blob.size(), "/resolve/main/");
    }
    return url;
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
    Result rc = socketInitializeDefault();
    if (R_FAILED(rc)) {
        err = "socketInitializeDefault failed: " + rc_to_hex(rc);
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
        err = "sslInitialize failed: " + rc_to_hex(rc);
        return false;
    }
    g_ssl_initialized = true;
    return true;
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
        if (getaddrinfo(u.host.c_str(), u.port.c_str(), &hints, &results) != 0 || !results) {
            err = "getaddrinfo failed for " + u.host;
            return false;
        }

        bool connected = false;
        for (struct addrinfo * it = results; it; it = it->ai_next) {
            sock_ = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (sock_ < 0) {
                continue;
            }
            if (::connect(sock_, it->ai_addr, it->ai_addrlen) == 0) {
                connected = true;
                break;
            }
            ::close(sock_);
            sock_ = -1;
        }
        freeaddrinfo(results);

        if (!connected || sock_ < 0) {
            err = "connect failed for " + u.host + ":" + u.port;
            return false;
        }

        if (!use_ssl_) {
            return true;
        }

        if (!ensure_ssl_service(err)) {
            return false;
        }

        Result rc = sslCreateContext(&ssl_ctx_, SslVersion_Auto);
        if (R_FAILED(rc)) {
            err = "sslCreateContext failed: " + rc_to_hex(rc);
            return false;
        }
        ssl_ctx_open_ = true;

        rc = sslContextCreateConnection(&ssl_ctx_, &ssl_conn_);
        if (R_FAILED(rc)) {
            err = "sslContextCreateConnection failed: " + rc_to_hex(rc);
            return false;
        }
        ssl_conn_open_ = true;

        (void) sslConnectionSetOption(&ssl_conn_, SslOptionType_DoNotCloseSocket, true);

        int out_fd = socketSslConnectionSetSocketDescriptor(&ssl_conn_, sock_);
        if (out_fd < 0 && errno != ENOENT) {
            err = "sslConnectionSetSocketDescriptor failed";
            return false;
        }

        rc = sslConnectionSetHostName(&ssl_conn_, u.host.c_str(), (u32) u.host.size());
        if (R_FAILED(rc)) {
            err = "sslConnectionSetHostName failed: " + rc_to_hex(rc);
            return false;
        }

        rc = sslConnectionSetVerifyOption(&ssl_conn_, SslVerifyOption_PeerCa | SslVerifyOption_HostName);
        if (R_FAILED(rc)) {
            err = "sslConnectionSetVerifyOption failed: " + rc_to_hex(rc);
            return false;
        }

        rc = sslConnectionDoHandshake(&ssl_conn_, nullptr, nullptr, nullptr, 0);
        if (R_FAILED(rc)) {
            err = "ssl handshake failed: " + rc_to_hex(rc);
            return false;
        }

        return true;
    }

    bool write_all(const char * data, size_t len, std::string & err) {
        size_t off = 0;
        while (off < len) {
            if (use_ssl_) {
                const u32 chunk = (u32) std::min<size_t>(len - off, 0x8000);
                u32 sent = 0;
                Result rc = sslConnectionWrite(&ssl_conn_, data + off, chunk, &sent);
                if (R_FAILED(rc)) {
                    err = "ssl write failed: " + rc_to_hex(rc);
                    return false;
                }
                if (sent == 0) {
                    err = "ssl write returned 0";
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

    ssize_t read_some(char * out, size_t size, std::string & err) {
        if (use_ssl_) {
            u32 read_size = 0;
            Result rc = sslConnectionRead(&ssl_conn_, out, (u32) std::min<size_t>(size, 0x8000), &read_size);
            if (R_FAILED(rc)) {
                err = "ssl read failed: " + rc_to_hex(rc);
                return -1;
            }
            return (ssize_t) read_size;
        }

        for (;;) {
            ssize_t n = ::recv(sock_, out, size, 0);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0) {
                err = std::string("recv failed: ") + std::strerror(errno);
            }
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
    bool use_ssl_ = false;
    bool ssl_ctx_open_ = false;
    bool ssl_conn_open_ = false;
    SslContext ssl_ctx_ {};
    SslConnection ssl_conn_ {};
};

bool parse_http_headers(
    const std::string & headers_raw,
    int & status_code,
    std::map<std::string, std::string> & headers) {
    std::istringstream stream(headers_raw);
    std::string line;

    if (!std::getline(stream, line)) {
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    status_code = 0;
    std::sscanf(line.c_str(), "HTTP/%*s %d", &status_code);
    if (status_code == 0) {
        return false;
    }

    while (std::getline(stream, line)) {
        if (line == "\r" || line.empty()) {
            break;
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = to_lower_ascii(trim(line.substr(0, colon)));
        std::string value = trim(line.substr(colon + 1));
        headers[key] = value;
    }

    return true;
}

} // namespace

bool download_file(
    const std::string & input_url,
    const std::string & destination,
    std::string & err,
    const download_progress_cb & progress_cb) {
    std::string current_url = normalize_hf_url(input_url);

    for (int redirect_count = 0; redirect_count < 6; ++redirect_count) {
        parsed_url u;
        if (!parse_url(current_url, u, err)) {
            return false;
        }

        net_connection conn;
        if (!conn.connect_to(u, err)) {
            return false;
        }

        std::string request =
            "GET " + u.path + " HTTP/1.1\r\n"
            "Host: " + url_authority(u) + "\r\n"
            "User-Agent: llama-switch-menu/1.0\r\n"
            "Accept: */*\r\n"
            "Connection: close\r\n\r\n";

        if (!conn.write_all(request.c_str(), request.size(), err)) {
            return false;
        }

        std::array<char, 8192> io_buf {};
        std::string response_prefix;
        size_t header_end = std::string::npos;

        while (header_end == std::string::npos) {
            ssize_t n = conn.read_some(io_buf.data(), io_buf.size(), err);
            if (n < 0) {
                return false;
            }
            if (n == 0) {
                err = "connection closed before headers";
                return false;
            }
            response_prefix.append(io_buf.data(), (size_t) n);
            header_end = response_prefix.find("\r\n\r\n");
            if (response_prefix.size() > 1024 * 1024) {
                err = "response headers too large";
                return false;
            }
        }

        const std::string raw_headers = response_prefix.substr(0, header_end + 2);
        int status_code = 0;
        std::map<std::string, std::string> headers;
        if (!parse_http_headers(raw_headers, status_code, headers)) {
            err = "failed to parse response headers";
            return false;
        }

        if (status_code >= 300 && status_code < 400) {
            auto it = headers.find("location");
            if (it == headers.end()) {
                err = "redirect without location header";
                return false;
            }
            current_url = resolve_redirect_url(u, it->second);
            continue;
        }

        if (status_code != 200) {
            err = "HTTP error " + std::to_string(status_code);
            return false;
        }

        const auto te_it = headers.find("transfer-encoding");
        if (te_it != headers.end() && to_lower_ascii(te_it->second).find("chunked") != std::string::npos) {
            err = "chunked transfer encoding is not supported in this Switch build";
            return false;
        }

        size_t content_length = 0;
        const auto cl_it = headers.find("content-length");
        if (cl_it != headers.end()) {
            content_length = (size_t) std::strtoull(cl_it->second.c_str(), nullptr, 10);
        }

        ensure_parent_dir(destination);
        FILE * fp = std::fopen(destination.c_str(), "wb");
        if (!fp) {
            err = std::string("failed to open output file: ") + std::strerror(errno);
            return false;
        }

        auto close_and_fail = [&](const std::string & e) {
            std::fclose(fp);
            std::remove(destination.c_str());
            err = e;
            return false;
        };

        size_t received = 0;
        const size_t body_offset = header_end + 4;
        if (response_prefix.size() > body_offset) {
            const size_t body_bytes = response_prefix.size() - body_offset;
            if (std::fwrite(response_prefix.data() + body_offset, 1, body_bytes, fp) != body_bytes) {
                return close_and_fail(std::string("file write failed: ") + std::strerror(errno));
            }
            received += body_bytes;
            if (progress_cb && !progress_cb(received, content_length)) {
                return close_and_fail("download cancelled");
            }
        }

        while (content_length == 0 || received < content_length) {
            ssize_t n = conn.read_some(io_buf.data(), io_buf.size(), err);
            if (n < 0) {
                std::fclose(fp);
                std::remove(destination.c_str());
                return false;
            }
            if (n == 0) {
                break;
            }
            const size_t n_bytes = (size_t) n;
            if (std::fwrite(io_buf.data(), 1, n_bytes, fp) != n_bytes) {
                return close_and_fail(std::string("file write failed: ") + std::strerror(errno));
            }
            received += n_bytes;
            if (progress_cb && !progress_cb(received, content_length)) {
                return close_and_fail("download cancelled");
            }
        }

        std::fclose(fp);

        if (content_length > 0 && received != content_length) {
            std::remove(destination.c_str());
            err = "download incomplete (" + std::to_string(received) + "/" + std::to_string(content_length) + ")";
            return false;
        }

        return true;
    }

    err = "too many redirects";
    return false;
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
