#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef __SWITCH__

#include "../../vendor/nlohmann/json.hpp"

#include <switch.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

constexpr const char * k_model_list_path = "sdmc:/switch/llama_cpp_switch/models.json";
constexpr const char * k_default_model_path = "sdmc:/switch/llama_cpp_switch/models/gemma-3-1b-pt-q8_0.gguf";
constexpr const char * k_default_model_url  = "https://huggingface.co/gguf-org/gemma-3-1b-pt-gguf/blob/main/gemma-3-1b-pt-q8_0.gguf";
constexpr int k_ctx_min = 256;
constexpr int k_ctx_max = 8192;

struct model_entry {
    std::string name;
    std::string url;
    std::string path;
    int   default_ctx  = 4096;
    float default_temp = 0.8f;
    float default_top_p = 0.95f;
    int   default_top_k = 40;
};

struct chat_turn {
    std::string user;
    std::string assistant;
};

struct app_state {
    std::vector<model_entry> models;
    size_t selected_model = 0;

    int n_ctx = 4096;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    std::string loaded_model_path;
    std::vector<chat_turn> chat_turns;

    u64 process_total = 0;
    u64 process_used = 0;
    u32 cpu_hz = 0;
    u32 gpu_hz = 0;
    u32 emc_hz = 0;
    u64 last_stats_us = 0;

    std::string status;
    bool quit = false;
};

enum menu_item : size_t {
    MENU_MODEL = 0,
    MENU_DOWNLOAD,
    MENU_CONTEXT,
    MENU_TEMP,
    MENU_TOP_P,
    MENU_TOP_K,
    MENU_LOAD,
    MENU_CHAT,
    MENU_UNLOAD,
    MENU_RELOAD_JSON,
    MENU_EXIT,
    MENU_COUNT
};

bool g_socket_initialized = false;
bool g_ssl_initialized = false;
bool g_clock_initialized = false;
bool g_use_clkrst = false;

u64 now_us() {
    return armTicksToNs(armGetSystemTick()) / 1000ULL;
}

std::string to_lower_ascii(std::string s) {
    for (char & ch : s) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    return s;
}

std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        ++b;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string rc_to_hex(Result rc) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08" PRIx32, rc);
    return std::string(buf);
}

std::string fmt_mb(u64 bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MB", (double) bytes / (1024.0 * 1024.0));
    return std::string(buf);
}

std::string fmt_hz(u32 hz) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MHz", hz / 1000000.0);
    return std::string(buf);
}

int clamp_ctx(int n_ctx) {
    return std::max(k_ctx_min, std::min(k_ctx_max, n_ctx));
}

float clamp_temp(float t) {
    return std::max(0.0f, std::min(2.0f, t));
}

float clamp_top_p(float p) {
    return std::max(0.0f, std::min(1.0f, p));
}

int clamp_top_k(int k) {
    return std::max(1, std::min(200, k));
}

void ensure_parent_dir(const std::string & file_path) {
    size_t slash = file_path.find_last_of('/');
    if (slash == std::string::npos) {
        return;
    }
    const std::string dir = file_path.substr(0, slash);
    if (dir.empty()) {
        return;
    }

    size_t start = 0;
    if (dir.rfind("sdmc:/", 0) == 0) {
        start = 6;
    }

    for (size_t i = start; i < dir.size(); ++i) {
        if (dir[i] != '/') {
            continue;
        }
        std::string sub = dir.substr(0, i);
        if (!sub.empty()) {
            mkdir(sub.c_str(), 0777);
        }
    }
    mkdir(dir.c_str(), 0777);
}

std::vector<model_entry> default_models() {
    return {
        {
            "Gemma 3 1B PT Q8_0",
            k_default_model_url,
            k_default_model_path,
            4096,
            0.8f,
            0.95f,
            40
        }
    };
}

bool write_model_list(const std::string & path, const std::vector<model_entry> & models, std::string & err) {
    json root;
    root["models"] = json::array();
    for (const auto & m : models) {
        json item;
        item["name"] = m.name;
        item["url"] = m.url;
        item["path"] = m.path;
        item["context"] = clamp_ctx(m.default_ctx);
        item["temperature"] = clamp_temp(m.default_temp);
        item["top_p"] = clamp_top_p(m.default_top_p);
        item["top_k"] = clamp_top_k(m.default_top_k);
        root["models"].push_back(item);
    }

    ensure_parent_dir(path);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        err = "failed to write model list: " + path;
        return false;
    }
    out << root.dump(2);
    return true;
}

bool load_model_list(const std::string & path, std::vector<model_entry> & out, std::string & err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out = default_models();
        if (!write_model_list(path, out, err)) {
            return false;
        }
        return true;
    }

    try {
        json root = json::parse(in);
        if (!root.contains("models") || !root["models"].is_array()) {
            err = "invalid models.json: missing models array";
            return false;
        }

        std::vector<model_entry> parsed;
        for (const auto & item : root["models"]) {
            if (!item.is_object()) {
                continue;
            }
            if (!item.contains("name") || !item.contains("url") || !item.contains("path")) {
                continue;
            }

            model_entry m;
            m.name = item["name"].get<std::string>();
            m.url  = item["url"].get<std::string>();
            m.path = item["path"].get<std::string>();

            if (item.contains("context")) {
                m.default_ctx = clamp_ctx(item["context"].get<int>());
            }
            if (item.contains("temperature")) {
                m.default_temp = clamp_temp(item["temperature"].get<float>());
            }
            if (item.contains("top_p")) {
                m.default_top_p = clamp_top_p(item["top_p"].get<float>());
            }
            if (item.contains("top_k")) {
                m.default_top_k = clamp_top_k(item["top_k"].get<int>());
            }

            parsed.push_back(std::move(m));
        }

        if (parsed.empty()) {
            err = "model list is empty";
            return false;
        }

        out = std::move(parsed);
        return true;
    } catch (const std::exception & ex) {
        err = std::string("failed to parse models.json: ") + ex.what();
        return false;
    }
}

bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

void apply_selected_model_defaults(app_state & state) {
    if (state.models.empty()) {
        return;
    }
    const auto & m = state.models[state.selected_model];
    state.n_ctx = clamp_ctx(m.default_ctx);
    state.temperature = clamp_temp(m.default_temp);
    state.top_p = clamp_top_p(m.default_top_p);
    state.top_k = clamp_top_k(m.default_top_k);
}

void unload_model(app_state & state) {
    if (state.sampler) {
        llama_sampler_free(state.sampler);
        state.sampler = nullptr;
    }
    if (state.ctx) {
        llama_free(state.ctx);
        state.ctx = nullptr;
    }
    if (state.model) {
        llama_model_free(state.model);
        state.model = nullptr;
    }
    state.loaded_model_path.clear();
}

bool load_model(app_state & state, std::string & err) {
    if (state.models.empty()) {
        err = "no models available";
        return false;
    }

    const auto & selected = state.models[state.selected_model];
    if (!file_exists(selected.path)) {
        err = "model file not found: " + selected.path;
        return false;
    }

    unload_model(state);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    state.model = llama_model_load_from_file(selected.path.c_str(), model_params);
    if (!state.model) {
        err = "failed to load model";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) clamp_ctx(state.n_ctx);
    ctx_params.n_batch = (uint32_t) std::min(512, clamp_ctx(state.n_ctx));
    ctx_params.no_perf = false;

    state.ctx = llama_init_from_model(state.model, ctx_params);
    if (!state.ctx) {
        unload_model(state);
        err = "failed to create context";
        return false;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    state.sampler = llama_sampler_chain_init(sparams);
    if (!state.sampler) {
        unload_model(state);
        err = "failed to create sampler chain";
        return false;
    }

    llama_sampler_chain_add(state.sampler, llama_sampler_init_top_k(clamp_top_k(state.top_k)));
    llama_sampler_chain_add(state.sampler, llama_sampler_init_top_p(clamp_top_p(state.top_p), 1));
    llama_sampler_chain_add(state.sampler, llama_sampler_init_temp(clamp_temp(state.temperature)));
    llama_sampler_chain_add(state.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    state.loaded_model_path = selected.path;
    return true;
}

std::string normalize_hf_url(std::string url) {
    const std::string blob = "/blob/main/";
    const size_t host_pos = url.find("huggingface.co/");
    const size_t blob_pos = url.find(blob);
    if (host_pos != std::string::npos && blob_pos != std::string::npos) {
        url.replace(blob_pos, blob.size(), "/resolve/main/");
    }
    return url;
}

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

bool download_file(
    const std::string & input_url,
    const std::string & destination,
    std::string & err,
    const std::function<void(size_t, size_t)> & progress_cb) {
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
            if (progress_cb) {
                progress_cb(received, content_length);
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
            if (progress_cb) {
                progress_cb(received, content_length);
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

void init_clock_service() {
    if (g_clock_initialized) {
        return;
    }

    if (hosversionAtLeast(8, 0, 0)) {
        if (R_SUCCEEDED(clkrstInitialize())) {
            g_use_clkrst = true;
            g_clock_initialized = true;
        }
        return;
    }

    if (R_SUCCEEDED(pcvInitialize())) {
        g_use_clkrst = false;
        g_clock_initialized = true;
    }
}

u32 read_clock_hz(PcvModule module, PcvModuleId module_id) {
    if (!g_clock_initialized) {
        return 0;
    }
    if (!g_use_clkrst) {
        u32 hz = 0;
        if (R_SUCCEEDED(pcvGetClockRate(module, &hz))) {
            return hz;
        }
        return 0;
    }

    ClkrstSession session {};
    u32 hz = 0;
    if (R_SUCCEEDED(clkrstOpenSession(&session, module_id, 3))) {
        (void) clkrstGetClockRate(&session, &hz);
        clkrstCloseSession(&session);
    }
    return hz;
}

void update_stats(app_state & state) {
    const u64 t = now_us();
    if (state.last_stats_us != 0 && (t - state.last_stats_us) < 250000) {
        return;
    }
    state.last_stats_us = t;

    u64 total = 0;
    u64 used = 0;
    if (R_SUCCEEDED(svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0))) {
        state.process_total = total;
    }
    if (R_SUCCEEDED(svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0))) {
        state.process_used = used;
    }

    init_clock_service();
    state.cpu_hz = read_clock_hz(PcvModule_CpuBus, PcvModuleId_CpuBus);
    state.gpu_hz = read_clock_hz(PcvModule_GPU, PcvModuleId_GPU);
    state.emc_hz = read_clock_hz(PcvModule_EMC, PcvModuleId_EMC);
}

void shutdown_services() {
    if (g_clock_initialized) {
        if (g_use_clkrst) {
            clkrstExit();
        } else {
            pcvExit();
        }
        g_clock_initialized = false;
    }
    if (g_ssl_initialized) {
        sslExit();
        g_ssl_initialized = false;
    }
    if (g_socket_initialized) {
        socketExit();
        g_socket_initialized = false;
    }
}

std::string clip_line(const std::string & input, size_t max_len = 78) {
    if (input.size() <= max_len) {
        return input;
    }
    return input.substr(0, max_len - 3) + "...";
}

bool show_prompt_keyboard(const std::string & current, std::string & output, std::string & err) {
    SwkbdConfig kbd {};
    Result rc = swkbdCreate(&kbd, 0);
    if (R_FAILED(rc)) {
        err = "swkbdCreate failed: " + rc_to_hex(rc);
        return false;
    }

    swkbdConfigMakePresetDefault(&kbd);
    swkbdConfigSetHeaderText(&kbd, "Chat Prompt");
    swkbdConfigSetGuideText(&kbd, "Enter prompt for the model");
    swkbdConfigSetSubText(&kbd, "Press + to submit");
    swkbdConfigSetOkButtonText(&kbd, "Send");
    swkbdConfigSetStringLenMax(&kbd, 512);
    swkbdConfigSetInitialText(&kbd, current.c_str());

    char buffer[513] = {};
    rc = swkbdShow(&kbd, buffer, sizeof(buffer));
    swkbdClose(&kbd);

    if (R_FAILED(rc)) {
        err = "keyboard canceled or failed: " + rc_to_hex(rc);
        return false;
    }

    output = buffer;
    return true;
}

void render_chat_ui(
    const app_state & state,
    const std::string & draft,
    const std::string & partial_reply,
    bool generating) {
    consoleClear();

    std::printf("llama.cpp Switch Chat\n");
    std::printf("Model: %s\n", state.model ? state.loaded_model_path.c_str() : "(not loaded)");
    std::printf("Ctx=%d Temp=%.2f Top-p=%.2f Top-k=%d\n\n",
        clamp_ctx(state.n_ctx), clamp_temp(state.temperature), clamp_top_p(state.top_p), clamp_top_k(state.top_k));

    std::printf("Controls: X edit prompt, A send, Y clear chat, B back\n");
    std::printf("Prompt: %s\n\n", clip_line(draft).c_str());

    std::printf("Conversation:\n");
    std::vector<std::string> lines;
    lines.reserve(state.chat_turns.size() * 2 + 2);
    for (const auto & turn : state.chat_turns) {
        lines.push_back("U: " + clip_line(turn.user));
        lines.push_back("A: " + clip_line(turn.assistant));
    }
    if (generating || !partial_reply.empty()) {
        lines.push_back("A*: " + clip_line(partial_reply));
    }

    const size_t max_lines = 14;
    const size_t start = lines.size() > max_lines ? lines.size() - max_lines : 0;
    for (size_t i = start; i < lines.size(); ++i) {
        std::printf("%s\n", lines[i].c_str());
    }

    std::printf("\nSwitch stats (always visible)\n");
    std::printf("RAM: %s / %s\n", fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str());
    std::printf("CPU: %s  GPU: %s  EMC: %s\n", fmt_hz(state.cpu_hz).c_str(), fmt_hz(state.gpu_hz).c_str(), fmt_hz(state.emc_hz).c_str());
    std::printf("Status: %s\n", state.status.c_str());
}

std::string build_prompt(const std::vector<chat_turn> & turns, size_t start, const std::string & user_input) {
    std::string prompt;
    for (size_t i = start; i < turns.size(); ++i) {
        prompt += "User: " + turns[i].user + "\n";
        prompt += "Assistant: " + turns[i].assistant + "\n";
    }
    prompt += "User: " + user_input + "\nAssistant:";
    return prompt;
}

bool build_prompt_fit_context(
    const app_state & state,
    const std::string & user_input,
    std::string & prompt_out,
    std::string & err) {
    if (!state.model || !state.ctx) {
        err = "model is not loaded";
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model);
    size_t start = state.chat_turns.size() > 16 ? state.chat_turns.size() - 16 : 0;

    while (start <= state.chat_turns.size()) {
        std::string prompt = build_prompt(state.chat_turns, start, user_input);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt > 0 && n_prompt < clamp_ctx(state.n_ctx) - 32) {
            prompt_out = std::move(prompt);
            return true;
        }
        if (start == state.chat_turns.size()) {
            break;
        }
        ++start;
    }

    err = "prompt too long for current context";
    return false;
}

bool generate_assistant_reply(
    app_state & state,
    const std::string & user_input,
    std::string & assistant_out,
    const std::function<void(const std::string &)> & partial_cb,
    std::string & err) {
    if (!state.model || !state.ctx || !state.sampler) {
        err = "model is not loaded";
        return false;
    }

    std::string prompt;
    if (!build_prompt_fit_context(state, user_input, prompt, err)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        err = "failed to tokenize prompt";
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        err = "failed to tokenize prompt";
        return false;
    }

    llama_memory_t mem = llama_get_memory(state.ctx);
    if (mem) {
        llama_memory_clear(mem, false);
    }
    llama_sampler_reset(state.sampler);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    int n_pos = batch.n_tokens;

    if (llama_model_has_encoder(state.model)) {
        if (llama_encode(state.ctx, batch)) {
            err = "failed to encode prompt";
            return false;
        }
        llama_token decoder_start_token_id = llama_model_decoder_start_token(state.model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
        n_pos = 1;
    }

    if (llama_decode(state.ctx, batch)) {
        err = "failed to evaluate prompt";
        return false;
    }

    assistant_out.clear();
    const int max_predict = std::min(256, clamp_ctx(state.n_ctx) / 2);
    for (int i = 0; i < max_predict && n_pos < clamp_ctx(state.n_ctx) - 1; ++i) {
        llama_token tok = llama_sampler_sample(state.sampler, state.ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }

        char piece[256];
        int n_piece = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (n_piece < 0) {
            err = "failed converting token to text";
            return false;
        }
        assistant_out.append(piece, (size_t) n_piece);
        if (partial_cb && ((i % 4) == 0 || i < 4)) {
            partial_cb(assistant_out);
        }

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(state.ctx, batch)) {
            err = "failed during decoding";
            return false;
        }
        ++n_pos;

        if (assistant_out.size() > 12 && assistant_out.find("\nUser:") != std::string::npos) {
            size_t p = assistant_out.find("\nUser:");
            assistant_out.resize(p);
            break;
        }
    }

    return true;
}

void run_chat_ui(app_state & state, PadState & pad) {
    if (!state.model || !state.ctx || !state.sampler) {
        state.status = "Load a model first";
        return;
    }

    std::string draft_prompt;
    std::string partial_reply;
    bool in_chat = true;

    while (appletMainLoop() && in_chat) {
        update_stats(state);
        render_chat_ui(state, draft_prompt, partial_reply, false);
        consoleUpdate(nullptr);

        padUpdate(&pad);
        const u64 k_down = padGetButtonsDown(&pad);

        if (k_down & HidNpadButton_B) {
            in_chat = false;
            break;
        }

        if (k_down & HidNpadButton_X) {
            std::string entered;
            std::string kbd_err;
            if (show_prompt_keyboard(draft_prompt, entered, kbd_err)) {
                draft_prompt = entered;
                state.status = "Prompt updated";
            } else {
                state.status = kbd_err;
            }
        }

        if (k_down & HidNpadButton_Y) {
            state.chat_turns.clear();
            partial_reply.clear();
            state.status = "Chat history cleared";
        }

        if ((k_down & HidNpadButton_A) && !draft_prompt.empty()) {
            state.status = "Generating...";
            partial_reply.clear();

            render_chat_ui(state, draft_prompt, partial_reply, true);
            consoleUpdate(nullptr);

            std::string assistant;
            std::string gen_err;
            bool ok = generate_assistant_reply(
                state,
                draft_prompt,
                assistant,
                [&](const std::string & partial) {
                    partial_reply = partial;
                    update_stats(state);
                    render_chat_ui(state, draft_prompt, partial_reply, true);
                    consoleUpdate(nullptr);
                },
                gen_err);

            if (ok) {
                state.chat_turns.push_back({ draft_prompt, assistant });
                draft_prompt.clear();
                partial_reply.clear();
                state.status = "Reply complete";
            } else {
                state.status = "Generation failed: " + gen_err;
            }
        }
    }
}

void render_ui(const app_state & state, size_t cursor) {
    consoleClear();

    const model_entry * sel = state.models.empty() ? nullptr : &state.models[state.selected_model];
    const bool model_present = sel && file_exists(sel->path);

    std::printf("llama.cpp Switch Menu\n");
    std::printf("JSON list: %s\n\n", k_model_list_path);

    std::printf("%c Model        : %s\n", cursor == MENU_MODEL ? '>' : ' ', sel ? sel->name.c_str() : "(none)");
    std::printf("%c Download     : %s\n", cursor == MENU_DOWNLOAD ? '>' : ' ', sel ? "A to fetch selected model" : "no model selected");
    std::printf("%c Context      : %d (hard cap %d)\n", cursor == MENU_CONTEXT ? '>' : ' ', clamp_ctx(state.n_ctx), k_ctx_max);
    std::printf("%c Temperature  : %.2f\n", cursor == MENU_TEMP ? '>' : ' ', clamp_temp(state.temperature));
    std::printf("%c Top-p        : %.2f\n", cursor == MENU_TOP_P ? '>' : ' ', clamp_top_p(state.top_p));
    std::printf("%c Top-k        : %d\n", cursor == MENU_TOP_K ? '>' : ' ', clamp_top_k(state.top_k));
    std::printf("%c Load model   : %s\n", cursor == MENU_LOAD ? '>' : ' ', model_present ? "A to load" : "file missing");
    std::printf("%c Chat         : %s\n", cursor == MENU_CHAT ? '>' : ' ', state.model ? "A to open chat UI" : "load model first");
    std::printf("%c Unload model : %s\n", cursor == MENU_UNLOAD ? '>' : ' ', state.model ? "A to unload" : "(not loaded)");
    std::printf("%c Reload JSON  : A to reload model list\n", cursor == MENU_RELOAD_JSON ? '>' : ' ');
    std::printf("%c Exit         : A or +\n", cursor == MENU_EXIT ? '>' : ' ');

    std::printf("\nModel path : %s\n", sel ? sel->path.c_str() : "(none)");
    std::printf("Model URL  : %s\n", sel ? normalize_hf_url(sel->url).c_str() : "(none)");
    std::printf("Loaded     : %s\n", state.model ? state.loaded_model_path.c_str() : "(none)");

    std::printf("\nSwitch stats (always visible)\n");
    std::printf("RAM  : %s / %s\n", fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str());
    std::printf("CPU  : %s\n", fmt_hz(state.cpu_hz).c_str());
    std::printf("GPU  : %s\n", fmt_hz(state.gpu_hz).c_str());
    std::printf("EMC  : %s\n", fmt_hz(state.emc_hz).c_str());

    std::printf("\nStatus: %s\n", state.status.c_str());
    std::printf("Controls: Up/Down navigate, Left/Right adjust, A action, + quit\n");
}

void adjust_selected_item(app_state & state, size_t cursor, int dir) {
    switch (cursor) {
        case MENU_MODEL: {
            if (state.models.empty()) {
                break;
            }
            const size_t n = state.models.size();
            state.selected_model = (state.selected_model + n + (dir > 0 ? 1 : n - 1)) % n;
            apply_selected_model_defaults(state);
            state.status = "Selected model: " + state.models[state.selected_model].name;
        } break;
        case MENU_CONTEXT:
            state.n_ctx = clamp_ctx(state.n_ctx + dir * 256);
            state.status = "Context updated";
            break;
        case MENU_TEMP:
            state.temperature = clamp_temp(state.temperature + dir * 0.05f);
            state.status = "Temperature updated";
            break;
        case MENU_TOP_P:
            state.top_p = clamp_top_p(state.top_p + dir * 0.01f);
            state.status = "Top-p updated";
            break;
        case MENU_TOP_K:
            state.top_k = clamp_top_k(state.top_k + dir);
            state.status = "Top-k updated";
            break;
        default:
            break;
    }
}

void run_action(app_state & state, size_t cursor, PadState & pad) {
    std::string err;

    switch (cursor) {
        case MENU_DOWNLOAD: {
            if (state.models.empty()) {
                state.status = "No model selected";
                break;
            }
            const model_entry & selected = state.models[state.selected_model];
            state.status = "Downloading " + selected.name + "...";

            u64 last_draw = 0;
            auto progress_cb = [&](size_t received, size_t total) {
                const u64 t = now_us();
                if (t - last_draw < 150000) {
                    return;
                }
                last_draw = t;
                if (total > 0) {
                    const double pct = 100.0 * (double) received / (double) total;
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "Downloading %.2f%% (%s / %s)", pct, fmt_mb(received).c_str(), fmt_mb(total).c_str());
                    state.status = buf;
                } else {
                    state.status = "Downloading " + fmt_mb(received);
                }
                update_stats(state);
                render_ui(state, cursor);
                consoleUpdate(NULL);
            };

            if (download_file(selected.url, selected.path, err, progress_cb)) {
                state.status = "Download complete: " + selected.path;
            } else {
                state.status = "Download failed: " + err;
            }
        } break;

        case MENU_LOAD:
            state.status = "Loading model...";
            if (load_model(state, err)) {
                state.status = "Model loaded";
            } else {
                state.status = "Load failed: " + err;
            }
            break;

        case MENU_CHAT:
            run_chat_ui(state, pad);
            break;

        case MENU_UNLOAD:
            unload_model(state);
            state.status = "Model unloaded";
            break;

        case MENU_RELOAD_JSON:
            if (load_model_list(k_model_list_path, state.models, err)) {
                state.selected_model = std::min(state.selected_model, state.models.size() - 1);
                apply_selected_model_defaults(state);
                state.status = "Reloaded model list from JSON";
            } else {
                state.status = "Reload JSON failed: " + err;
            }
            break;

        case MENU_EXIT:
            state.quit = true;
            break;

        default:
            break;
    }
}

} // namespace

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    consoleInit(NULL);

    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);

    llama_backend_init();
    ggml_backend_load_all();

    app_state state;
    std::string err;
    if (!load_model_list(k_model_list_path, state.models, err)) {
        state.models = default_models();
        state.status = "Using fallback model list: " + err;
    } else {
        state.status = "Loaded models from JSON";
    }
    if (!state.models.empty()) {
        apply_selected_model_defaults(state);
    }

    size_t cursor = 0;

    while (appletMainLoop() && !state.quit) {
        update_stats(state);
        render_ui(state, cursor);
        consoleUpdate(NULL);

        padUpdate(&pad);
        const u64 k_down = padGetButtonsDown(&pad);

        if (k_down & HidNpadButton_Up) {
            cursor = (cursor + MENU_COUNT - 1) % MENU_COUNT;
        }
        if (k_down & HidNpadButton_Down) {
            cursor = (cursor + 1) % MENU_COUNT;
        }
        if (k_down & HidNpadButton_Left) {
            adjust_selected_item(state, cursor, -1);
        }
        if (k_down & HidNpadButton_Right) {
            adjust_selected_item(state, cursor, +1);
        }
        if (k_down & HidNpadButton_A) {
            run_action(state, cursor, pad);
        }
        if (k_down & HidNpadButton_Plus) {
            state.quit = true;
        }
    }

    unload_model(state);
    shutdown_services();
    llama_backend_free();

    consoleExit(NULL);
    return 0;
}

#else

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

#endif
