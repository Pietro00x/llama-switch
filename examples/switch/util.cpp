#include "app.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>

#include <sys/stat.h>

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

std::string fmt_mb(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MB", (double) bytes / (1024.0 * 1024.0));
    return std::string(buf);
}

std::string fmt_hz(uint32_t hz) {
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

bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

std::string clip_line(const std::string & input, size_t max_len) {
    if (input.size() <= max_len) {
        return input;
    }
    return input.substr(0, max_len - 3) + "...";
}
