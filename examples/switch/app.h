#pragma once

// Shared state for the llama.cpp Switch app.
// Keep this header free of libnx includes so portable parts (util.cpp, generate.cpp)
// can also be compiled on the host.

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
    std::string system_prompt; // optional; empty = none
    std::string chat_template; // optional override; empty = use the template stored in the GGUF
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

    uint64_t process_total = 0;
    uint64_t process_used = 0;
    uint32_t cpu_hz = 0;
    uint32_t gpu_hz = 0;
    uint32_t emc_hz = 0;
    uint64_t last_stats_us = 0;

    std::string status;
    bool quit = false;
};

// util.cpp
std::string to_lower_ascii(std::string s);
std::string trim(const std::string & s);
std::string fmt_mb(uint64_t bytes);
std::string fmt_hz(uint32_t hz);
int   clamp_ctx(int n_ctx);
float clamp_temp(float t);
float clamp_top_p(float p);
int   clamp_top_k(int k);
void  ensure_parent_dir(const std::string & file_path); // mkdir -p of the file's directory
bool  file_exists(const std::string & path);
std::string clip_line(const std::string & input, size_t max_len = 78);
