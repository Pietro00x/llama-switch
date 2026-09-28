#include "catalog.h"

// Built-in model catalog for the Switch (Tegra X1: 4x Cortex-A57, ~3 cores usable, 4 GB LPDDR4 shared).
//
// Memory budget
// -------------
// Launched through a title override, the app gets ~3.2 GB. Applet (album) mode gets only ~400 MB,
// which no model in this list fits into. Weights are fully resident, so every entry has to satisfy
//
//     file size + KV cache (+ recurrent state) + compute buffer + ~100 MB app overhead <= ~3 GB
//
// KV bytes per token = n_attn_layers * n_head_kv * head_dim * 2 (K and V) * 2 bytes (F16).
// The hybrid architectures (LFM2 conv+attention, Qwen3.5 Gated DeltaNet+attention and Granite 4.0-H
// Mamba2+attention) keep KV only in their few attention layers. Their recurrent state is fixed-size,
// so a larger context costs them almost nothing. Plain transformers (Qwen3, SmolLM3) pay the full price.
// Gemma 3 is iSWA, but llama.cpp allocates a full-size SWA cache by default, so all 26 layers count.
//
// The compute buffer is dominated by the logits of one micro-batch: n_ubatch (512) * n_vocab * 4 bytes.
// That means models with a large vocabulary (Gemma 3: 262k, Qwen3.5: 248k) need ~0.5 GB of compute
// buffer, even though their KV cache is small.
//
// Quant choice on the A57
// -----------------------
// ARMv8.0 has NEON but no dotprod, i8mm or fp16 arithmetic. As a result, none of the ggml-cpu repack
// fast paths (Q4_0 4x4/4x8, Q4_K 8x4/8x8, Q8_0 4x4, IQ4_NL 4x4) are enabled, see repack.cpp. Every
// quant goes through the generic NEON vec_dot with an emulated vdotq_s32 (vmull + vpadal).
//   - Q4_0:   the cheapest 4-bit decode and the smallest file. Lower quality with plain post-training
//             quantization, but LiquidAI ships a quantization-aware-distilled (QAD) Q4_0 build.
//   - Q4_K_M: roughly 5% bigger than Q4_0 and a bit more unpacking work per block, with clearly
//             better quality. This is the default for models of about 1B parameters and up.
//   - Q8_0:   the simplest kernel but twice the bytes per weight. It is used only for models of
//             0.8B and under, where 4-bit hurts quality the most and the file stays small.
// Token generation on the A57 is limited by both memory bandwidth and int8 multiply throughput,
// so fewer bytes per weight is the main speed lever.
//
// Entries
// -------
// File sizes are the exact Hugging Face sizes. KV and compute are MiB at the default context.
// "m" = measured on the host with llama_memory_breakdown_print (same CPU graph as on the Switch),
// "e" = estimated from config.json.
//  #  model                         arch           file     ctx   KV/tok   KV+state  compute  total
//  1  LFM2.5 1.2B Instruct QAD Q4_0 lfm2           696 MB   4096  12 KiB     48 m     136 m   ~0.9 GB
//  2  Qwen3 1.7B Q4_K_M             qwen3          1.1 GB   4096  112 KiB   448 e     310 e   ~1.9 GB
//  3  Gemma 3 1B IT Q4_K_M          gemma3         806 MB   4096  26 KiB    104 m     516 m   ~1.4 GB
//  4  Qwen3.5 0.8B Q8_0             qwen35         812 MB   4096  12 KiB     67 m     489 m   ~1.4 GB
//  5  Qwen3.5 2B Q4_K_M             qwen35         1.3 GB   4096  12 KiB     67 e     500 e   ~1.9 GB
//  6  Granite 4.0 H 1B Q4_K_M       granitehybrid  901 MB   4096  8 KiB      87 m     213 m   ~1.2 GB
//  7  Qwen3 0.6B Q8_0               qwen3          639 MB   4096  112 KiB   448 m     308 m   ~1.4 GB
//  8  LFM2.5 350M Q8_0              lfm2           379 MB   4096  12 KiB     48 e     136 e   ~0.6 GB
//  9  SmolLM3 3B Q4_K_M             smollm3        1.9 GB   2048  72 KiB    144 e     260 e   ~2.4 GB
//
// Why these models:
//   1. The best default: fast (10 conv + 6 attention layers) and a good chat model, with a QAD-tuned Q4_0.
//   2. The strongest reasoning and multilingual model that stays comfortably inside the budget.
//   3. Google's instruct model, good at writing.
//   4. The newest Qwen generation: 18 DeltaNet + 6 attention layers, non-thinking by default.
//   5. A bigger Qwen3.5 with the same tiny KV cache, but slower.
//   6. IBM's hybrid model (36 Mamba2 + 4 attention layers).
//   7. A small, quick fallback from the official Qwen GGUF repo.
//   8. The smallest and fastest; good for extraction and short answers.
//   9. The largest model that fits. It is slow, and SmolLM3 thinks by default.
//
// Every GGUF above carries a chat template that llm_chat_detect_template() maps to a built-in one:
// chatml (LFM2.5, Qwen3, Qwen3.5, SmolLM3), gemma and granite. So chat_template stays empty.
//
// Models considered and left out:
//   - Gemma 3n E2B: the smallest Q4 is ~3.0 GB and its per-layer embeddings must stay resident.
//   - Phi-4-mini (3.8B): Q4_K_M is ~2.5 GB plus 128 KiB/token of KV (512 MiB at 4k). That is over
//     budget, and it would run at ~1 tok/s.
//   - Qwen3-4B / Qwen3.5-4B: 2.5-2.7 GB at Q4. That leaves no headroom and they are too slow.
//   - Llama 3.2 1B/3B: older (2024) and outperformed at the same size by the entries above.
//   - google/gemma-3-1b-it-qat-q4_0-gguf: gated repo (HTTP 401 without a token), so it can't be downloaded.
//
// Sampling defaults follow each model card. The app's sampler chain has no repetition or presence
// penalty, so the LFM2.5 temperature is raised from the card's 0.1 to 0.3 to avoid greedy loops.
// "/no_think" is the documented Qwen3/SmolLM3 system-prompt switch that skips the <think> phase.
// On an A57 that phase can cost hundreds of tokens (minutes) before the first visible word.
// The switch only holds across turns if earlier <think>...</think> blocks are dropped from the history,
// which is what the official template does. Qwen3.5 has no /no_think switch. With the built-in chatml
// template and these sampling settings, it answers without thinking (it emits an empty <think></think>),
// but it can still start thinking at a higher temperature.

static constexpr const char * k_models_dir = "sdmc:/switch/llama_cpp_switch/models/";

static model_entry make_entry(const char * name, const char * repo, const char * file, int ctx,
                              float temp, float top_p, int top_k, const char * system_prompt) {
    model_entry m;
    m.name          = name;
    m.url           = std::string("https://huggingface.co/") + repo + "/resolve/main/" + file;
    m.path          = std::string(k_models_dir) + file;
    m.default_ctx   = ctx;
    m.default_temp  = temp;
    m.default_top_p = top_p;
    m.default_top_k = top_k;
    m.system_prompt = system_prompt;
    return m;
}

std::vector<model_entry> default_models() {
    return {
        make_entry("LFM2.5 1.2B Instruct Q4_0 (696 MB)", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
                   "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf", 4096, 0.3f, 0.95f, 50, ""),
        make_entry("Qwen3 1.7B Q4_K_M (1.1 GB)", "unsloth/Qwen3-1.7B-GGUF",
                   "Qwen3-1.7B-Q4_K_M.gguf", 4096, 0.7f, 0.8f, 20, "/no_think"),
        make_entry("Gemma 3 1B IT Q4_K_M (806 MB)", "ggml-org/gemma-3-1b-it-GGUF",
                   "gemma-3-1b-it-Q4_K_M.gguf", 4096, 1.0f, 0.95f, 64, ""),
        make_entry("Qwen3.5 0.8B Q8_0 (812 MB)", "unsloth/Qwen3.5-0.8B-GGUF",
                   "Qwen3.5-0.8B-Q8_0.gguf", 4096, 0.7f, 0.8f, 20, ""),
        make_entry("Qwen3.5 2B Q4_K_M (1.3 GB)", "unsloth/Qwen3.5-2B-GGUF",
                   "Qwen3.5-2B-Q4_K_M.gguf", 4096, 0.7f, 0.8f, 20, ""),
        make_entry("Granite 4.0 H 1B Q4_K_M (901 MB)", "ibm-granite/granite-4.0-h-1b-GGUF",
                   "granite-4.0-h-1b-Q4_K_M.gguf", 4096, 0.3f, 0.95f, 40, ""),
        make_entry("Qwen3 0.6B Q8_0 (639 MB)", "Qwen/Qwen3-0.6B-GGUF",
                   "Qwen3-0.6B-Q8_0.gguf", 4096, 0.7f, 0.8f, 20, "/no_think"),
        make_entry("LFM2.5 350M Q8_0 (379 MB)", "LiquidAI/LFM2.5-350M-GGUF",
                   "LFM2.5-350M-Q8_0.gguf", 4096, 0.3f, 0.95f, 50, ""),
        make_entry("SmolLM3 3B Q4_K_M (1.9 GB)", "unsloth/SmolLM3-3B-GGUF",
                   "SmolLM3-3B-Q4_K_M.gguf", 2048, 0.6f, 0.95f, 40, "/no_think"),
    };
}
