#include "catalog.h"

static constexpr const char * k_default_model_path = "sdmc:/switch/llama_cpp_switch/models/gemma-3-1b-pt-q8_0.gguf";
static constexpr const char * k_default_model_url  = "https://huggingface.co/gguf-org/gemma-3-1b-pt-gguf/blob/main/gemma-3-1b-pt-q8_0.gguf";

std::vector<model_entry> default_models() {
    return {
        {
            "Gemma 3 1B PT Q8_0",
            k_default_model_url,
            k_default_model_path,
            4096,
            0.8f,
            0.95f,
            40,
            "",
            ""
        }
    };
}
