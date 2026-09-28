#include "download.h"
#include "engine.h"
#include "models.h"
#include "screens.h"
#include "sysinfo.h"

#include "llama.h"

#include <switch.h>

#include <string>

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    consoleInit(NULL);

    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);

    // before llama_backend_init so backend init logs also go to the log file, not the console
    engine_log_init();
    llama_backend_init();
    ggml_backend_load_all();

    app_state state;
    std::string err;
    if (!load_model_list(k_model_list_path, state.models, err)) {
        state.models = fallback_model_list();
        state.status = "Using fallback model list: " + err;
    } else if (!err.empty()) {
        state.status = "Loaded " + std::to_string(state.models.size()) + " models (warning: " + err + ")";
    } else {
        state.status = "Loaded " + std::to_string(state.models.size()) + " models from models.json";
    }
    if (!state.models.empty()) {
        apply_selected_model_defaults(state);
    }

    run_main_menu(state, pad);

    unload_model(state);
    download_exit();
    sysinfo_exit();
    llama_backend_free();

    consoleExit(NULL);
    return 0;
}
