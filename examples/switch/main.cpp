#include "download.h"
#include "engine.h"
#include "catalog.h"
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

    llama_backend_init();
    ggml_backend_load_all();

    app_state state;
    std::string err;
    if (!load_model_list(k_model_list_path, state.models, err)) {
        state.models = default_models();
        state.status = "Using fallback model list: " + err;
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
