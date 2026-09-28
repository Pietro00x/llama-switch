#pragma once

#include "app.h"

#include <string>

// Loads the selected model from state.models into state.model/ctx/sampler (CPU only).
bool load_model(app_state & state, std::string & err);

// Frees sampler, context and model. Safe to call when nothing is loaded.
void unload_model(app_state & state);
