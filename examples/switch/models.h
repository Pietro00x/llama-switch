#pragma once

#include "app.h"

#include <string>
#include <vector>

constexpr const char * k_model_list_path = "sdmc:/switch/llama_cpp_switch/models.json";

// models.json format:
// { "models": [ { "name", "url", "path", "context", "temperature", "top_p", "top_k",
//                 "system_prompt" (optional), "chat_template" (optional) } ] }
bool write_model_list(const std::string & path, const std::vector<model_entry> & models, std::string & err);

// Loads the list; if the file does not exist it is created from default_models().
bool load_model_list(const std::string & path, std::vector<model_entry> & out, std::string & err);

// Copies the selected model's default ctx/sampling values into state.
void apply_selected_model_defaults(app_state & state);
