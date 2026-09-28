#pragma once

#include "app.h"

#include <cstdint>
#include <string>
#include <vector>

constexpr const char * k_model_list_path = "sdmc:/switch/llama_cpp_switch/models.json";
constexpr const char * k_models_dir      = "sdmc:/switch/llama_cpp_switch/models";

// models.json format:
// { "models": [ { "name", "url", "path", "context", "temperature", "top_p", "top_k",
//                 "system_prompt" (optional), "chat_template" (optional) } ] }
//
// - "path" is required. "name" falls back to the file name. "url" is optional: an entry without
//   a url can be loaded if the file is on the SD card, but it cannot be downloaded.
// - Numbers may also be given as strings ("4096"). A field with a wrong type is ignored (the
//   default is kept) and an entry without a usable path is skipped; both are reported as
//   warnings in `err`, while the rest of the file is still used. // comments are allowed.
//
// models.json is never rewritten once it exists (only a missing or empty file is created from
// default_models()). Everything added by load_model_list() is kept in memory only:
//   - built-in catalog entries (default_models()) whose path the file does not list
//     (from_catalog = true), so newer catalog models show up without editing the JSON;
//     a top-level "catalog": false in models.json turns this off;
//   - *.gguf files in the models dir that no entry points to (from_scan = true, url empty).
// To keep such an entry, add it to models.json (or pass the list to write_model_list()).
bool write_model_list(const std::string & path, const std::vector<model_entry> & models, std::string & err);

// Loads the model list:
//   1. parses `path`; if it does not exist it is created from default_models(),
//   2. appends missing_catalog_models(),
//   3. refresh_model_files() + scan_model_dir(models_dir),
//   4. sort_model_list().
// Returns false only if the file cannot be read or parsed at all, or the result is empty; `out` is
// left untouched in that case (see fallback_model_list()). On success `err` is cleared or holds
// non-fatal warnings (skipped entries, bad fields, models.json could not be created) that the
// caller may show.
bool load_model_list(const std::string & path, std::vector<model_entry> & out, std::string & err); // uses k_models_dir
bool load_model_list(const std::string & path, const std::string & models_dir, std::vector<model_entry> & out, std::string & err);

// Catalog + scanned files, for when load_model_list() fails (better than default_models() alone,
// because local GGUFs stay visible).
std::vector<model_entry> fallback_model_list(const std::string & models_dir = k_models_dir);

// Like the "Reload JSON" action: reloads k_model_list_path into state.models, keeps the
// selected model (matched by path) and applies its defaults. state is unchanged on failure.
bool reload_model_list(app_state & state, std::string & err);

// Built-in catalog entries whose path is not used by any entry in `models` (from_catalog = true).
std::vector<model_entry> missing_catalog_models(const std::vector<model_entry> & models);

// Appends the *.gguf files in `dir` that no entry in `models` points to (from_scan = true,
// url empty, name and context taken from the GGUF header, context = min(trained ctx, 4096)).
// Skips hidden files, non-first split shards, mmproj/adapter GGUFs and files without a valid
// GGUF header. Returns the number of entries added.
size_t scan_model_dir(const std::string & dir, std::vector<model_entry> & models);

// Updates file_size / on_sd / arch of every entry from the file system. GGUF headers are cached
// by path + size + mtime, so repeated calls only stat() the files.
void refresh_model_files(std::vector<model_entry> & models);

// Stable sort: models present on the SD card first, then the ones that still need a download.
void sort_model_list(std::vector<model_entry> & models);

// Metadata read from a GGUF header. Reading stops at the first tokenizer.* key, so it only touches
// the first few KB of the file (gguf_init_from_file() would parse the whole vocabulary). Keys
// written after the tokenizer are not seen. Returns false if the header is not GGUF, is truncated
// or has no general.architecture.
struct gguf_file_info {
    std::string arch;         // general.architecture
    std::string name;         // general.name
    std::string type;         // general.type ("model", "mmproj", "adapter", ...)
    std::string size_label;   // general.size_label, e.g. "0.5B"
    std::string quant;        // from general.file_type, else guessed from the file name; may be empty
    uint32_t    n_ctx_train = 0; // <arch>.context_length, 0 if unknown
};
bool read_gguf_info(const std::string & path, gguf_file_info & info);

// Copies the selected model's default ctx/sampling values into state.
void apply_selected_model_defaults(app_state & state);
