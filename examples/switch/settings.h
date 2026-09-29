#pragma once

// App settings kept across launches in sdmc:/switch/llama_cpp_switch/settings.json.
// Portable (no libnx), so the parsing can be tested on the host.
//
// Format, every key optional (// comments allowed):
//   { "cpu_clock": "stock" | "boost" | "overclock", "overclock_cpu_hz": 0 }
// "overclock_cpu_hz" is the CPU step chosen in overclock mode, 0 = highest available.
// Keys this version does not know are kept when the file is written again.

#include "app.h"

#include <cstdint>
#include <string>

constexpr const char * k_settings_path = "sdmc:/switch/llama_cpp_switch/settings.json";

struct app_settings {
    clock_mode cpu_clock        = CLOCK_MODE_BOOST;
    uint32_t   overclock_cpu_hz = 0;
};

// "stock" / "boost" / "overclock" (the settings.json values)
const char * settings_clock_mode_key(clock_mode mode);

// Parses settings.json text. Missing keys keep the values already in `out`; a key with a wrong type or an
// unknown value is skipped and reported in `err` (warnings, `err` is cleared otherwise). Returns false only if
// the text is not a JSON object, `out` is untouched then.
bool settings_parse(const std::string & text, app_settings & out, std::string & err);

// JSON text for `s`. The keys of `existing` (the current file content; may be empty or invalid) that are not
// settings keys are kept.
std::string settings_to_json(const app_settings & s, const std::string & existing);

// A missing or empty file is not an error: load returns true and leaves `out` as it is.
bool settings_load(const std::string & path, app_settings & out, std::string & err);
bool settings_save(const std::string & path, const app_settings & s, std::string & err);
