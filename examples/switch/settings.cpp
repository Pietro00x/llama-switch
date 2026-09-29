#include "settings.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

static const char * const k_key_cpu_clock = "cpu_clock";
static const char * const k_key_oc_cpu_hz = "overclock_cpu_hz";

static const char * const k_clock_mode_keys[CLOCK_MODE_COUNT] = { "stock", "boost", "overclock" };

static void settings_warn(std::string & err, const std::string & msg) {
    if (!err.empty()) {
        err += "; ";
    }
    err += msg;
}

const char * settings_clock_mode_key(clock_mode mode) {
    if ((int) mode < 0 || (int) mode >= CLOCK_MODE_COUNT) {
        return k_clock_mode_keys[CLOCK_MODE_BOOST];
    }
    return k_clock_mode_keys[mode];
}

bool settings_parse(const std::string & text, app_settings & out, std::string & err) {
    err.clear();
    // no exceptions: a broken file only means defaults; comments are allowed like in models.json
    const json root = json::parse(text, nullptr, false, true);
    if (root.is_discarded() || !root.is_object()) {
        err = "not a JSON object";
        return false;
    }

    app_settings s = out;

    if (root.contains(k_key_cpu_clock)) {
        const json & v = root[k_key_cpu_clock];
        bool found = false;
        if (v.is_string()) {
            const std::string key = to_lower_ascii(trim(v.get<std::string>()));
            for (int i = 0; i < CLOCK_MODE_COUNT; ++i) {
                if (key == k_clock_mode_keys[i]) {
                    s.cpu_clock = (clock_mode) i;
                    found = true;
                }
            }
        }
        if (!found) {
            settings_warn(err, std::string("invalid ") + k_key_cpu_clock + " (use stock, boost or overclock)");
        }
    }

    if (root.contains(k_key_oc_cpu_hz)) {
        const json & v = root[k_key_oc_cpu_hz];
        if (v.is_number_unsigned() && v.get<uint64_t>() <= UINT32_MAX) {
            s.overclock_cpu_hz = (uint32_t) v.get<uint64_t>();
        } else {
            settings_warn(err, std::string("invalid ") + k_key_oc_cpu_hz + " (a rate in Hz, 0 = highest)");
        }
    }

    out = s;
    return true;
}

std::string settings_to_json(const app_settings & s, const std::string & existing) {
    json root = json::parse(existing, nullptr, false, true);
    if (root.is_discarded() || !root.is_object()) {
        root = json::object();
    }
    root[k_key_cpu_clock] = settings_clock_mode_key(s.cpu_clock);
    root[k_key_oc_cpu_hz] = s.overclock_cpu_hz;
    // replace: never throw on strings that are not valid UTF-8 (the parser has already rejected them anyway)
    return root.dump(2, ' ', false, json::error_handler_t::replace) + "\n";
}

static bool settings_read_file(const std::string & path, std::string & text, bool & exists) {
    text.clear();
    std::ifstream in(path, std::ios::binary);
    exists = (bool) in;
    if (!in) {
        return true;
    }
    std::ostringstream ss;
    ss << in.rdbuf(); // an empty file only sets failbit on ss
    if (in.bad()) {
        return false;
    }
    text = ss.str();
    return true;
}

// settings_save writes the new content here first, then replaces the file with it
static std::string settings_tmp_path(const std::string & path) {
    return path + ".tmp";
}

// the file, or the complete new copy settings_save left behind if it stopped between removing and renaming
static bool settings_read_current(const std::string & path, std::string & text, bool & exists) {
    if (!settings_read_file(path, text, exists)) {
        return false;
    }
    if (!exists) {
        return settings_read_file(settings_tmp_path(path), text, exists);
    }
    return true;
}

bool settings_load(const std::string & path, app_settings & out, std::string & err) {
    err.clear();
    std::string text;
    bool exists = false;
    if (!settings_read_current(path, text, exists)) {
        err = "cannot read " + path;
        return false;
    }
    if (!exists || trim(text).empty()) {
        return true;
    }
    return settings_parse(text, out, err);
}

bool settings_save(const std::string & path, const app_settings & s, std::string & err) {
    err.clear();
    std::string existing;
    bool exists = false;
    settings_read_current(path, existing, exists); // unreadable = start from an empty object

    const std::string text = settings_to_json(s, existing);
    const std::string tmp  = settings_tmp_path(path);

    // never truncate the only copy: write the new one completely, then swap it in
    ensure_parent_dir(path);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            err = "cannot write " + tmp;
            return false;
        }
        out << text;
        out.flush();
        if (!out) {
            out.close();
            std::remove(tmp.c_str());
            err = "cannot write " + tmp;
            return false;
        }
    }
    // FAT (the SD card) cannot rename over an existing file
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "cannot rename " + tmp;
        return false;
    }
    return true;
}
