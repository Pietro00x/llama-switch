#include "models.h"

#include "catalog.h"

#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <dirent.h>
#include <sys/stat.h>

using json = nlohmann::json;

// context given to models found by scan_model_dir(), capped by the trained context
constexpr int k_scan_ctx_default = 4096;

//
// helpers
//

static void append_warning(std::string & warn, const std::string & msg) {
    if (!warn.empty()) {
        warn += "; ";
    }
    warn += msg;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool contains_ci(const std::string & hay, const std::string & needle) {
    return to_lower_ascii(hay).find(to_lower_ascii(needle)) != std::string::npos;
}

static std::string file_name_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string file_stem_of(const std::string & path) {
    std::string name = file_name_of(path);
    if (ends_with(to_lower_ascii(name), ".gguf")) {
        name.resize(name.size() - 5);
    }
    return name;
}

// key used to compare paths: case-insensitive (FAT), "sdmc:" prefix optional, repeated slashes collapsed
static std::string normalize_model_path(const std::string & path) {
    std::string s = to_lower_ascii(trim(path));
    if (s.rfind("sdmc:", 0) == 0) {
        s = s.substr(5);
    }
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        if (ch == '\\') {
            ch = '/';
        }
        if (ch == '/' && !out.empty() && out.back() == '/') {
            continue;
        }
        out.push_back(ch);
    }
    return out;
}

static bool path_in_list(const std::vector<std::string> & list, const std::string & normalized) {
    for (const auto & p : list) {
        if (p == normalized) {
            return true;
        }
    }
    return false;
}

static std::vector<std::string> normalized_paths(const std::vector<model_entry> & models) {
    std::vector<std::string> out;
    out.reserve(models.size());
    for (const auto & m : models) {
        out.push_back(normalize_model_path(m.path));
    }
    return out;
}

static bool stat_regular_file(const std::string & path, uint64_t & size, int64_t & mtime) {
    struct stat st;
    if (path.empty() || stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }
    size  = (uint64_t) st.st_size;
    mtime = (int64_t) st.st_mtime;
    return true;
}

//
// GGUF header
//

struct ftype_name_entry {
    int          ftype;
    const char * name;
};

static const ftype_name_entry k_ftype_names[] = {
    { LLAMA_FTYPE_ALL_F32,          "F32"     },
    { LLAMA_FTYPE_MOSTLY_F16,       "F16"     },
    { LLAMA_FTYPE_MOSTLY_BF16,      "BF16"    },
    { LLAMA_FTYPE_MOSTLY_Q4_0,      "Q4_0"    },
    { LLAMA_FTYPE_MOSTLY_Q4_1,      "Q4_1"    },
    { LLAMA_FTYPE_MOSTLY_Q5_0,      "Q5_0"    },
    { LLAMA_FTYPE_MOSTLY_Q5_1,      "Q5_1"    },
    { LLAMA_FTYPE_MOSTLY_Q8_0,      "Q8_0"    },
    { LLAMA_FTYPE_MOSTLY_Q2_K,      "Q2_K"    },
    { LLAMA_FTYPE_MOSTLY_Q2_K_S,    "Q2_K_S"  },
    { LLAMA_FTYPE_MOSTLY_Q3_K_S,    "Q3_K_S"  },
    { LLAMA_FTYPE_MOSTLY_Q3_K_M,    "Q3_K_M"  },
    { LLAMA_FTYPE_MOSTLY_Q3_K_L,    "Q3_K_L"  },
    { LLAMA_FTYPE_MOSTLY_Q4_K_S,    "Q4_K_S"  },
    { LLAMA_FTYPE_MOSTLY_Q4_K_M,    "Q4_K_M"  },
    { LLAMA_FTYPE_MOSTLY_Q5_K_S,    "Q5_K_S"  },
    { LLAMA_FTYPE_MOSTLY_Q5_K_M,    "Q5_K_M"  },
    { LLAMA_FTYPE_MOSTLY_Q6_K,      "Q6_K"    },
    { LLAMA_FTYPE_MOSTLY_IQ1_S,     "IQ1_S"   },
    { LLAMA_FTYPE_MOSTLY_IQ1_M,     "IQ1_M"   },
    { LLAMA_FTYPE_MOSTLY_IQ2_XXS,   "IQ2_XXS" },
    { LLAMA_FTYPE_MOSTLY_IQ2_XS,    "IQ2_XS"  },
    { LLAMA_FTYPE_MOSTLY_IQ2_S,     "IQ2_S"   },
    { LLAMA_FTYPE_MOSTLY_IQ2_M,     "IQ2_M"   },
    { LLAMA_FTYPE_MOSTLY_IQ3_XXS,   "IQ3_XXS" },
    { LLAMA_FTYPE_MOSTLY_IQ3_XS,    "IQ3_XS"  },
    { LLAMA_FTYPE_MOSTLY_IQ3_S,     "IQ3_S"   },
    { LLAMA_FTYPE_MOSTLY_IQ3_M,     "IQ3_M"   },
    { LLAMA_FTYPE_MOSTLY_IQ4_NL,    "IQ4_NL"  },
    { LLAMA_FTYPE_MOSTLY_IQ4_XS,    "IQ4_XS"  },
    { LLAMA_FTYPE_MOSTLY_TQ1_0,     "TQ1_0"   },
    { LLAMA_FTYPE_MOSTLY_TQ2_0,     "TQ2_0"   },
    { LLAMA_FTYPE_MOSTLY_MXFP4_MOE, "MXFP4"   },
};

static std::string ftype_name(int64_t ftype) {
    for (const auto & e : k_ftype_names) {
        if (e.ftype == ftype) {
            return e.name;
        }
    }
    return "";
}

// longest quant name contained in the file name, e.g. "qwen2.5-0.5b-instruct-q4_k_m.gguf" -> "Q4_K_M"
static std::string quant_from_file_name(const std::string & path) {
    const std::string stem = to_lower_ascii(file_stem_of(path));
    std::string best;
    for (const auto & e : k_ftype_names) {
        const std::string name = e.name;
        if (name.size() > best.size() && stem.find(to_lower_ascii(name)) != std::string::npos) {
            best = name;
        }
    }
    return best;
}

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static bool gguf_type_is_int(uint32_t type) {
    return gguf_scalar_size(type) > 0 && type != GGUF_TYPE_BOOL && type != GGUF_TYPE_FLOAT32 && type != GGUF_TYPE_FLOAT64;
}

static bool gguf_read_raw(FILE * fp, void * dst, size_t n) {
    return std::fread(dst, 1, n, fp) == n;
}

static bool gguf_skip(FILE * fp, uint64_t n) {
    // small skips are read through the stdio buffer: an fseek may drop the buffer and cost a new SD read
    if (n <= 64 * 1024) {
        char scratch[1024];
        while (n > 0) {
            const size_t chunk = n < sizeof(scratch) ? (size_t) n : sizeof(scratch);
            if (!gguf_read_raw(fp, scratch, chunk)) {
                return false;
            }
            n -= chunk;
        }
        return true;
    }
    if (n > (uint64_t) LONG_MAX) {
        return false;
    }
    return std::fseek(fp, (long) n, SEEK_CUR) == 0;
}

static bool gguf_read_u32(FILE * fp, uint32_t & v) {
    return gguf_read_raw(fp, &v, sizeof(v));
}

static bool gguf_read_u64(FILE * fp, uint64_t & v) {
    return gguf_read_raw(fp, &v, sizeof(v));
}

// reads a gguf string; strings longer than max_len are skipped and returned empty
static bool gguf_read_str(FILE * fp, std::string & s, uint64_t max_len) {
    uint64_t len = 0;
    if (!gguf_read_u64(fp, len)) {
        return false;
    }
    if (len > max_len) {
        s.clear();
        return gguf_skip(fp, len);
    }
    s.resize((size_t) len);
    return len == 0 || gguf_read_raw(fp, &s[0], (size_t) len);
}

static bool gguf_read_int(FILE * fp, uint32_t type, int64_t & v) {
    const size_t n = gguf_scalar_size(type);
    uint8_t buf[8] = {};
    if (n == 0 || n > sizeof(buf) || !gguf_read_raw(fp, buf, n)) {
        return false;
    }
    switch (type) {
        case GGUF_TYPE_UINT8:  { uint8_t  x; std::memcpy(&x, buf, n); v = x; } break;
        case GGUF_TYPE_INT8:   { int8_t   x; std::memcpy(&x, buf, n); v = x; } break;
        case GGUF_TYPE_UINT16: { uint16_t x; std::memcpy(&x, buf, n); v = x; } break;
        case GGUF_TYPE_INT16:  { int16_t  x; std::memcpy(&x, buf, n); v = x; } break;
        case GGUF_TYPE_UINT32: { uint32_t x; std::memcpy(&x, buf, n); v = x; } break;
        case GGUF_TYPE_INT32:  { int32_t  x; std::memcpy(&x, buf, n); v = x; } break;
        case GGUF_TYPE_UINT64: { uint64_t x; std::memcpy(&x, buf, n); v = x > (uint64_t) INT64_MAX ? INT64_MAX : (int64_t) x; } break;
        case GGUF_TYPE_INT64:  { int64_t  x; std::memcpy(&x, buf, n); v = x; } break;
        default: return false;
    }
    return true;
}

static bool gguf_skip_value(FILE * fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t len = 0;
        return gguf_read_u64(fp, len) && gguf_skip(fp, len);
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t n = 0;
        if (!gguf_read_u32(fp, elem_type) || !gguf_read_u64(fp, n)) {
            return false;
        }
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; ++i) {
                if (!gguf_skip_value(fp, GGUF_TYPE_STRING)) {
                    return false;
                }
            }
            return true;
        }
        const size_t elem_size = gguf_scalar_size(elem_type);
        if (elem_size == 0 || n > ((uint64_t) 1 << 40)) {
            return false; // nested arrays are not valid GGUF
        }
        return gguf_skip(fp, n * elem_size);
    }
    const size_t n = gguf_scalar_size(type);
    return n > 0 && gguf_skip(fp, n);
}

static bool read_gguf_info_fp(FILE * fp, const std::string & path, gguf_file_info & info) {
    char magic[4];
    if (!gguf_read_raw(fp, magic, sizeof(magic)) || std::memcmp(magic, GGUF_MAGIC, sizeof(magic)) != 0) {
        return false;
    }
    uint32_t version   = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv      = 0;
    if (!gguf_read_u32(fp, version) || version < 2 || version > GGUF_VERSION ||
        !gguf_read_u64(fp, n_tensors) || !gguf_read_u64(fp, n_kv) || n_kv > (1u << 20)) {
        return false;
    }

    std::string ctx_arch;          // <arch> of the last "<arch>.context_length" seen
    int64_t     ctx_value = 0;
    int64_t     file_type = -1;
    bool        ok        = true;

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t    type = 0;
        if (!gguf_read_str(fp, key, 1024) || key.empty() || !gguf_read_u32(fp, type)) {
            ok = false; // truncated or corrupt
            break;
        }

        // hparams are written before the tokenizer: stop before the (large) vocab arrays. Keys after them
        // (e.g. general.file_type in re-quantized files) are not read; the quant is then guessed from the name.
        if (key.rfind("tokenizer.", 0) == 0 && !info.arch.empty()) {
            break;
        }

        if (type == GGUF_TYPE_STRING && key == "general.architecture") {
            ok = gguf_read_str(fp, info.arch, 256);
        } else if (type == GGUF_TYPE_STRING && key == "general.name") {
            ok = gguf_read_str(fp, info.name, 256);
        } else if (type == GGUF_TYPE_STRING && key == "general.type") {
            ok = gguf_read_str(fp, info.type, 256);
        } else if (type == GGUF_TYPE_STRING && key == "general.size_label") {
            ok = gguf_read_str(fp, info.size_label, 64);
        } else if (gguf_type_is_int(type) && key == "general.file_type") {
            ok = gguf_read_int(fp, type, file_type);
        } else if (gguf_type_is_int(type) && ends_with(key, ".context_length") &&
                   key.find('.') == key.size() - 15 &&
                   (info.arch.empty() || key.compare(0, key.size() - 15, info.arch) == 0)) {
            ctx_arch = key.substr(0, key.size() - 15);
            ok = gguf_read_int(fp, type, ctx_value);
        } else {
            ok = gguf_skip_value(fp, type);
        }
        if (!ok) {
            break;
        }
    }

    if (!info.arch.empty() && ctx_arch == info.arch && ctx_value > 0) {
        info.n_ctx_train = ctx_value > (int64_t) UINT32_MAX ? UINT32_MAX : (uint32_t) ctx_value;
    }
    info.quant = file_type >= 0 ? ftype_name(file_type) : "";
    if (info.quant.empty()) {
        info.quant = quant_from_file_name(path);
    }
    return ok && !info.arch.empty();
}

bool read_gguf_info(const std::string & path, gguf_file_info & info) {
    info = gguf_file_info();
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        return false;
    }
    // the interesting keys are in the first few KB: a single read on the SD card
    std::setvbuf(fp, nullptr, _IOFBF, 16 * 1024);
    const bool ok = read_gguf_info_fp(fp, path, info);
    std::fclose(fp);
    return ok;
}

struct gguf_info_cache_entry {
    std::string    path;
    uint64_t       size  = 0;
    int64_t        mtime = 0;
    bool           ok    = false;
    gguf_file_info info;
};

static std::vector<gguf_info_cache_entry> g_gguf_info_cache;

static bool read_gguf_info_cached(const std::string & path, uint64_t size, int64_t mtime, gguf_file_info & info) {
    for (auto & e : g_gguf_info_cache) {
        if (e.path != path) {
            continue;
        }
        if (e.size != size || e.mtime != mtime) {
            e.size  = size;
            e.mtime = mtime;
            e.ok    = read_gguf_info(path, e.info);
        }
        info = e.info;
        return e.ok;
    }
    gguf_info_cache_entry e;
    e.path  = path;
    e.size  = size;
    e.mtime = mtime;
    e.ok    = read_gguf_info(path, e.info);
    info    = e.info;
    g_gguf_info_cache.push_back(e);
    return e.ok;
}

// true if the name already carries a parameter count such as "1B", "0.5b" or "270M"
static bool has_size_token(const std::string & name) {
    for (size_t i = 1; i < name.size(); ++i) {
        const char c = (char) std::tolower((unsigned char) name[i]);
        if ((c == 'b' || c == 'm') && std::isdigit((unsigned char) name[i - 1]) &&
            (i + 1 == name.size() || !std::isalpha((unsigned char) name[i + 1]))) {
            return true;
        }
    }
    return false;
}

static std::string display_name_for(const gguf_file_info & info, const std::string & path) {
    std::string name = trim(info.name);
    if (name.empty()) {
        return file_stem_of(path);
    }
    const std::string size_label = trim(info.size_label);
    if (!size_label.empty() && !contains_ci(name, size_label) && !has_size_token(name)) {
        name += " " + size_label;
    }
    if (!info.quant.empty() && !contains_ci(name, info.quant)) {
        name += " " + info.quant;
    }
    return name;
}

// "<name>-00002-of-00003" (llama split naming): only the first shard is loadable
static bool is_extra_split_shard(const std::string & file_name) {
    const std::string stem = file_stem_of(file_name);
    if (stem.size() < 15) {
        return false;
    }
    const std::string tail = stem.substr(stem.size() - 15); // "-NNNNN-of-NNNNN"
    if (tail[0] != '-' || tail.compare(6, 4, "-of-") != 0) {
        return false;
    }
    for (size_t i = 1; i < 15; ++i) {
        if ((i < 6 || i >= 10) && (tail[i] < '0' || tail[i] > '9')) {
            return false;
        }
    }
    return tail.compare(1, 5, "00001") != 0;
}

//
// models.json
//

static bool json_get_number(const json & v, double & out) {
    if (v.is_number()) {
        out = v.get<double>();
        return std::isfinite(out);
    }
    if (v.is_string()) {
        const std::string s = trim(v.get<std::string>());
        if (s.empty()) {
            return false;
        }
        char * end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0' || !std::isfinite(d)) {
            return false;
        }
        out = d;
        return true;
    }
    return false;
}

// optional fields: absent or null keeps the default, a wrong type is reported and ignored
static void parse_int_field(const json & item, const char * key, int & dst, const std::string & where, std::string & warn) {
    const auto it = item.find(key);
    if (it == item.end() || it->is_null()) {
        return;
    }
    double v = 0.0;
    if (!json_get_number(*it, v)) {
        append_warning(warn, where + ": bad \"" + key + "\" ignored");
        return;
    }
    v = std::max(-1e9, std::min(1e9, v));
    dst = (int) std::lround(v);
}

static void parse_float_field(const json & item, const char * key, float & dst, const std::string & where, std::string & warn) {
    const auto it = item.find(key);
    if (it == item.end() || it->is_null()) {
        return;
    }
    double v = 0.0;
    if (!json_get_number(*it, v)) {
        append_warning(warn, where + ": bad \"" + key + "\" ignored");
        return;
    }
    v = std::max(-1e9, std::min(1e9, v));
    dst = (float) v;
}

static void parse_string_field(const json & item, const char * key, std::string & dst, const std::string & where, std::string & warn) {
    const auto it = item.find(key);
    if (it == item.end() || it->is_null()) {
        return;
    }
    if (!it->is_string()) {
        append_warning(warn, where + ": bad \"" + key + "\" ignored");
        return;
    }
    dst = it->get<std::string>();
}

static void parse_model_entries(const json & models, std::vector<model_entry> & out, std::string & warn) {
    size_t index = 0;
    for (const auto & item : models) {
        ++index;
        const std::string where_idx = "entry " + std::to_string(index);
        if (!item.is_object()) {
            append_warning(warn, where_idx + " skipped: not an object");
            continue;
        }

        model_entry m;
        parse_string_field(item, "path", m.path, where_idx, warn);
        m.path = trim(m.path);
        if (m.path.empty()) {
            append_warning(warn, where_idx + " skipped: no \"path\"");
            continue;
        }

        parse_string_field(item, "name", m.name, where_idx, warn);
        m.name = trim(m.name);
        if (m.name.empty()) {
            m.name = file_stem_of(m.path);
        }
        const std::string where = "'" + m.name + "'";

        parse_string_field(item, "url", m.url, where, warn);
        m.url = trim(m.url);

        parse_int_field  (item, "context",     m.default_ctx,   where, warn);
        parse_float_field(item, "temperature", m.default_temp,  where, warn);
        parse_float_field(item, "top_p",       m.default_top_p, where, warn);
        parse_int_field  (item, "top_k",       m.default_top_k, where, warn);
        parse_string_field(item, "system_prompt", m.system_prompt, where, warn);
        parse_string_field(item, "chat_template", m.chat_template, where, warn);

        m.default_ctx   = clamp_ctx(m.default_ctx);
        m.default_temp  = clamp_temp(m.default_temp);
        m.default_top_p = clamp_top_p(m.default_top_p);
        m.default_top_k = clamp_top_k(m.default_top_k);

        out.push_back(std::move(m));
    }
}

bool write_model_list(const std::string & path, const std::vector<model_entry> & models, std::string & err) {
    json root;
    root["models"] = json::array();
    for (const auto & m : models) {
        json item;
        item["name"] = m.name;
        item["url"] = m.url;
        item["path"] = m.path;
        item["context"] = clamp_ctx(m.default_ctx);
        item["temperature"] = clamp_temp(m.default_temp);
        item["top_p"] = clamp_top_p(m.default_top_p);
        item["top_k"] = clamp_top_k(m.default_top_k);
        if (!m.system_prompt.empty()) {
            item["system_prompt"] = m.system_prompt;
        }
        if (!m.chat_template.empty()) {
            item["chat_template"] = m.chat_template;
        }
        root["models"].push_back(item);
    }

    ensure_parent_dir(path);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        err = "failed to write model list: " + path;
        return false;
    }
    out << root.dump(2);
    out.flush();
    if (!out) {
        err = "failed to write model list: " + path;
        return false;
    }
    return true;
}

std::vector<model_entry> missing_catalog_models(const std::vector<model_entry> & models) {
    std::vector<std::string> known = normalized_paths(models);
    std::vector<model_entry> result;
    for (auto & m : default_models()) {
        const std::string key = normalize_model_path(m.path);
        if (key.empty() || path_in_list(known, key)) {
            continue;
        }
        m.from_catalog = true;
        result.push_back(m);
        known.push_back(key);
    }
    return result;
}

size_t scan_model_dir(const std::string & dir, std::vector<model_entry> & models) {
    if (dir.empty()) {
        return 0;
    }
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return 0;
    }
    std::vector<std::string> names;
    while (struct dirent * de = readdir(d)) {
        const std::string name  = de->d_name;
        const std::string lower = to_lower_ascii(name);
        // skip hidden files (including macOS "._foo.gguf" resource forks), projectors and extra shards
        if (name.empty() || name[0] == '.' || !ends_with(lower, ".gguf") || lower.size() <= 5 ||
            lower.rfind("mmproj", 0) == 0 || is_extra_split_shard(lower)) {
            continue;
        }
        names.push_back(name);
    }
    closedir(d);
    std::sort(names.begin(), names.end(), [](const std::string & a, const std::string & b) {
        return to_lower_ascii(a) < to_lower_ascii(b);
    });

    std::vector<std::string> known = normalized_paths(models);
    const std::string prefix = dir.back() == '/' ? dir : dir + "/";
    size_t added = 0;
    for (const auto & name : names) {
        const std::string path = prefix + name;
        const std::string key  = normalize_model_path(path);
        if (path_in_list(known, key)) {
            continue;
        }
        uint64_t size  = 0;
        int64_t  mtime = 0;
        gguf_file_info info;
        if (!stat_regular_file(path, size, mtime) || !read_gguf_info_cached(path, size, mtime, info)) {
            continue;
        }
        if ((!info.type.empty() && info.type != "model") || info.arch == "clip") {
            continue;
        }

        model_entry m;
        m.name = display_name_for(info, path);
        m.path = path;
        m.default_ctx = clamp_ctx(info.n_ctx_train > 0 ? (int) std::min<uint32_t>(info.n_ctx_train, k_scan_ctx_default) : k_scan_ctx_default);
        m.file_size = size;
        m.on_sd     = true;
        m.arch      = info.arch;
        m.from_scan = true;
        models.push_back(m);
        known.push_back(key);
        ++added;
    }
    return added;
}

void refresh_model_files(std::vector<model_entry> & models) {
    for (auto & m : models) {
        uint64_t size  = 0;
        int64_t  mtime = 0;
        if (!stat_regular_file(m.path, size, mtime)) {
            m.on_sd     = false;
            m.file_size = 0;
            continue; // keep a catalog-provided arch, if any
        }
        m.on_sd     = true;
        m.file_size = size;
        gguf_file_info info;
        if (read_gguf_info_cached(m.path, size, mtime, info)) {
            m.arch = info.arch;
        }
    }
}

void sort_model_list(std::vector<model_entry> & models) {
    std::stable_partition(models.begin(), models.end(), [](const model_entry & m) { return m.on_sd; });
}

bool load_model_list(const std::string & path, std::vector<model_entry> & out, std::string & err) {
    return load_model_list(path, k_models_dir, out, err);
}

bool load_model_list(const std::string & path, const std::string & models_dir, std::vector<model_entry> & out, std::string & err) {
    std::string warn;
    std::vector<model_entry> list;

    // models.json is only (re)created when it does not exist or is empty; any other problem is an error
    struct stat st;
    const bool have_file = stat(path.c_str(), &st) == 0;
    std::string text;
    if (have_file) {
        std::ifstream in(path, std::ios::binary);
        if (!S_ISREG(st.st_mode) || !in) {
            err = "failed to open " + path;
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf(); // an empty file only sets failbit on ss
        if (in.bad()) {
            err = "failed to read " + path;
            return false;
        }
        text = ss.str();
    }

    bool use_catalog = true;
    if (!have_file || trim(text).empty()) {
        // no list yet (or an empty file): create it from the built-in catalog
        list = default_models();
        std::string werr;
        if (!write_model_list(path, list, werr)) {
            append_warning(warn, werr + " (using the built-in list)");
        }
    } else {
        try {
            const json root = json::parse(text, nullptr, true, true);
            const json * models = nullptr;
            if (root.is_array()) {
                models = &root;
            } else if (root.is_object() && root.contains("models") && root["models"].is_array()) {
                models = &root["models"];
            }
            if (!models) {
                err = "invalid models.json: missing models array";
                return false;
            }
            if (root.is_object() && root.contains("catalog") && root["catalog"].is_boolean()) {
                use_catalog = root["catalog"].get<bool>();
            }
            parse_model_entries(*models, list, warn);
            if (list.empty() && !models->empty()) {
                append_warning(warn, "no usable entry in models.json");
            }
        } catch (const std::exception & ex) {
            err = std::string("failed to parse models.json: ") + ex.what();
            return false;
        }
    }

    if (use_catalog) {
        const std::vector<model_entry> missing = missing_catalog_models(list);
        list.insert(list.end(), missing.begin(), missing.end());
    }

    refresh_model_files(list);
    scan_model_dir(models_dir, list);
    sort_model_list(list);

    if (list.empty()) {
        err = warn.empty() ? "model list is empty" : warn;
        return false;
    }

    out = std::move(list);
    err = warn;
    return true;
}

std::vector<model_entry> fallback_model_list(const std::string & models_dir) {
    std::vector<model_entry> list = default_models();
    refresh_model_files(list);
    scan_model_dir(models_dir, list);
    sort_model_list(list);
    return list;
}

bool reload_model_list(app_state & state, std::string & err) {
    std::string selected_path;
    if (!state.models.empty()) {
        selected_path = state.models[std::min(state.selected_model, state.models.size() - 1)].path;
    }

    std::vector<model_entry> list;
    if (!load_model_list(k_model_list_path, list, err)) {
        return false;
    }

    state.models = std::move(list);
    state.selected_model = 0;
    const std::string key = normalize_model_path(selected_path);
    for (size_t i = 0; i < state.models.size(); ++i) {
        if (!key.empty() && normalize_model_path(state.models[i].path) == key) {
            state.selected_model = i;
            break;
        }
    }
    apply_selected_model_defaults(state);
    return true;
}

void apply_selected_model_defaults(app_state & state) {
    if (state.models.empty()) {
        return;
    }
    if (state.selected_model >= state.models.size()) {
        state.selected_model = state.models.size() - 1;
    }
    const auto & m = state.models[state.selected_model];
    state.n_ctx = clamp_ctx(m.default_ctx);
    state.temperature = clamp_temp(m.default_temp);
    state.top_p = clamp_top_p(m.default_top_p);
    state.top_k = clamp_top_k(m.default_top_k);
}
