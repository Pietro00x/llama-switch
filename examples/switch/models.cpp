#include "models.h"

#include "catalog.h"

#include <nlohmann/json.hpp>

#include <fstream>

using json = nlohmann::json;

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
    return true;
}

bool load_model_list(const std::string & path, std::vector<model_entry> & out, std::string & err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out = default_models();
        if (!write_model_list(path, out, err)) {
            return false;
        }
        return true;
    }

    try {
        json root = json::parse(in);
        if (!root.contains("models") || !root["models"].is_array()) {
            err = "invalid models.json: missing models array";
            return false;
        }

        std::vector<model_entry> parsed;
        for (const auto & item : root["models"]) {
            if (!item.is_object()) {
                continue;
            }
            if (!item.contains("name") || !item.contains("url") || !item.contains("path")) {
                continue;
            }

            model_entry m;
            m.name = item["name"].get<std::string>();
            m.url  = item["url"].get<std::string>();
            m.path = item["path"].get<std::string>();

            if (item.contains("context")) {
                m.default_ctx = clamp_ctx(item["context"].get<int>());
            }
            if (item.contains("temperature")) {
                m.default_temp = clamp_temp(item["temperature"].get<float>());
            }
            if (item.contains("top_p")) {
                m.default_top_p = clamp_top_p(item["top_p"].get<float>());
            }
            if (item.contains("top_k")) {
                m.default_top_k = clamp_top_k(item["top_k"].get<int>());
            }
            if (item.contains("system_prompt")) {
                m.system_prompt = item["system_prompt"].get<std::string>();
            }
            if (item.contains("chat_template")) {
                m.chat_template = item["chat_template"].get<std::string>();
            }

            parsed.push_back(std::move(m));
        }

        if (parsed.empty()) {
            err = "model list is empty";
            return false;
        }

        out = std::move(parsed);
        return true;
    } catch (const std::exception & ex) {
        err = std::string("failed to parse models.json: ") + ex.what();
        return false;
    }
}

void apply_selected_model_defaults(app_state & state) {
    if (state.models.empty()) {
        return;
    }
    const auto & m = state.models[state.selected_model];
    state.n_ctx = clamp_ctx(m.default_ctx);
    state.temperature = clamp_temp(m.default_temp);
    state.top_p = clamp_top_p(m.default_top_p);
    state.top_k = clamp_top_k(m.default_top_k);
}
