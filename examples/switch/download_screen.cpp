#include "download.h"
#include "screens.h"
#include "sysinfo.h"
#include "ui.h"

#include <cstdio>
#include <string>

static void render_download_screen(const app_state & state, const model_entry & m, size_t received, size_t total) {
    ui_frame_begin();
    ui_title_bar("Download model", "");

    ui_move(2, 1);
    std::printf("Model : %s\n", clip_line(m.name).c_str());
    std::printf(" URL   : %s\n", clip_line(normalize_hf_url(m.url), 70).c_str());
    std::printf(" Path  : %s\n", clip_line(m.path, 70).c_str());

    ui_move(7, 1);
    if (total > 0) {
        const double frac = (double) received / (double) total;
        std::printf("%s %5.1f%%\n", ui_progress_bar(frac, 60).c_str(), 100.0 * frac);
        std::printf(" %s / %s\n", fmt_mb(received).c_str(), fmt_mb(total).c_str());
    } else {
        std::printf("%s\n", fmt_mb(received).c_str());
    }

    ui_move(11, 1);
    std::printf("RAM: %s / %s\n", fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str());

    ui_footer("B cancel");
    ui_frame_end();
}

void run_download_screen(app_state & state, PadState & pad) {
    if (state.models.empty()) {
        state.status = "No model selected";
        return;
    }
    const model_entry & selected = state.models[state.selected_model];

    uint64_t last_draw = 0;
    auto progress_cb = [&](size_t received, size_t total) {
        padUpdate(&pad);
        if (padGetButtonsDown(&pad) & HidNpadButton_B) {
            return false;
        }
        const uint64_t t = now_us();
        if (t - last_draw < 150000) {
            return true;
        }
        last_draw = t;
        update_stats(state);
        render_download_screen(state, selected, received, total);
        return true;
    };

    render_download_screen(state, selected, 0, 0);

    std::string err;
    if (download_file(selected.url, selected.path, err, progress_cb)) {
        state.status = "Download complete: " + selected.path;
    } else {
        state.status = "Download failed: " + err;
    }
}
