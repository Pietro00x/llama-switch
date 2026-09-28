#include "ui.h"

#include <algorithm>
#include <cstdio>

int ui_cols() {
    PrintConsole * con = consoleGetDefault();
    return con && con->consoleWidth > 0 ? con->consoleWidth : 80;
}

int ui_rows() {
    PrintConsole * con = consoleGetDefault();
    return con && con->consoleHeight > 0 ? con->consoleHeight : 45;
}

void ui_frame_begin() {
    std::printf(UI_RESET);
    consoleClear();
}

void ui_frame_end() {
    std::printf(UI_RESET);
    std::fflush(stdout);
    consoleUpdate(nullptr);
}

void ui_move(int row, int col) {
    std::printf("\x1b[%d;%dH", row + 1, col + 1);
}

std::string ui_fit(const std::string & s, int width) {
    if (width <= 0) {
        return std::string();
    }
    if ((int) s.size() >= width) {
        return s.substr(0, (size_t) width);
    }
    return s + std::string((size_t) width - s.size(), ' ');
}

void ui_title_bar(const std::string & title, const std::string & right) {
    const int cols = ui_cols();
    std::string line = " " + title;
    const int room = cols - (int) line.size() - 1;
    if (!right.empty() && room > (int) right.size()) {
        line += std::string((size_t) (room - (int) right.size()), ' ') + right;
    }
    ui_move(0, 0);
    // the last column is left alone: writing it moves the cursor to the next line
    std::printf(UI_REVERSE "%s" UI_RESET, ui_fit(line, cols - 1).c_str());
}

void ui_footer(const std::string & hints) {
    const int cols = ui_cols();
    ui_move(ui_rows() - 1, 0);
    std::printf(UI_REVERSE "%s" UI_RESET, ui_fit(" " + hints, cols - 1).c_str());
}

void ui_hline(int row, char ch) {
    ui_move(row, 0);
    std::printf("%s", std::string((size_t) (ui_cols() - 1), ch).c_str());
}

void ui_box(int row, int col, int height, int width, const std::string & title) {
    if (height < 2 || width < 2) {
        return;
    }
    std::string top = "+" + std::string((size_t) (width - 2), '-') + "+";
    if (!title.empty() && (int) title.size() + 4 <= width) {
        top.replace(2, title.size() + 2, " " + title + " ");
    }
    const std::string mid = "|" + std::string((size_t) (width - 2), ' ') + "|";
    const std::string bot = "+" + std::string((size_t) (width - 2), '-') + "+";

    ui_move(row, col);
    std::printf("%s", top.c_str());
    for (int r = 1; r < height - 1; ++r) {
        ui_move(row + r, col);
        std::printf("%s", mid.c_str());
    }
    ui_move(row + height - 1, col);
    std::printf("%s", bot.c_str());
}

std::string ui_ascii(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ) {
        const unsigned char c = (unsigned char) text[i];
        if (c < 0x80) {
            if (c == '\n') {
                out += '\n';
            } else if (c == '\t') {
                out += ' ';
            } else if (c < 0x20 || c == 0x7f) {
                out += ' ';
            } else {
                out += (char) c;
            }
            ++i;
            continue;
        }
        // skip one UTF-8 sequence (lead byte + continuation bytes)
        ++i;
        while (i < text.size() && (((unsigned char) text[i]) & 0xC0) == 0x80) {
            ++i;
        }
        out += '?';
    }
    return out;
}

std::vector<std::string> ui_wrap(const std::string & text, int width) {
    std::vector<std::string> lines;
    if (width <= 0) {
        return lines;
    }

    const std::string clean = ui_ascii(text);
    size_t pos = 0;
    while (pos <= clean.size()) {
        size_t nl = clean.find('\n', pos);
        if (nl == std::string::npos) {
            nl = clean.size();
        }
        std::string para = clean.substr(pos, nl - pos);

        if (para.empty()) {
            lines.push_back(std::string());
        }
        while (!para.empty()) {
            if ((int) para.size() <= width) {
                lines.push_back(para);
                break;
            }
            size_t cut = para.rfind(' ', (size_t) width);
            if (cut == std::string::npos || cut == 0) {
                cut = (size_t) width; // no space: hard break
                lines.push_back(para.substr(0, cut));
                para.erase(0, cut);
            } else {
                lines.push_back(para.substr(0, cut));
                para.erase(0, cut + 1);
            }
        }

        if (nl == clean.size()) {
            break;
        }
        pos = nl + 1;
    }
    return lines;
}

std::string ui_progress_bar(double frac, int width) {
    if (width < 3) {
        return std::string();
    }
    frac = std::max(0.0, std::min(1.0, frac));
    const int inner  = width - 2;
    const int filled = (int) (frac * inner + 0.5);
    return "[" + std::string((size_t) filled, '#') + std::string((size_t) (inner - filled), '.') + "]";
}

static bool g_ui_exit_requested = false;

bool ui_app_running() {
    if (!g_ui_exit_requested && !appletMainLoop()) {
        g_ui_exit_requested = true;
    }
    return !g_ui_exit_requested;
}

bool ui_confirm(PadState & pad, const std::string & question) {
    const int cols  = ui_cols();
    const int w     = std::min(cols - 4, std::max(40, (int) question.size() + 6));
    const int h     = 7;
    const int row   = ui_rows() / 2 - h / 2;
    const int col   = (cols - w) / 2;

    const std::vector<std::string> text = ui_wrap(question, w - 4);

    ui_box(row, col, h, w, "Confirm");
    for (size_t i = 0; i < text.size() && i < 2; ++i) {
        ui_move(row + 2 + (int) i, col + 2);
        std::printf(UI_FG_WHITE "%s" UI_RESET, text[i].c_str());
    }
    ui_move(row + h - 2, col + 2);
    std::printf(UI_FG_GREEN "(A) Yes" UI_RESET "   " UI_FG_RED "(B) No" UI_RESET);
    ui_frame_end();

    while (ui_app_running()) {
        padUpdate(&pad);
        const u64 k_down = padGetButtonsDown(&pad);
        if (k_down & HidNpadButton_A) {
            return true;
        }
        if (k_down & HidNpadButton_B) {
            return false;
        }
        svcSleepThread(16 * 1000 * 1000ULL);
    }
    return false;
}
