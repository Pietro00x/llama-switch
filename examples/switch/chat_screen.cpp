#include "generate.h"
#include "screens.h"
#include "sysinfo.h"
#include "ui.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

//
// Screen layout (80x45 console, rows are 0-based):
//
//   row 0            title bar: model file name, context size
//   rows 1 .. R-7    transcript (R-7 rows), scrollbar in column C-2
//   row  R-6         separator with the scroll position
//   rows R-5, R-4    prompt draft
//   row  R-3         status (live generation speed / last result)
//   row  R-2         RAM / clock / sampler stats
//   row  R-1         footer with button hints
//
// Transcript lines: a 5-column speaker label ("You" / "AI") starting at column 1, the text at
// column 6 and at most C-9 characters wide, so the text ends at column C-4 and column C-1 is
// never written (that would wrap the cursor).
//

static const int k_label_cols = 5;
static const int k_body_col   = 1 + k_label_cols;

// A token takes ~170 ms at 6 tok/s and the time spent drawing is time the model does not run (a full
// redraw took ~10 ms): while generating, the screen is redrawn at most every k_redraw_gen_us and when
// the phase changes (first token, end of a <think> block). The buttons are still read after every token.
static const uint64_t k_redraw_gen_us  = 400000;  // redraw while generating (~2.5 fps)
static const uint64_t k_redraw_idle_us = 1000000; // stats refresh while idle
static const uint64_t k_repeat_delay_us  = 350000;
static const uint64_t k_repeat_line_us   = 50000;
static const uint64_t k_repeat_page_us   = 200000;

enum chat_line_kind {
    CHAT_LINE_KIND_BLANK,
    CHAT_LINE_KIND_USER,
    CHAT_LINE_KIND_AI,
    CHAT_LINE_KIND_THINK,
    CHAT_LINE_KIND_HEADING,
    CHAT_LINE_KIND_CODE,
    CHAT_LINE_KIND_NOTE,
    CHAT_LINE_KIND_STOPPED,
};

enum chat_label {
    CHAT_LABEL_NONE,
    CHAT_LABEL_USER,
    CHAT_LABEL_AI,
};

struct chat_line {
    chat_line_kind kind  = CHAT_LINE_KIND_BLANK;
    chat_label     label = CHAT_LABEL_NONE; // set on the first line of each message
    std::string    text;
};

// ---------------------------------------------------------------------------------------------
// transcript layout (display-only transforms; the stored chat text is never modified)
// ---------------------------------------------------------------------------------------------

// maps common typographic UTF-8 sequences to ASCII look-alikes; ui_ascii() turns the rest into '?'
static std::string chat_fold_utf8(const std::string & s) {
    struct fold {
        const char * from;
        const char * to;
    };
    static const fold folds[] = {
        { "\xe2\x80\x98", "'"   }, { "\xe2\x80\x99", "'"   },
        { "\xe2\x80\x9c", "\""  }, { "\xe2\x80\x9d", "\""  },
        { "\xe2\x80\x93", "-"   }, { "\xe2\x80\x94", "-"   },
        { "\xe2\x80\xa6", "..." }, { "\xe2\x80\xa2", "*"   },
        { "\xe2\x86\x92", "->"  }, { "\xc2\xa0",     " "   },
    };

    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        bool matched = false;
        if ((unsigned char) s[i] >= 0x80) {
            for (const fold & f : folds) {
                const size_t n = std::strlen(f.from);
                if (s.compare(i, n, f.from) == 0) {
                    out += f.to;
                    i += n;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            out += s[i];
            ++i;
        }
    }
    return out;
}

static std::string chat_strip_trailing_space(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && (s[e - 1] == ' ' || s[e - 1] == '\n' || s[e - 1] == '\r' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(0, e);
}

static std::string chat_strip_leading_newlines(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == '\n' || s[b] == '\r')) {
        ++b;
    }
    return s.substr(b);
}

// appends text word-wrapped to width; the first emitted line takes the pending speaker label
static void chat_push_wrapped(
    std::vector<chat_line> & out,
    chat_label & label,
    chat_line_kind kind,
    const std::string & text,
    int width) {
    std::vector<std::string> wrapped = ui_wrap(text, width);
    if (wrapped.empty()) {
        wrapped.push_back(std::string());
    }
    for (size_t i = 0; i < wrapped.size(); ++i) {
        chat_line line;
        line.kind  = kind;
        line.label = label;
        line.text  = wrapped[i];
        out.push_back(line);
        label = CHAT_LABEL_NONE;
    }
}

static std::string chat_remove_all(std::string s, const char * what) {
    const size_t n = std::strlen(what);
    size_t pos = 0;
    while ((pos = s.find(what, pos)) != std::string::npos) {
        s.erase(pos, n);
    }
    return s;
}

// light markdown rendering of one reply body: headings, code fences, bold markers, bullets
static void chat_push_markdown(std::vector<chat_line> & out, chat_label & label, const std::string & text, int width) {
    bool in_code = false;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) {
            nl = text.size();
        }
        std::string para = text.substr(pos, nl - pos);
        if (!para.empty() && para.back() == '\r') {
            para.pop_back();
        }

        size_t indent = 0;
        while (indent < para.size() && para[indent] == ' ') {
            ++indent;
        }

        if (para.compare(indent, 3, "```") == 0) {
            in_code = !in_code;
            chat_push_wrapped(out, label, CHAT_LINE_KIND_CODE, para, width);
        } else if (in_code) {
            chat_push_wrapped(out, label, CHAT_LINE_KIND_CODE, para, width);
        } else {
            size_t hashes = 0;
            while (hashes < para.size() && para[hashes] == '#') {
                ++hashes;
            }
            if (hashes > 0 && hashes <= 6 && hashes < para.size() && para[hashes] == ' ') {
                chat_push_wrapped(out, label, CHAT_LINE_KIND_HEADING, chat_remove_all(para.substr(hashes + 1), "**"), width);
            } else {
                para = chat_remove_all(para, "**");
                if (para.compare(indent, 2, "* ") == 0) {
                    para[indent] = '-';
                }
                chat_push_wrapped(out, label, CHAT_LINE_KIND_AI, para, width);
            }
        }

        if (nl == text.size()) {
            break;
        }
        pos = nl + 1;
    }
}

// lays out one exchange; reply may be partial (generating) or cancelled (stopped)
// the caller adds the blank separator line between turns
static void chat_layout_turn(
    std::vector<chat_line> & out,
    const std::string & user,
    const std::string & reply,
    bool stopped,
    bool generating,
    int width) {
    chat_label label = CHAT_LABEL_USER;
    chat_push_wrapped(out, label, CHAT_LINE_KIND_USER, chat_strip_trailing_space(chat_fold_utf8(user)), width);

    label = CHAT_LABEL_AI;
    std::string text = chat_fold_utf8(reply);

    // a leading <think>...</think> block (reasoning models) is drawn faint
    size_t lead = 0;
    while (lead < text.size() && (text[lead] == ' ' || text[lead] == '\n' || text[lead] == '\r')) {
        ++lead;
    }
    if (text.compare(lead, 7, "<think>") == 0) {
        const size_t end = text.find("</think>", lead + 7);
        std::string think;
        if (end == std::string::npos) {
            think = text.substr(lead + 7);
            text.clear();
        } else {
            think = text.substr(lead + 7, end - lead - 7);
            text  = text.substr(end + 8);
        }
        think = chat_strip_trailing_space(chat_strip_leading_newlines(think));
        if (!think.empty()) {
            chat_push_wrapped(out, label, CHAT_LINE_KIND_THINK, think, width);
        } else if (end == std::string::npos) {
            chat_push_wrapped(out, label, CHAT_LINE_KIND_THINK, "(thinking)", width);
        }
    }

    text = chat_strip_trailing_space(chat_strip_leading_newlines(text));
    if (!text.empty()) {
        chat_push_markdown(out, label, text, width);
    }

    if (stopped) {
        chat_push_wrapped(out, label, CHAT_LINE_KIND_STOPPED, "[stopped]", width);
    } else if (label != CHAT_LABEL_NONE) {
        // nothing was drawn for the reply yet
        chat_push_wrapped(out, label, CHAT_LINE_KIND_NOTE, generating ? "..." : "(empty reply)", width);
    }
}

static std::vector<chat_line> chat_layout_history(const std::vector<chat_turn> & turns, int width) {
    std::vector<chat_line> out;
    for (const chat_turn & turn : turns) {
        if (!out.empty()) {
            out.push_back(chat_line()); // blank line between turns
        }
        chat_layout_turn(out, turn.user, turn.assistant, turn.stopped, false, width);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// scrolling
// ---------------------------------------------------------------------------------------------

struct chat_scroll {
    int  top    = 0;    // first visible line when not following
    bool follow = true; // pinned to the bottom (auto-scroll)
};

static int chat_scroll_top(const chat_scroll & s, int total, int view) {
    const int max_top = std::max(0, total - view);
    if (s.follow) {
        return max_top;
    }
    return std::max(0, std::min(s.top, max_top));
}

static void chat_scroll_by(chat_scroll & s, int delta, int total, int view) {
    const int max_top = std::max(0, total - view);
    const int top     = std::max(0, std::min(chat_scroll_top(s, total, view) + delta, max_top));
    s.top    = top;
    s.follow = top >= max_top;
}

struct chat_repeat {
    u64      held     = 0;
    uint64_t since_us = 0;
    uint64_t last_us  = 0;
};

// D-pad / right stick scroll one line, ZL/ZR one page; held buttons repeat
static int chat_scroll_input(u64 k_down, u64 k_held, int page, uint64_t t_us, chat_repeat & rep) {
    const u64 k_up   = HidNpadButton_Up   | HidNpadButton_StickRUp;
    const u64 k_dn   = HidNpadButton_Down | HidNpadButton_StickRDown;
    const u64 k_mask = k_up | k_dn | HidNpadButton_ZL | HidNpadButton_ZR;

    u64 bits = 0;
    if (k_down & k_mask) {
        bits = k_down & k_mask;
        rep.held     = k_held & k_mask;
        rep.since_us = t_us;
        rep.last_us  = t_us;
    } else if (k_held & rep.held) {
        rep.held &= k_held;
        const bool     paging   = (rep.held & (HidNpadButton_ZL | HidNpadButton_ZR)) != 0;
        const uint64_t interval = paging ? k_repeat_page_us : k_repeat_line_us;
        if (t_us - rep.since_us >= k_repeat_delay_us && t_us - rep.last_us >= interval) {
            bits = rep.held;
            rep.last_us = t_us;
        }
    } else {
        rep.held = 0;
    }

    int delta = 0;
    if (bits & k_up) {
        delta -= 1;
    }
    if (bits & k_dn) {
        delta += 1;
    }
    if (bits & HidNpadButton_ZL) {
        delta -= page;
    }
    if (bits & HidNpadButton_ZR) {
        delta += page;
    }
    return delta;
}

// ---------------------------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------------------------

static bool show_prompt_keyboard(
    const std::string & current,
    const char * ok_text,
    std::string & output,
    std::string & err) {
    SwkbdConfig kbd {};
    Result rc = swkbdCreate(&kbd, 0);
    if (R_FAILED(rc)) {
        err = "swkbdCreate failed: " + rc_to_hex(rc);
        return false;
    }

    swkbdConfigMakePresetDefault(&kbd);
    swkbdConfigSetHeaderText(&kbd, "Chat Prompt");
    swkbdConfigSetGuideText(&kbd, "Enter prompt for the model");
    swkbdConfigSetSubText(&kbd, "Press + to submit");
    swkbdConfigSetOkButtonText(&kbd, ok_text);
    swkbdConfigSetStringLenMax(&kbd, 512);
    swkbdConfigSetInitialText(&kbd, current.c_str());

    char buffer[512 * 3 + 1] = {}; // 512 UTF-16 units are up to 3 UTF-8 bytes each
    rc = swkbdShow(&kbd, buffer, sizeof(buffer));
    swkbdClose(&kbd);

    if (R_FAILED(rc)) {
        if (rc == MAKERESULT(Module_Libnx, LibnxError_LibAppletBadExit)) {
            err = "Keyboard closed, prompt unchanged";
        } else {
            err = "keyboard failed: " + rc_to_hex(rc);
        }
        return false;
    }

    output = buffer;
    return true;
}

// ---------------------------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------------------------

struct chat_view {
    std::vector<chat_line> history; // lines of state.chat_turns
    std::vector<chat_line> pending; // lines of the exchange being generated
    chat_scroll scroll;
    std::string draft;
    bool        generating = false;
    std::string live_status;        // shown instead of state.status while generating

    std::vector<std::string> drawn;           // rows on screen after the last render_chat (empty: unknown)
    uint64_t                 drawn_frame = 0; // ui_frame_count() after that frame
    uint64_t                 draw_us     = 0; // duration of the last redraw (shown in the stats row)
};

// the next render_chat clears the screen and draws every row (after something else drew over it)
static void chat_view_invalidate(chat_view & v) {
    v.drawn.clear();
}

static int chat_view_rows() {
    return std::max(1, ui_rows() - 7);
}

static int chat_body_width() {
    return std::max(10, ui_cols() - 9);
}

static int chat_view_total(const chat_view & v) {
    return (int) (v.history.size() + v.pending.size());
}

static const chat_line & chat_view_line(const chat_view & v, int i) {
    if (i < (int) v.history.size()) {
        return v.history[(size_t) i];
    }
    return v.pending[(size_t) i - v.history.size()];
}

static std::string chat_model_name(const app_state & state) {
    const std::string & p = state.loaded_model_path;
    const size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? p : p.substr(slash + 1);
}

static const char * chat_line_color(chat_line_kind kind) {
    switch (kind) {
        case CHAT_LINE_KIND_USER:    return UI_FG_GRAY;
        case CHAT_LINE_KIND_AI:      return UI_FG_WHITE;
        case CHAT_LINE_KIND_THINK:   return UI_FAINT UI_FG_GRAY;
        case CHAT_LINE_KIND_HEADING: return UI_FG_YELLOW;
        case CHAT_LINE_KIND_CODE:    return UI_FG_MAGENTA;
        case CHAT_LINE_KIND_NOTE:    return UI_FAINT UI_FG_GRAY;
        case CHAT_LINE_KIND_STOPPED: return UI_FG_RED;
        default:                     return UI_RESET;
    }
}

static void render_empty_help(const app_state & state, int view) {
    static const char * const help[] = {
        "No messages yet. Write a prompt to start chatting.",
        "",
        "X        write / edit the prompt (keyboard)",
        "A        send the prompt (opens the keyboard if empty)",
        "B        hold to stop a reply / back to the menu",
        "Y        clear the chat",
        "-        take back the last prompt to edit it",
        "Up/Down  scroll one line (also the right stick)",
        "ZL/ZR    scroll one page",
    };
    const int n_help = (int) (sizeof(help) / sizeof(help[0]));

    char info[96];
    std::snprintf(info, sizeof(info), "Temp %.2f  Top-p %.2f  Top-k %d",
        clamp_temp(state.temperature), clamp_top_p(state.top_p), clamp_top_k(state.top_k));

    const int cols  = ui_cols();
    const int w     = std::min(cols - 4, 62);
    const int h     = n_help + 6;
    const int row   = 1 + std::max(0, (view - h) / 3);
    const int col   = (cols - w) / 2;
    if (h > view) {
        return;
    }

    ui_box(row, col, h, w, "Chat");
    for (int i = 0; i < n_help; ++i) {
        ui_move(row + 2 + i, col + 3);
        std::printf("%s%s" UI_RESET, i == 0 ? UI_FG_WHITE : UI_FG_GRAY, ui_fit(help[i], w - 6).c_str());
    }
    ui_move(row + h - 3, col + 3);
    std::printf(UI_FAINT UI_FG_GRAY "%s" UI_RESET, ui_fit(info, w - 6).c_str());
}

// One screen row being built: SGR escapes plus the visible characters counted in `used`. Every colored
// part ends with UI_RESET, so each row starts and ends in the default colors and can be drawn on its own.
struct chat_row {
    std::string bytes;
    int         used = 0;
};

// appends ASCII text (no escapes) in color (nullptr: default colors), clipped at column end_col
static void chat_row_add(chat_row & r, int end_col, const char * color, const std::string & text) {
    const int room = end_col - r.used;
    if (room <= 0 || text.empty()) {
        return;
    }
    const size_t n = std::min(text.size(), (size_t) room);
    if (color) {
        r.bytes += color;
    }
    r.bytes.append(text, 0, n);
    if (color) {
        r.bytes += UI_RESET;
    }
    r.used += (int) n;
}

// ui_ascii() on one row: a '\n' would move the cursor to the next row
static std::string chat_one_line(const std::string & s) {
    std::string out = ui_ascii(s);
    std::replace(out.begin(), out.end(), '\n', ' ');
    return out;
}

// pads with spaces up to column col
static void chat_row_pad(chat_row & r, int col) {
    if (col > r.used) {
        r.bytes.append((size_t) (col - r.used), ' ');
        r.used = col;
    }
}

// Every row of the screen is built as one string of exactly C-1 characters (column C-1 is never
// written) and only the rows that differ from the previous frame are drawn, over the old ones and
// without clearing: while generating that is usually the last transcript line and the status. The
// screen is cleared and fully drawn only when v.drawn does not describe it: first frame, after a
// dialog or another screen drew (ui_frame_count changed), and for the empty-chat help box, which is
// drawn over the rows.
static void render_chat(const app_state & state, chat_view & v) {
    const int cols  = ui_cols();
    const int rows  = ui_rows();
    const int view  = chat_view_rows();
    const int width = chat_body_width();
    const int total = chat_view_total(v);
    const int top   = chat_scroll_top(v.scroll, total, view);
    const int row_w = cols - 1;

    const std::string blank((size_t) row_w, ' ');
    std::vector<std::string> lines((size_t) rows, blank);
    auto set_line = [&](int row, chat_row & r) {
        chat_row_pad(r, row_w);
        if (row >= 0 && row < rows) {
            lines[(size_t) row].swap(r.bytes);
        }
    };

    // title
    char right[48];
    std::snprintf(right, sizeof(right), "ctx %d", state.ctx ? (int) llama_n_ctx(state.ctx) : clamp_ctx(state.n_ctx));
    const int name_room = std::max(8, cols - 12 - (int) std::strlen(right) - 3);
    lines[0] = ui_title_bar_line("Chat | " + clip_line(ui_ascii(chat_model_name(state)), (size_t) name_room), right);

    // transcript, with the scrollbar in column C-2
    const bool bar     = total > view;
    const int  thumb   = bar ? std::max(1, view * view / total) : 0;
    const int  thumb_r = bar ? (view - thumb) * top / (total - view) : 0;
    for (int r = 0; r < view; ++r) {
        chat_row row;
        if (top + r < total) {
            const chat_line & line = chat_view_line(v, top + r);
            if (line.kind != CHAT_LINE_KIND_BLANK) {
                chat_row_pad(row, 1);
                if (line.label == CHAT_LABEL_USER) {
                    chat_row_add(row, k_body_col, UI_FG_CYAN, "You");
                } else if (line.label == CHAT_LABEL_AI) {
                    chat_row_add(row, k_body_col, UI_FG_GREEN, "AI");
                }
                chat_row_pad(row, k_body_col);
                chat_row_add(row, k_body_col + width, chat_line_color(line.kind), line.text);
            }
        }
        if (bar) {
            chat_row_pad(row, cols - 2);
            if (r >= thumb_r && r < thumb_r + thumb) {
                chat_row_add(row, row_w, UI_FG_WHITE, "#");
            } else {
                chat_row_add(row, row_w, UI_FAINT UI_FG_GRAY, "|");
            }
        }
        set_line(1 + r, row);
    }

    // separator with the scroll position
    {
        std::string pos;
        if (bar) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), " %d-%d/%d%s ", top + 1, std::min(total, top + view), total,
                v.scroll.follow ? "" : " (ZR: newer)");
            pos = buf;
        }
        const std::string dashes((size_t) row_w, '-');
        const int pos_col = pos.empty() ? row_w : std::max(0, cols - 3 - (int) pos.size());
        chat_row row;
        chat_row_add(row, pos_col, UI_FAINT UI_FG_GRAY, dashes);
        chat_row_add(row, row_w, UI_FG_GRAY, pos);
        chat_row_add(row, row_w, UI_FAINT UI_FG_GRAY, dashes);
        set_line(rows - 6, row);
    }

    // prompt draft (two rows)
    {
        const int draft_col = 9;
        const int draft_end = cols - 2;
        const int draft_width = draft_end - draft_col;
        chat_row row0;
        chat_row row1;
        chat_row_pad(row0, 1);
        chat_row_add(row0, draft_col, UI_FG_CYAN, "Prompt");
        chat_row_pad(row0, draft_col);
        if (v.generating) {
            chat_row_add(row0, draft_end, UI_FAINT UI_FG_GRAY, "(generating - hold B to stop)");
        } else if (v.draft.empty()) {
            chat_row_add(row0, draft_end, UI_FAINT UI_FG_GRAY, "(empty - press X or A to write a prompt)");
        } else {
            std::string flat = chat_fold_utf8(v.draft);
            std::replace(flat.begin(), flat.end(), '\n', ' ');
            const std::vector<std::string> wrapped = ui_wrap(flat, draft_width);
            if (!wrapped.empty()) {
                chat_row_add(row0, draft_end, UI_FG_WHITE, wrapped[0]);
            }
            if (wrapped.size() > 1) {
                std::string text = wrapped[1];
                if (wrapped.size() > 2) {
                    text = text.substr(0, (size_t) std::max(0, draft_width - 3)) + "...";
                }
                chat_row_pad(row1, draft_col);
                chat_row_add(row1, draft_end, UI_FG_WHITE, text);
            }
        }
        set_line(rows - 5, row0);
        set_line(rows - 4, row1);
    }

    // status
    {
        chat_row row;
        chat_row_pad(row, 1);
        if (v.generating) {
            chat_row_add(row, cols - 2, UI_FG_YELLOW, chat_one_line(v.live_status));
        } else {
            chat_row_add(row, cols - 2, UI_FG_WHITE, chat_one_line(state.status));
        }
        set_line(rows - 3, row);
    }

    // stats; "draw" is how long the previous render_chat took (building the rows, drawing, presenting)
    {
        char stats[192];
        std::snprintf(stats, sizeof(stats), "RAM %s / %s  CPU %s  T %.2f P %.2f K %d  draw %.1f ms",
            fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str(), fmt_hz(state.cpu_hz).c_str(),
            clamp_temp(state.temperature), clamp_top_p(state.top_p), clamp_top_k(state.top_k), v.draw_us / 1000.0);
        chat_row row;
        chat_row_pad(row, 1);
        chat_row_add(row, cols - 2, UI_FAINT UI_FG_GRAY, stats);
        set_line(rows - 2, row);
    }

    // footer
    if (v.generating) {
        lines[(size_t) rows - 1] = ui_footer_line("Hold B Stop   Up/Down ZL/ZR Scroll (auto-scrolls at the bottom)");
    } else {
        lines[(size_t) rows - 1] = ui_footer_line("A Send  X Edit  Y Clear  - Undo last  Up/Dn ZL/ZR Scroll  B Back");
    }

    const bool empty = total == 0;
    const bool full  = empty || v.drawn.size() != lines.size() || v.drawn_frame != ui_frame_count();
    if (full) {
        ui_frame_begin();
    } else {
        ui_frame_begin_overwrite();
    }
    for (size_t i = 0; i < lines.size(); ++i) {
        if (full ? lines[i] == blank : lines[i] == v.drawn[i]) {
            continue; // already on screen: cleared, or unchanged since the last frame
        }
        ui_move((int) i, 0);
        std::fputs(lines[i].c_str(), stdout);
    }
    if (empty) {
        render_empty_help(state, view);
    }
    ui_frame_end();

    v.drawn_frame = ui_frame_count();
    if (empty) {
        chat_view_invalidate(v); // the help box is not part of `lines`
    } else {
        v.drawn.swap(lines);
    }
}

// render_chat, timed for the "draw" figure in the stats row
static void chat_draw(const app_state & state, chat_view & v) {
    const uint64_t t0 = now_us();
    render_chat(state, v);
    v.draw_us = now_us() - t0;
}

// ---------------------------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------------------------

// tok/s is the model speed from gen_stats, which leaves out the time spent in the token callback (UI);
// wall_tps (0: unknown) is the real rate including it
static std::string chat_fmt_live(const gen_stats & st, double wall_tps) {
    char buf[128];
    if (st.n_gen <= 0) {
        const int n_todo = st.n_prompt_total - st.n_cached;
        if (n_todo > 0) {
            std::snprintf(buf, sizeof(buf), "Processing prompt... %d/%d tok (%d reused)",
                st.n_prompt, n_todo, st.n_cached);
        } else {
            std::snprintf(buf, sizeof(buf), "Processing prompt...");
        }
    } else {
        const double tps = st.gen_ms > 0.0 ? 1000.0 * st.n_gen / st.gen_ms : 0.0;
        const int n = std::snprintf(buf, sizeof(buf), "%s %d tok  %.1f tok/s",
            st.thinking ? "Thinking..." : "Generating...", st.n_gen, tps);
        if (wall_tps > 0.0 && n > 0 && (size_t) n < sizeof(buf)) {
            std::snprintf(buf + n, sizeof(buf) - (size_t) n, " (wall %.1f)", wall_tps);
        }
    }
    return buf;
}

static std::string chat_fmt_done(const gen_stats & st, double wall_tps) {
    char wall[32] = "";
    if (wall_tps > 0.0) {
        std::snprintf(wall, sizeof(wall), " (wall %.1f)", wall_tps);
    }
    char buf[160];
    const double tps = st.gen_ms > 0.0 ? 1000.0 * st.n_gen / st.gen_ms : 0.0;
    std::snprintf(buf, sizeof(buf), "%s%d tok, %.1f tok/s%s, prefill %.0f ms (%d tok)",
        st.cancelled ? "Stopped: " : "Done: ", st.n_gen, tps, wall, st.prefill_ms, st.n_prompt);
    return buf;
}

// what the status row shows: 0 prompt processing, 1 thinking, 2 answering
static int chat_gen_phase(const gen_stats & st) {
    if (st.n_gen <= 0) {
        return 0;
    }
    return st.thinking ? 1 : 2;
}

// lays out the exchange being generated after the history lines
static void chat_layout_pending(chat_view & v, const std::string & user, const std::string & partial, int width) {
    v.pending.clear();
    if (!v.history.empty()) {
        v.pending.push_back(chat_line());
    }
    chat_layout_turn(v.pending, user, partial, false, true, width);
}

// sends v.draft; on success the exchange is appended to state.chat_turns and the draft cleared.
// B is polled once per generated token, so it has to be held until the next token arrives (during
// prompt processing that is the first token). An exit request from the system sets state.quit.
static void chat_send(app_state & state, PadState & pad, chat_view & v) {
    const int width = chat_body_width();
    const int view  = chat_view_rows();

    const std::string user = v.draft;

    v.generating  = true;
    v.scroll      = chat_scroll();
    v.live_status = chat_fmt_live(gen_stats(), 0.0);
    chat_layout_pending(v, user, std::string(), width);
    chat_draw(state, v);

    chat_repeat rep;
    uint64_t    last_draw   = now_us();
    int         drawn_phase = 0;
    int         n_callback  = 0;
    bool        stop_early  = false; // B was already held when the first token arrived

    // Wall clock of the generation phase, UI included. gen_stats.gen_ms leaves out the time spent in
    // this callback; it has not counted any of it yet at the first token, which dates the start.
    uint64_t t_gen_start = 0; // 0 until the first token
    uint64_t t_gen_last  = 0; // end of the callback of the latest token
    int      n_gen_last  = 0;
    auto wall_tps = [&](int n_gen, uint64_t t) {
        return n_gen > 0 && t_gen_start != 0 && t > t_gen_start ? 1e6 * n_gen / (double) (t - t_gen_start) : 0.0;
    };

    auto on_token = [&](const std::string & text, const gen_stats & st, uint64_t t) {
        ++n_callback;
        if (!ui_app_running()) {
            state.quit = true;
            return false;
        }

        // the buttons are read after every token; only the drawing below is throttled
        padUpdate(&pad);
        const u64 k_down = padGetButtonsDown(&pad);
        const u64 k_held = padGetButtons(&pad);
        if (k_held & HidNpadButton_B) {
            stop_early = n_callback == 1;
            return false;
        }

        const int phase = chat_gen_phase(st);
        const int delta = chat_scroll_input(k_down, k_held, view - 1, t, rep);
        if (delta == 0 && phase == drawn_phase && t - last_draw < k_redraw_gen_us) {
            return true;
        }

        last_draw   = t;
        drawn_phase = phase;
        chat_layout_pending(v, user, text, width); // first: the scroll range depends on it
        if (delta != 0) {
            chat_scroll_by(v.scroll, delta, chat_view_total(v), view);
        }
        v.live_status = chat_fmt_live(st, wall_tps(st.n_gen, t));
        update_stats(state);
        chat_draw(state, v);
        return true;
    };

    std::string assistant;
    std::string gen_err;
    gen_stats   stats;
    const bool ok = generate_assistant_reply(
        state,
        user,
        assistant,
        [&](const std::string & text, const gen_stats & st) {
            const uint64_t t = now_us();
            if (st.n_gen > 0 && t_gen_start == 0) {
                const uint64_t gen_us = st.gen_ms > 0.0 ? (uint64_t) (st.gen_ms * 1000.0) : 0;
                t_gen_start = t > gen_us ? t - gen_us : 1;
            }
            const bool keep_going = on_token(text, st, t);
            if (st.n_gen > 0) {
                t_gen_last = now_us();
                n_gen_last = st.n_gen;
            }
            return keep_going;
        },
        stats,
        gen_err);

    v.generating = false;
    v.pending.clear();

    // a B press that arrived after the last token must not also close the screen
    padUpdate(&pad);

    if (!ok) {
        state.status = "Generation failed: " + gen_err;
        return;
    }

    if (stats.cancelled && (stop_early || trim(assistant).empty())) {
        // stopped during prompt processing: keep the prompt in the draft instead of storing a stub turn
        state.status = "Stopped before the reply started; prompt kept in the draft";
        return;
    }

    chat_turn turn;
    turn.user      = user;
    turn.assistant = assistant;
    turn.stopped   = stats.cancelled;
    state.chat_turns.push_back(turn);
    if (!v.history.empty()) {
        v.history.push_back(chat_line());
    }
    chat_layout_turn(v.history, turn.user, turn.assistant, turn.stopped, false, width);

    v.draft.clear();
    state.status = chat_fmt_done(stats, wall_tps(n_gen_last, t_gen_last));
}

// ---------------------------------------------------------------------------------------------
// screen loop
// ---------------------------------------------------------------------------------------------

void run_chat_screen(app_state & state, PadState & pad) {
    if (!state.model || !state.ctx || !state.sampler) {
        state.status = "Load a model first";
        return;
    }

    chat_view v;
    v.history = chat_layout_history(state.chat_turns, chat_body_width());

    chat_repeat rep;
    bool        dirty     = true;
    uint64_t    last_draw = 0;

    while (true) {
        if (!ui_app_running()) {
            state.quit = true;
            break;
        }

        padUpdate(&pad);
        const u64      k_down = padGetButtonsDown(&pad);
        const u64      k_held = padGetButtons(&pad);
        const uint64_t t      = now_us();
        const int      view   = chat_view_rows();
        const int      width  = chat_body_width();

        if (k_down & HidNpadButton_B) {
            break;
        }

        const int delta = chat_scroll_input(k_down, k_held, view - 1, t, rep);
        if (delta != 0) {
            chat_scroll_by(v.scroll, delta, chat_view_total(v), view);
            dirty = true;
        }

        if (k_down & HidNpadButton_X) {
            std::string entered;
            std::string kbd_err;
            if (show_prompt_keyboard(v.draft, "OK", entered, kbd_err)) {
                v.draft = entered;
                state.status = trim(entered).empty() ? "Prompt is empty" : "Prompt ready - press A to send";
            } else {
                state.status = kbd_err;
            }
            padUpdate(&pad); // the keyboard consumed the buttons; do not act on them twice
            chat_view_invalidate(v); // redraw everything after the keyboard applet
            dirty = true;
        } else if (k_down & HidNpadButton_A) {
            bool send = !trim(v.draft).empty();
            if (!send) {
                std::string entered;
                std::string kbd_err;
                if (show_prompt_keyboard(v.draft, "Send", entered, kbd_err)) {
                    v.draft = entered;
                    send = !trim(entered).empty();
                    if (!send) {
                        state.status = "Prompt is empty";
                    }
                } else {
                    state.status = kbd_err;
                }
                padUpdate(&pad);
                chat_view_invalidate(v);
            }
            if (send) {
                chat_send(state, pad, v);
                if (state.quit) {
                    break;
                }
            }
            dirty = true;
        } else if (k_down & HidNpadButton_Y) {
            if (state.chat_turns.empty()) {
                state.status = "Chat is already empty";
            } else {
                char q[96];
                std::snprintf(q, sizeof(q), "Clear the whole chat (%d turns)?", (int) state.chat_turns.size());
                if (ui_confirm(pad, q)) {
                    state.chat_turns.clear();
                    v.history.clear();
                    v.scroll = chat_scroll();
                    state.status = "Chat cleared";
                }
            }
            dirty = true;
        } else if (k_down & HidNpadButton_Minus) {
            if (state.chat_turns.empty()) {
                state.status = "No prompt to take back";
            } else {
                const char * q = v.draft.empty()
                    ? "Take back the last prompt? Its reply is deleted."
                    : "Take back the last prompt? Its reply is deleted and the draft replaced.";
                if (ui_confirm(pad, q)) {
                    v.draft = state.chat_turns.back().user;
                    state.chat_turns.pop_back();
                    v.history = chat_layout_history(state.chat_turns, width);
                    v.scroll  = chat_scroll();
                    state.status = "Last prompt moved to the draft - X to edit, A to send";
                }
            }
            dirty = true;
        }

        update_stats(state);
        if (dirty || t - last_draw >= k_redraw_idle_us) {
            chat_draw(state, v);
            dirty     = false;
            last_draw = now_us();
        }

        svcSleepThread(16 * 1000 * 1000ULL);
    }
}
