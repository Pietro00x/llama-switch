#pragma once

// Small drawing toolkit for the libnx text console (80x45 characters with the default font).
// Only 7-bit ASCII is drawn reliably: pass model output through ui_ascii()/ui_wrap() first.

#include <switch.h>

#include <string>
#include <vector>

// ANSI SGR sequences understood by the libnx console
#define UI_RESET    "\x1b[0m"
#define UI_BOLD     "\x1b[1m"
#define UI_FAINT    "\x1b[2m"
#define UI_REVERSE  "\x1b[7m"

#define UI_FG_BLACK   "\x1b[30m"
#define UI_FG_RED     "\x1b[31;1m"
#define UI_FG_GREEN   "\x1b[32;1m"
#define UI_FG_YELLOW  "\x1b[33;1m"
#define UI_FG_BLUE    "\x1b[34;1m"
#define UI_FG_MAGENTA "\x1b[35;1m"
#define UI_FG_CYAN    "\x1b[36;1m"
#define UI_FG_WHITE   "\x1b[37;1m"
#define UI_FG_GRAY    "\x1b[37m"

#define UI_BG_BLACK   "\x1b[40m"
#define UI_BG_RED     "\x1b[41m"
#define UI_BG_GREEN   "\x1b[42m"
#define UI_BG_YELLOW  "\x1b[43m"
#define UI_BG_BLUE    "\x1b[44m"
#define UI_BG_MAGENTA "\x1b[45m"
#define UI_BG_CYAN    "\x1b[46m"
#define UI_BG_WHITE   "\x1b[47m"

// console size in characters
int ui_cols();
int ui_rows();

// ui_frame_begin() clears the screen and homes the cursor; ui_frame_end() presents it (consoleUpdate).
// Draw a whole frame between the two and only redraw when something changed.
void ui_frame_begin();
void ui_frame_end();

// moves the cursor to a 0-based row/column
void ui_move(int row, int col);

// full-width reverse-video bars: title on row 0 (right text right-aligned), hints on the last row
void ui_title_bar(const std::string & title, const std::string & right);
void ui_footer(const std::string & hints);

// horizontal line across the screen at row
void ui_hline(int row, char ch = '-');

// ASCII box with an optional title in the top border
void ui_box(int row, int col, int height, int width, const std::string & title);

// replaces every non-ASCII UTF-8 sequence with '?' and control characters (except '\n') with ' '
std::string ui_ascii(const std::string & text);

// word-wraps text (after ui_ascii) to lines of at most width characters; honors '\n'
std::vector<std::string> ui_wrap(const std::string & text, int width);

// "[#####.....]" with width characters in total (including brackets); frac is clamped to [0, 1]
std::string ui_progress_bar(double frac, int width);

// pads or clips s to exactly width characters
std::string ui_fit(const std::string & s, int width);

// modal yes/no dialog drawn over the current screen: A = yes, B = no
bool ui_confirm(PadState & pad, const std::string & question);
