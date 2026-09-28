#pragma once

#include "app.h"

#include <switch.h>

// Each screen runs its own input/render loop and returns to the caller when closed.

// Main menu; returns when the user quits (menu.cpp).
void run_main_menu(app_state & state, PadState & pad);

// Chat with the loaded model (chat_screen.cpp).
void run_chat_screen(app_state & state, PadState & pad);

// Downloads the selected model with a progress view; B cancels (download_screen.cpp).
void run_download_screen(app_state & state, PadState & pad);
