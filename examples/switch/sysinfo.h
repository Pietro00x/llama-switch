#pragma once

#include "app.h"

#include <switch.h>

#include <cstdint>
#include <string>

// Monotonic time in microseconds.
uint64_t now_us();

// Formats a libnx Result as 0x%08x.
std::string rc_to_hex(Result rc);

// Refreshes process RAM usage and CPU/GPU/EMC clocks in state (at most every 250 ms).
void update_stats(app_state & state);

// Closes the clock services opened by update_stats.
void sysinfo_exit();
