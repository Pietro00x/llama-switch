#pragma once

// CPU and memory (EMC) clocks of the Tegra X1 while a model is loaded. Token generation is bound by the CPU
// and the memory bandwidth, so both are raised.
//
// Modes (clock_mode in app.h, chosen in the menu, persisted in settings.json):
//   STOCK      the system configuration is left alone (CPU 1020 MHz, EMC 1331.2 / 1600 MHz handheld / docked).
//   BOOST      CPU 1785 MHz and EMC 1600 MHz, the highest rates the system itself uses (docked boost), also in
//              handheld. appletSetCpuBoostMode(FastLoad) gives CPU 1785 MHz and drops the unused GPU to its
//              minimum; clkrst (pcv before 8.0.0) then sets the exact CPU and EMC rates. If clkrst fails,
//              FastLoad alone remains (EMC 1331.2 MHz in handheld).
//   OVERCLOCK  the highest CPU and EMC rates the system reports above those (GetPossibleClockRates). A patched
//              pcv (sys-clk-OC / OC suite) adds them; when none are reported OVERCLOCK behaves like BOOST.
//
// While the clocks are raised they are re-applied when the system resets them: applet hooks for focus regained,
// resume from sleep, dock/undock and performance mode changes, plus a check every 2 s that gives up on a clock
// after 3 re-applies in a row (skipped when sys-clk is running: it applies its own profiles and the two would
// fight). Restoring sets the rates from before the first raise (the normal ones if the dock state changed) and
// ends the boost mode, which makes the system apply its own configuration again.
//
// The rate selection (clocks.cpp) is pure and host-testable. The runtime is in clocks_switch.cpp; on the host
// clocks.cpp provides versions that only update app_state.

#include "app.h"

#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t k_clock_cpu_stock_hz    = 1020000000; // CPU rate of the system's normal configurations
constexpr uint32_t k_clock_cpu_official_hz = 1785000000; // CPU rate of the system's boost configurations
constexpr uint32_t k_clock_emc_official_hz = 1600000000; // EMC rate of the docked configurations
constexpr uint32_t k_clock_emc_handheld_hz = 1331200000; // EMC rate of the handheld configurations

// possible rates of one clock, as reported by GetPossibleClockRates
struct clock_rates {
    std::vector<uint32_t> hz;            // ascending, no zeros or duplicates (clock_rates_normalize); empty = unknown
    bool                  range = false; // any rate in [hz.front(), hz.back()] is allowed
};

// what to request for a mode
struct clock_plan {
    clock_mode mode            = CLOCK_MODE_STOCK; // effective mode (OVERCLOCK falls back to BOOST)
    uint32_t   cpu_hz          = 0;                // 0 = leave untouched
    uint32_t   emc_hz          = 0;
    uint32_t   fastload_emc_hz = 0;                // EMC that FastLoad alone gives (docked 1600, handheld 1331.2)
};

// sorts and removes zeros and duplicates
void clock_rates_normalize(clock_rates & r);

// rates strictly above min_hz, ascending (for a range: only its maximum)
std::vector<uint32_t> clock_rates_above(const clock_rates & r, uint32_t min_hz);

// the system reports a CPU rate above 1785 MHz or an EMC rate above 1600 MHz
bool clock_oc_available(const clock_rates & cpu, const clock_rates & emc);

// Rates for `mode`:
//   STOCK      nothing.
//   BOOST      the highest reported rates <= the official max (the official max itself if the list is unknown).
//   OVERCLOCK  the highest reported CPU rate <= oc_cpu_hz (0 = no cap; a cap below every rate above 1785 MHz
//              means the lowest of them) and the highest reported EMC rate, never below BOOST; plain BOOST when
//              clock_oc_available() is false.
// `docked` only changes fastload_emc_hz.
clock_plan clock_plan_make(clock_mode mode, uint32_t oc_cpu_hz, const clock_rates & cpu, const clock_rates & emc, bool docked);

// One Left (dir < 0) or Right (dir > 0) press on the "CPU clock" menu row. The values are ordered
//   Stock, Boost, Overclock at each rate of oc_cpu_steps (the CPU rates above 1785 MHz, ascending),
// except that Right on Boost goes straight to the highest step (oc_cpu_hz = 0): Overclock starts at the
// maximum and Left lowers the CPU step by step, then back to Boost.
struct clock_step {
    clock_mode mode        = CLOCK_MODE_BOOST;
    uint32_t   oc_cpu_hz   = 0;
    bool       confirm     = false; // Boost -> Overclock: ask the user before applying
    bool       unavailable = false; // Overclock requested but not available: nothing changed
};
clock_step clock_step_setting(clock_mode mode, uint32_t oc_cpu_hz, const std::vector<uint32_t> & oc_cpu_steps,
    bool oc_available, int dir);

// "Stock" / "Boost" / "Overclock"
const char * clock_mode_name(clock_mode mode);

// MHz without decimals when whole ("1785"), else with one ("1331.2")
std::string clock_fmt_mhz(uint32_t hz);

//
// runtime (Switch; the host versions only update app_state)
//

// Loads settings.json into state.clock_setting/clock_oc_cpu_hz, queries the possible rates, detects sys-clk and
// installs the applet hooks and the check thread. Call once after appletInitialize (i.e. in main) and before
// load_model. Returns a note for the status line, empty if there is nothing to report.
std::string clocks_init(app_state & state);

// Applies state.clock_setting (load_model calls it, and clocks_set_setting while a model is loaded). STOCK
// restores. Updates state.clock_applied/clock_cpu_target_hz/clock_emc_target_hz/cpu_boost.
void clocks_apply(app_state & state);

// Sets the clocks back to what they were before clocks_apply and ends the boost mode. Safe to call when
// nothing is raised.
void clocks_restore(app_state & state);

// Best-effort restore for the abort handler (any thread, never blocks).
void clocks_restore_emergency();

// clocks_restore, then removes the hooks, stops the check thread and closes the services.
void clocks_exit(app_state & state);

// CPU rates above 1785 MHz the system reports (ascending; the Overclock steps of the menu).
const std::vector<uint32_t> & clocks_oc_cpu_steps();

// clock_plan_make for state.clock_setting with the rates the system reports, e.g. to show what a mode means.
clock_plan clocks_plan_for(const app_state & state);

// Changes the mode, applies it right away when a model is loaded and saves settings.json. Returns false if
// settings.json could not be written (err is set; the new mode is still used).
bool clocks_set_setting(app_state & state, clock_mode mode, uint32_t oc_cpu_hz, std::string & err);
