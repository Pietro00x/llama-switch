#include "sysinfo.h"

#include <cinttypes>
#include <cstdio>

static bool g_clock_initialized = false;
static bool g_use_clkrst        = false;

// clkrst: one session per clock, opened on first use and kept until sysinfo_exit() (opening and closing
// a session around every read cost two extra IPCs per clock)
enum sysinfo_clock {
    SYSINFO_CLOCK_CPU,
    SYSINFO_CLOCK_GPU,
    SYSINFO_CLOCK_EMC,
    SYSINFO_CLOCK_COUNT,
};

static ClkrstSession g_clk_session[SYSINFO_CLOCK_COUNT];
static bool          g_clk_open[SYSINFO_CLOCK_COUNT] = {};

uint64_t now_us() {
    return armTicksToNs(armGetSystemTick()) / 1000ULL;
}

std::string rc_to_hex(Result rc) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08" PRIx32, rc);
    return std::string(buf);
}

static void init_clock_service() {
    if (g_clock_initialized) {
        return;
    }

    if (hosversionAtLeast(8, 0, 0)) {
        if (R_SUCCEEDED(clkrstInitialize())) {
            g_use_clkrst = true;
            g_clock_initialized = true;
        }
        return;
    }

    if (R_SUCCEEDED(pcvInitialize())) {
        g_use_clkrst = false;
        g_clock_initialized = true;
    }
}

static u32 read_clock_hz(sysinfo_clock clock, PcvModule module, PcvModuleId module_id) {
    if (!g_clock_initialized) {
        return 0;
    }
    if (!g_use_clkrst) {
        u32 hz = 0;
        if (R_SUCCEEDED(pcvGetClockRate(module, &hz))) {
            return hz;
        }
        return 0;
    }

    if (!g_clk_open[clock]) {
        // a failed open is retried on the next refresh
        if (R_FAILED(clkrstOpenSession(&g_clk_session[clock], module_id, 3))) {
            return 0;
        }
        g_clk_open[clock] = true;
    }
    u32 hz = 0;
    if (R_FAILED(clkrstGetClockRate(&g_clk_session[clock], &hz))) {
        // reopen next time in case the session went bad
        clkrstCloseSession(&g_clk_session[clock]);
        g_clk_open[clock] = false;
        return 0;
    }
    return hz;
}

void update_stats(app_state & state) {
    const u64 t = now_us();
    if (state.last_stats_us != 0 && (t - state.last_stats_us) < 250000) {
        return;
    }
    state.last_stats_us = t;

    u64 total = 0;
    u64 used = 0;
    if (R_SUCCEEDED(svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0))) {
        state.process_total = total;
    }
    if (R_SUCCEEDED(svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0))) {
        state.process_used = used;
    }

    init_clock_service();
    state.cpu_hz = read_clock_hz(SYSINFO_CLOCK_CPU, PcvModule_CpuBus, PcvModuleId_CpuBus);
    state.gpu_hz = read_clock_hz(SYSINFO_CLOCK_GPU, PcvModule_GPU,    PcvModuleId_GPU);
    state.emc_hz = read_clock_hz(SYSINFO_CLOCK_EMC, PcvModule_EMC,    PcvModuleId_EMC);
}

void sysinfo_exit() {
    for (int i = 0; i < SYSINFO_CLOCK_COUNT; ++i) {
        if (g_clk_open[i]) {
            clkrstCloseSession(&g_clk_session[i]);
            g_clk_open[i] = false;
        }
    }
    if (g_clock_initialized) {
        if (g_use_clkrst) {
            clkrstExit();
        } else {
            pcvExit();
        }
        g_clock_initialized = false;
    }
}
