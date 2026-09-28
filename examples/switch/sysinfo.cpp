#include "sysinfo.h"

#include <cinttypes>
#include <cstdio>

static bool g_clock_initialized = false;
static bool g_use_clkrst        = false;

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

static u32 read_clock_hz(PcvModule module, PcvModuleId module_id) {
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

    ClkrstSession session {};
    u32 hz = 0;
    if (R_SUCCEEDED(clkrstOpenSession(&session, module_id, 3))) {
        (void) clkrstGetClockRate(&session, &hz);
        clkrstCloseSession(&session);
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
    state.cpu_hz = read_clock_hz(PcvModule_CpuBus, PcvModuleId_CpuBus);
    state.gpu_hz = read_clock_hz(PcvModule_GPU, PcvModuleId_GPU);
    state.emc_hz = read_clock_hz(PcvModule_EMC, PcvModuleId_EMC);
}

void sysinfo_exit() {
    if (g_clock_initialized) {
        if (g_use_clkrst) {
            clkrstExit();
        } else {
            pcvExit();
        }
        g_clock_initialized = false;
    }
}
