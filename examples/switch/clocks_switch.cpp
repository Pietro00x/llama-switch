// Switch runtime of clocks.h: clkrst/pcv, the boost mode, applet hooks and the check thread.

#include "clocks.h"

#include "engine.h"
#include "settings.h"

#include <switch.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

enum clock_domain {
    CLOCK_DOMAIN_CPU = 0,
    CLOCK_DOMAIN_EMC,
    CLOCK_DOMAIN_COUNT,
};

static const char * const k_domain_names[CLOCK_DOMAIN_COUNT]   = { "CPU", "EMC" };
static const PcvModuleId  k_domain_ids[CLOCK_DOMAIN_COUNT]     = { PcvModuleId_CpuBus, PcvModuleId_EMC };
static const PcvModule    k_domain_modules[CLOCK_DOMAIN_COUNT] = { PcvModule_CpuBus, PcvModule_EMC };

static constexpr int      k_rates_max          = 64;
static constexpr int      k_check_interval_ms  = 2000;
static constexpr uint32_t k_check_tolerance_hz = 1000000;
static constexpr int      k_check_log_max      = 20; // the check thread logs its first re-applies only
static constexpr int      k_check_retry_max    = 3;  // checks in a row that found a clock changed: then give up on it

// sys-clk (and the sys-clk-OC fork) register this service; its boot2 flag is the fallback when sm cannot tell
static const char * const k_sysclk_service    = "sys:clk";
static const char * const k_sysclk_boot2_flag = "sdmc:/atmosphere/contents/00FF0000636C6BFF/flags/boot2.flag";

struct clocks_runtime {
    // guards everything below: the main thread (menu, engine, applet hooks) and the check thread
    std::mutex mutex;

    bool                  svc_ready  = false;
    bool                  use_clkrst = false; // clkrst on 8.0.0+, pcv before
    bool                  sysclk     = false;
    clock_rates           rates[CLOCK_DOMAIN_COUNT];
    std::vector<uint32_t> oc_cpu_steps;

    AppletHookCookie hook_cookie;
    bool             hooked = false;

    std::thread             check_thread;
    std::condition_variable check_wake;
    bool                    check_stop   = false;
    int                     check_logged = 0;

    bool       active       = false;              // something is raised and needs a restore
    bool       fastload     = false;              // we set ApmCpuBoostMode_FastLoad
    bool       saved_docked = false;              // operation mode when saved_hz was read
    uint32_t   saved_hz[CLOCK_DOMAIN_COUNT]    = {}; // rates before the first raise, 0 = unknown
    uint32_t   target_hz[CLOCK_DOMAIN_COUNT]   = {}; // rates set successfully, 0 = not set
    bool       changed[CLOCK_DOMAIN_COUNT]     = {}; // set at least once since the raise: restore it
    int        drift_checks[CLOCK_DOMAIN_COUNT] = {}; // checks in a row that found the rate changed
    clock_plan plan;
};

static clocks_runtime g_clk;

static bool clocks_svc_init() {
    if (g_clk.svc_ready) {
        return true;
    }
    // reference counted by libnx: sysinfo.cpp opens the same services to read the clocks
    if (hosversionAtLeast(8, 0, 0)) {
        if (R_SUCCEEDED(clkrstInitialize())) {
            g_clk.use_clkrst = true;
            g_clk.svc_ready  = true;
        }
    } else if (R_SUCCEEDED(pcvInitialize())) {
        g_clk.use_clkrst = false;
        g_clk.svc_ready  = true;
    }
    return g_clk.svc_ready;
}

static void clocks_svc_exit() {
    if (!g_clk.svc_ready) {
        return;
    }
    if (g_clk.use_clkrst) {
        clkrstExit();
    } else {
        pcvExit();
    }
    g_clk.svc_ready = false;
}

// A clkrst session does not hold the rate: sys-clk and RetroArch also open, set and close right away.
static Result clocks_hw_get(clock_domain d, uint32_t & hz) {
    u32 v = 0;
    Result rc;
    if (g_clk.use_clkrst) {
        ClkrstSession session {};
        rc = clkrstOpenSession(&session, k_domain_ids[d], 3);
        if (R_SUCCEEDED(rc)) {
            rc = clkrstGetClockRate(&session, &v);
            clkrstCloseSession(&session);
        }
    } else {
        rc = pcvGetClockRate(k_domain_modules[d], &v);
    }
    hz = R_SUCCEEDED(rc) ? v : 0;
    return rc;
}

static Result clocks_hw_set(clock_domain d, uint32_t hz) {
    if (!g_clk.use_clkrst) {
        return pcvSetClockRate(k_domain_modules[d], hz);
    }
    ClkrstSession session {};
    Result rc = clkrstOpenSession(&session, k_domain_ids[d], 3);
    if (R_SUCCEEDED(rc)) {
        rc = clkrstSetClockRate(&session, hz);
        clkrstCloseSession(&session);
    }
    return rc;
}

static Result clocks_hw_list(clock_domain d, clock_rates & out) {
    u32 buf[k_rates_max] = {};
    s32 count = 0;
    PcvClockRatesListType type = PcvClockRatesListType_Invalid;
    Result rc;
    if (g_clk.use_clkrst) {
        ClkrstSession session {};
        rc = clkrstOpenSession(&session, k_domain_ids[d], 3);
        if (R_SUCCEEDED(rc)) {
            rc = clkrstGetPossibleClockRates(&session, buf, k_rates_max, &type, &count);
            clkrstCloseSession(&session);
        }
    } else {
        rc = pcvGetPossibleClockRates(k_domain_modules[d], buf, k_rates_max, &type, &count);
    }
    out = clock_rates();
    if (R_SUCCEEDED(rc)) {
        count = std::max(0, std::min(count, (s32) k_rates_max));
        out.hz.assign(buf, buf + count);
        out.range = type == PcvClockRatesListType_Range;
        clock_rates_normalize(out);
    }
    return rc;
}

static std::string clocks_fmt_list(const clock_rates & r) {
    if (r.hz.empty()) {
        return "unknown";
    }
    std::string s = r.range ? "range" : "";
    for (uint32_t hz : r.hz) {
        if (!s.empty()) {
            s += ' ';
        }
        s += clock_fmt_mhz(hz);
    }
    return s;
}

static bool clocks_detect_sysclk() {
    // Atmosphere's sm extension AtmosphereHasService (65100) answers right away, while GetServiceHandle waits
    // for a missing service to be registered. libnx itself talks tipc to sm on Atmosphere.
    if (hosversionIsAtmosphere()) {
        u8 has = 0;
        const SmServiceName name = smEncodeName(k_sysclk_service);
        if (R_SUCCEEDED(tipcDispatchInOut(smGetServiceSessionTipc(), 65100, name, has))) {
            return has != 0;
        }
    }
    return file_exists(k_sysclk_boot2_flag);
}

// caller holds g_clk.mutex
static void clocks_publish_locked(app_state & state) {
    const bool active = g_clk.active;
    const uint32_t cpu = g_clk.target_hz[CLOCK_DOMAIN_CPU];
    const uint32_t emc = g_clk.target_hz[CLOCK_DOMAIN_EMC];
    state.clock_applied       = active ? g_clk.plan.mode : CLOCK_MODE_STOCK;
    // clkrst failed: FastLoad alone still gives its own rates
    state.clock_cpu_target_hz = !active ? 0 : cpu != 0 ? cpu : g_clk.fastload ? k_clock_cpu_official_hz : 0;
    state.clock_emc_target_hz = !active ? 0 : emc != 0 ? emc : g_clk.fastload ? g_clk.plan.fastload_emc_hz : 0;
    state.cpu_boost           = active;
}

// caller holds g_clk.mutex; sets one clock (target_hz[d] = hz on success, 0 on failure) and logs it
static void clocks_set_locked(clock_domain d, uint32_t hz) {
    g_clk.target_hz[d]    = 0;
    g_clk.drift_checks[d] = 0;
    if (!g_clk.svc_ready || hz == 0) {
        return;
    }
    const Result rc = clocks_hw_set(d, hz);
    if (R_FAILED(rc)) {
        engine_log_printf("clocks: %s %s MHz failed: 0x%x\n", k_domain_names[d], clock_fmt_mhz(hz).c_str(), (unsigned) rc);
        return;
    }
    uint32_t now = 0;
    clocks_hw_get(d, now);
    engine_log_printf("clocks: %s -> %s MHz (reads %s MHz)\n", k_domain_names[d],
        clock_fmt_mhz(hz).c_str(), clock_fmt_mhz(now).c_str());
    g_clk.target_hz[d] = hz;
    g_clk.changed[d]   = true;
}

// caller holds g_clk.mutex; state may be null (emergency), quiet = no logging (the log may be locked)
static void clocks_restore_locked(app_state * state, bool quiet) {
    if (g_clk.active || g_clk.fastload) {
        // clkrst first: ending the boost mode below then lets the system apply its configuration on top
        const bool docked = appletGetOperationMode() == AppletOperationMode_Console;
        for (int i = 0; i < CLOCK_DOMAIN_COUNT; ++i) {
            const clock_domain d = (clock_domain) i;
            if (!g_clk.changed[d] || !g_clk.svc_ready) {
                continue;
            }
            // previous rate unknown or read in the other operation mode: the rate of the normal configurations
            const uint32_t stock_hz = d == CLOCK_DOMAIN_CPU ? k_clock_cpu_stock_hz
                : docked ? k_clock_emc_official_hz : k_clock_emc_handheld_hz;
            const bool     saved_ok = g_clk.saved_hz[d] != 0 && g_clk.saved_docked == docked;
            const uint32_t hz       = saved_ok ? g_clk.saved_hz[d] : stock_hz;
            const Result rc = clocks_hw_set(d, hz);
            if (!quiet && R_FAILED(rc)) {
                engine_log_printf("clocks: restoring %s %s MHz failed: 0x%x\n", k_domain_names[d],
                    clock_fmt_mhz(hz).c_str(), (unsigned) rc);
            }
        }
        if (g_clk.fastload) {
            const Result rc = appletSetCpuBoostMode(ApmCpuBoostMode_Normal);
            if (!quiet && R_FAILED(rc)) {
                engine_log_printf("clocks: appletSetCpuBoostMode(Normal) failed: 0x%x\n", (unsigned) rc);
            }
            g_clk.fastload = false;
        }
        if (!quiet) {
            uint32_t cpu = 0;
            uint32_t emc = 0;
            if (g_clk.svc_ready) {
                clocks_hw_get(CLOCK_DOMAIN_CPU, cpu);
                clocks_hw_get(CLOCK_DOMAIN_EMC, emc);
            }
            engine_log_printf("clocks: restored, CPU %s MHz, EMC %s MHz\n", clock_fmt_mhz(cpu).c_str(), clock_fmt_mhz(emc).c_str());
        }
    }
    g_clk.active = false;
    for (int i = 0; i < CLOCK_DOMAIN_COUNT; ++i) {
        g_clk.target_hz[i]    = 0;
        g_clk.changed[i]      = false;
        g_clk.drift_checks[i] = 0;
    }
    g_clk.plan = clock_plan();
    if (state) {
        clocks_publish_locked(*state);
    }
}

// caller holds g_clk.mutex; reassert = request FastLoad again even if it should still be on (after system events)
static void clocks_apply_locked(app_state & state, bool reassert) {
    const bool docked = appletGetOperationMode() == AppletOperationMode_Console;
    const clock_plan plan = clock_plan_make(state.clock_setting, state.clock_oc_cpu_hz,
        g_clk.rates[CLOCK_DOMAIN_CPU], g_clk.rates[CLOCK_DOMAIN_EMC], docked);
    if (plan.mode == CLOCK_MODE_STOCK) {
        clocks_restore_locked(&state, false);
        return;
    }

    if (!g_clk.active) {
        // what clocks_restore goes back to
        g_clk.saved_docked = docked;
        for (int i = 0; i < CLOCK_DOMAIN_COUNT; ++i) {
            g_clk.saved_hz[i] = 0;
            if (g_clk.svc_ready) {
                clocks_hw_get((clock_domain) i, g_clk.saved_hz[i]);
            }
        }
        engine_log_printf("clocks: raising (%s, %s), was CPU %s MHz, EMC %s MHz\n", clock_mode_name(plan.mode),
            docked ? "docked" : "handheld", clock_fmt_mhz(g_clk.saved_hz[CLOCK_DOMAIN_CPU]).c_str(),
            clock_fmt_mhz(g_clk.saved_hz[CLOCK_DOMAIN_EMC]).c_str());
    } else if (plan.mode != g_clk.plan.mode || plan.cpu_hz != g_clk.plan.cpu_hz || plan.emc_hz != g_clk.plan.emc_hz) {
        engine_log_printf("clocks: switching to %s\n", clock_mode_name(plan.mode));
    }

    // FastLoad first: the system applies its boost configuration, then clkrst sets the exact rates on top
    if (!g_clk.fastload || reassert) {
        const Result rc = appletSetCpuBoostMode(ApmCpuBoostMode_FastLoad);
        if (R_SUCCEEDED(rc)) {
            if (!g_clk.fastload) {
                engine_log_printf("clocks: boost mode FastLoad on\n");
            }
            g_clk.fastload = true;
        } else {
            engine_log_printf("clocks: appletSetCpuBoostMode(FastLoad) failed: 0x%x\n", (unsigned) rc);
        }
    }

    clocks_set_locked(CLOCK_DOMAIN_CPU, plan.cpu_hz);
    clocks_set_locked(CLOCK_DOMAIN_EMC, plan.emc_hz);

    g_clk.plan   = plan;
    g_clk.active = g_clk.fastload || g_clk.target_hz[CLOCK_DOMAIN_CPU] != 0 || g_clk.target_hz[CLOCK_DOMAIN_EMC] != 0;
    if (!g_clk.active) {
        engine_log_printf("clocks: could not raise the clocks\n");
    }
    clocks_publish_locked(state);
}

// Runs on the main thread from appletMainLoop (ui_app_running). The system has just reset the clocks or is
// about to; the check thread catches the second case.
static void clocks_hook(AppletHookType hook, void * param) {
    const char * why = nullptr;
    switch (hook) {
        case AppletHookType_OnFocusState:
            if (appletGetFocusState() != AppletFocusState_InFocus) {
                return;
            }
            why = "focus regained";
            break;
        case AppletHookType_OnResume:
            why = "resumed";
            break;
        case AppletHookType_OnOperationMode:
            why = appletGetOperationMode() == AppletOperationMode_Console ? "docked" : "undocked";
            break;
        case AppletHookType_OnPerformanceMode:
            why = "performance mode changed";
            break;
        default:
            return;
    }
    app_state & state = *(app_state *) param;
    std::lock_guard<std::mutex> lock(g_clk.mutex);
    if (!g_clk.active) {
        return;
    }
    engine_log_printf("clocks: %s, re-applying\n", why);
    clocks_apply_locked(state, true);
}

static void clocks_check_main() {
    std::unique_lock<std::mutex> lock(g_clk.mutex);
    while (!g_clk.check_stop) {
        g_clk.check_wake.wait_for(lock, std::chrono::milliseconds(k_check_interval_ms));
        // out of focus (HOME menu, sleep, a library applet such as the keyboard) the system owns the clocks
        if (g_clk.check_stop || !g_clk.active || appletGetFocusState() != AppletFocusState_InFocus) {
            continue;
        }
        for (int i = 0; i < CLOCK_DOMAIN_COUNT; ++i) {
            const clock_domain d = (clock_domain) i;
            const uint32_t want = g_clk.target_hz[d];
            // not raised, or given up until the next clocks_apply_locked
            if (want == 0 || g_clk.drift_checks[d] > k_check_retry_max) {
                continue;
            }
            uint32_t now = 0;
            if (R_FAILED(clocks_hw_get(d, now)) || now == 0) {
                continue;
            }
            const uint32_t diff = now > want ? now - want : want - now;
            if (diff <= k_check_tolerance_hz) {
                g_clk.drift_checks[d] = 0;
                continue;
            }
            // the target stays as it is: the menu shows the clock in yellow while it does not hold
            if (++g_clk.drift_checks[d] > k_check_retry_max) {
                engine_log_printf("clocks: %s keeps reading %s MHz instead of %s MHz, not re-applying it any more\n",
                    k_domain_names[d], clock_fmt_mhz(now).c_str(), clock_fmt_mhz(want).c_str());
                continue;
            }
            const bool log = g_clk.check_logged < k_check_log_max;
            if (log) {
                ++g_clk.check_logged;
                engine_log_printf("clocks: %s reads %s MHz instead of %s MHz, re-applying%s\n", k_domain_names[d],
                    clock_fmt_mhz(now).c_str(), clock_fmt_mhz(want).c_str(),
                    g_clk.check_logged == k_check_log_max ? " (not logged any more)" : "");
            }
            const Result rc = clocks_hw_set(d, want);
            if (log && R_FAILED(rc)) {
                engine_log_printf("clocks: %s %s MHz failed: 0x%x\n", k_domain_names[d], clock_fmt_mhz(want).c_str(), (unsigned) rc);
            }
        }
    }
}

std::string clocks_init(app_state & state) {
    std::string note;
    app_settings s;
    std::string err;
    if (!settings_load(k_settings_path, s, err)) {
        note = "settings.json invalid (" + err + "), using defaults";
    } else if (!err.empty()) {
        note = "settings.json warning: " + err;
    }
    state.clock_setting   = s.cpu_clock;
    state.clock_oc_cpu_hz = s.overclock_cpu_hz;

    std::lock_guard<std::mutex> lock(g_clk.mutex);
    if (clocks_svc_init()) {
        for (int i = 0; i < CLOCK_DOMAIN_COUNT; ++i) {
            const Result rc = clocks_hw_list((clock_domain) i, g_clk.rates[i]);
            if (R_FAILED(rc)) {
                engine_log_printf("clocks: %s rate list failed: 0x%x\n", k_domain_names[i], (unsigned) rc);
            }
            engine_log_printf("clocks: %s rates (MHz): %s\n", k_domain_names[i], clocks_fmt_list(g_clk.rates[i]).c_str());
        }
    } else {
        engine_log_printf("clocks: %s unavailable, only the boost mode can raise the clocks\n",
            hosversionAtLeast(8, 0, 0) ? "clkrst" : "pcv");
    }
    g_clk.oc_cpu_steps       = clock_rates_above(g_clk.rates[CLOCK_DOMAIN_CPU], k_clock_cpu_official_hz);
    g_clk.sysclk             = clocks_detect_sysclk();
    state.clock_oc_available = clock_oc_available(g_clk.rates[CLOCK_DOMAIN_CPU], g_clk.rates[CLOCK_DOMAIN_EMC]);
    state.clock_sysclk       = g_clk.sysclk;

    engine_log_printf("clocks: setting %s, overclock %s, sys-clk %s\n", clock_mode_name(state.clock_setting),
        state.clock_oc_available ? "available" : "not available", g_clk.sysclk ? "detected" : "not detected");

    if (state.clock_setting == CLOCK_MODE_OVERCLOCK && !state.clock_oc_available) {
        // settings.json keeps OVERCLOCK: it comes back once sys-clk-OC is installed again
        state.clock_setting = CLOCK_MODE_BOOST;
        note += std::string(note.empty() ? "" : "; ") + "Overclock not available (needs sys-clk-OC), using Boost";
    }
    if (g_clk.sysclk) {
        note += std::string(note.empty() ? "" : "; ") + "sys-clk detected: it may override the CPU clock setting";
    }

    if (!g_clk.hooked) {
        appletHook(&g_clk.hook_cookie, clocks_hook, &state);
        g_clk.hooked = true;
    }
    // sys-clk re-applies its own profiles every few hundred ms: a check thread would only fight it.
    // Without clkrst/pcv there is nothing it could check.
    if (g_clk.svc_ready && !g_clk.sysclk && !g_clk.check_thread.joinable()) {
        g_clk.check_stop   = false;
        g_clk.check_thread = std::thread(clocks_check_main);
    }
    return note;
}

void clocks_apply(app_state & state) {
    std::lock_guard<std::mutex> lock(g_clk.mutex);
    clocks_apply_locked(state, false);
}

void clocks_restore(app_state & state) {
    std::lock_guard<std::mutex> lock(g_clk.mutex);
    clocks_restore_locked(&state, false);
}

void clocks_restore_emergency() {
    if (!g_clk.mutex.try_lock()) {
        return;
    }
    clocks_restore_locked(nullptr, true);
    g_clk.mutex.unlock();
}

void clocks_exit(app_state & state) {
    {
        std::lock_guard<std::mutex> lock(g_clk.mutex);
        clocks_restore_locked(&state, false);
        g_clk.check_stop = true;
    }
    g_clk.check_wake.notify_all();
    if (g_clk.check_thread.joinable()) {
        g_clk.check_thread.join();
    }

    std::lock_guard<std::mutex> lock(g_clk.mutex);
    if (g_clk.hooked) {
        appletUnhook(&g_clk.hook_cookie);
        g_clk.hooked = false;
    }
    clocks_svc_exit();
}

const std::vector<uint32_t> & clocks_oc_cpu_steps() {
    return g_clk.oc_cpu_steps; // written once by clocks_init
}

clock_plan clocks_plan_for(const app_state & state) {
    std::lock_guard<std::mutex> lock(g_clk.mutex);
    return clock_plan_make(state.clock_setting, state.clock_oc_cpu_hz, g_clk.rates[CLOCK_DOMAIN_CPU],
        g_clk.rates[CLOCK_DOMAIN_EMC], appletGetOperationMode() == AppletOperationMode_Console);
}

bool clocks_set_setting(app_state & state, clock_mode mode, uint32_t oc_cpu_hz, std::string & err) {
    err.clear();
    {
        std::lock_guard<std::mutex> lock(g_clk.mutex);
        state.clock_setting   = mode;
        state.clock_oc_cpu_hz = oc_cpu_hz;
        std::string cap;
        if (mode == CLOCK_MODE_OVERCLOCK && oc_cpu_hz != 0) {
            cap = ", CPU up to " + clock_fmt_mhz(oc_cpu_hz) + " MHz";
        }
        engine_log_printf("clocks: setting %s%s\n", clock_mode_name(mode), cap.c_str());
        if (state.model) {
            clocks_apply_locked(state, false);
        }
    }
    app_settings s;
    s.cpu_clock        = mode;
    s.overclock_cpu_hz = oc_cpu_hz;
    return settings_save(k_settings_path, s, err);
}
