// Rate selection (portable) and the host versions of the runtime functions.
// The Switch runtime (libnx) lives in clocks_switch.cpp.

#include "clocks.h"

#include <algorithm>
#include <cstdio>

//
// rate selection (pure)
//

void clock_rates_normalize(clock_rates & r) {
    std::vector<uint32_t> out;
    out.reserve(r.hz.size());
    for (uint32_t hz : r.hz) {
        if (hz != 0) {
            out.push_back(hz);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    r.hz = out;
    if (r.hz.size() < 2) {
        r.range = false;
    }
}

// highest allowed rate <= cap_hz, 0 if there is none or the list is unknown
static uint32_t clock_rates_pick(const clock_rates & r, uint32_t cap_hz) {
    if (r.hz.empty()) {
        return 0;
    }
    if (r.range) {
        return cap_hz < r.hz.front() ? 0 : std::min(cap_hz, r.hz.back());
    }
    uint32_t best = 0;
    for (uint32_t hz : r.hz) {
        if (hz <= cap_hz) {
            best = hz; // ascending: the last match is the highest
        }
    }
    return best;
}

std::vector<uint32_t> clock_rates_above(const clock_rates & r, uint32_t min_hz) {
    std::vector<uint32_t> out;
    if (r.range) {
        if (!r.hz.empty() && r.hz.back() > min_hz) {
            out.push_back(r.hz.back());
        }
        return out;
    }
    for (uint32_t hz : r.hz) {
        if (hz > min_hz) {
            out.push_back(hz);
        }
    }
    return out;
}

bool clock_oc_available(const clock_rates & cpu, const clock_rates & emc) {
    return !clock_rates_above(cpu, k_clock_cpu_official_hz).empty() ||
           !clock_rates_above(emc, k_clock_emc_official_hz).empty();
}

clock_plan clock_plan_make(clock_mode mode, uint32_t oc_cpu_hz, const clock_rates & cpu, const clock_rates & emc, bool docked) {
    clock_plan plan;
    if (mode != CLOCK_MODE_BOOST && mode != CLOCK_MODE_OVERCLOCK) {
        return plan; // STOCK (and anything invalid): leave the clocks alone
    }

    plan.mode            = CLOCK_MODE_BOOST;
    plan.cpu_hz          = clock_rates_pick(cpu, k_clock_cpu_official_hz);
    plan.emc_hz          = clock_rates_pick(emc, k_clock_emc_official_hz);
    plan.fastload_emc_hz = docked ? k_clock_emc_official_hz : k_clock_emc_handheld_hz;
    // unknown lists: the official rates are always valid
    if (plan.cpu_hz == 0) {
        plan.cpu_hz = k_clock_cpu_official_hz;
    }
    if (plan.emc_hz == 0) {
        plan.emc_hz = k_clock_emc_official_hz;
    }

    if (mode == CLOCK_MODE_OVERCLOCK && clock_oc_available(cpu, emc)) {
        plan.mode   = CLOCK_MODE_OVERCLOCK;
        plan.emc_hz = std::max(plan.emc_hz, clock_rates_pick(emc, UINT32_MAX));
        const std::vector<uint32_t> steps = clock_rates_above(cpu, k_clock_cpu_official_hz);
        if (!steps.empty()) {
            // like clock_step_setting: a cap below every step means the lowest step
            const uint32_t hz = clock_rates_pick(cpu, oc_cpu_hz != 0 ? oc_cpu_hz : UINT32_MAX);
            plan.cpu_hz = hz > k_clock_cpu_official_hz ? hz : steps.front();
        }
    }
    return plan;
}

clock_step clock_step_setting(clock_mode mode, uint32_t oc_cpu_hz, const std::vector<uint32_t> & oc_cpu_steps,
        bool oc_available, int dir) {
    clock_step r;
    r.mode      = mode;
    r.oc_cpu_hz = oc_cpu_hz;
    if (dir == 0) {
        return r;
    }

    switch (mode) {
        case CLOCK_MODE_STOCK:
            if (dir > 0) {
                r.mode = CLOCK_MODE_BOOST;
            }
            break;
        case CLOCK_MODE_BOOST:
            if (dir < 0) {
                r.mode = CLOCK_MODE_STOCK;
            } else if (!oc_available) {
                r.unavailable = true;
            } else {
                r.mode      = CLOCK_MODE_OVERCLOCK;
                r.oc_cpu_hz = 0;
                r.confirm   = true;
            }
            break;
        case CLOCK_MODE_OVERCLOCK: {
            if (!oc_available || oc_cpu_steps.empty()) {
                // a single Overclock value (only the EMC can go higher), or a stale setting
                if (dir < 0) {
                    r.mode      = CLOCK_MODE_BOOST;
                    r.oc_cpu_hz = 0;
                } else if (!oc_available) {
                    r.unavailable = true;
                }
                break;
            }
            // current step: the highest one <= the cap (no cap = the last one, a cap below all = the first one)
            const size_t n = oc_cpu_steps.size();
            size_t cur = n - 1;
            if (oc_cpu_hz != 0) {
                cur = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (oc_cpu_steps[i] <= oc_cpu_hz) {
                        cur = i;
                    }
                }
            }
            if (dir > 0) {
                const size_t next = std::min(cur + 1, n - 1);
                r.oc_cpu_hz = next == n - 1 ? 0 : oc_cpu_steps[next];
            } else if (cur > 0) {
                r.oc_cpu_hz = oc_cpu_steps[cur - 1];
            } else {
                r.mode      = CLOCK_MODE_BOOST;
                r.oc_cpu_hz = 0;
            }
        } break;
        default:
            r.mode      = CLOCK_MODE_BOOST;
            r.oc_cpu_hz = 0;
            break;
    }
    return r;
}

const char * clock_mode_name(clock_mode mode) {
    switch (mode) {
        case CLOCK_MODE_STOCK:     return "Stock";
        case CLOCK_MODE_BOOST:     return "Boost";
        case CLOCK_MODE_OVERCLOCK: return "Overclock";
        default:                   return "?";
    }
}

std::string clock_fmt_mhz(uint32_t hz) {
    const uint64_t tenths = ((uint64_t) hz + 50000) / 100000; // 0.1 MHz, rounded
    char buf[32];
    if (tenths % 10 == 0) {
        std::snprintf(buf, sizeof(buf), "%u", (unsigned) (tenths / 10));
    } else {
        std::snprintf(buf, sizeof(buf), "%u.%u", (unsigned) (tenths / 10), (unsigned) (tenths % 10));
    }
    return std::string(buf);
}

//
// runtime: host versions (nothing to control)
//

#ifndef __SWITCH__

std::string clocks_init(app_state & state) {
    (void) state;
    return std::string();
}

void clocks_apply(app_state & state) {
    state.clock_applied       = CLOCK_MODE_STOCK;
    state.clock_cpu_target_hz = 0;
    state.clock_emc_target_hz = 0;
    state.cpu_boost           = false;
}

void clocks_restore(app_state & state) {
    clocks_apply(state);
}

void clocks_restore_emergency() {
}

void clocks_exit(app_state & state) {
    clocks_restore(state);
}

const std::vector<uint32_t> & clocks_oc_cpu_steps() {
    static const std::vector<uint32_t> none;
    return none;
}

clock_plan clocks_plan_for(const app_state & state) {
    return clock_plan_make(state.clock_setting, state.clock_oc_cpu_hz, clock_rates(), clock_rates(), false);
}

bool clocks_set_setting(app_state & state, clock_mode mode, uint32_t oc_cpu_hz, std::string & err) {
    err.clear();
    state.clock_setting   = mode;
    state.clock_oc_cpu_hz = oc_cpu_hz;
    return true;
}

#endif
