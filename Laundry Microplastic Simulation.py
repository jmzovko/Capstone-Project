#!/usr/bin/env python3
#test
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

PSI_TO_PA = 6894.757
GPM_TO_M3S = 0.00378541/60.0

@dataclass
class FilterParams:
    area_m2: float = 0.002025802
    thickness_m: float = 0.0075
    mu_Pa_s: float = 0.001
    k0_m2: float = 9.5e-11
    alpha_clog_per_g: float = 0.01
    eff0: float = 0.6
    eff_max: float = 0.98
    gamma_eff_per_g: float = 0.0005
    dP_cleaning_threshold_Pa: float = 10*PSI_TO_PA
    baseline_replacement_threshold_Pa: float = 8*PSI_TO_PA
    residual_fouling_per_clean: float = 0.12
    Q_pump_m3s: float = 10*GPM_TO_M3S

@dataclass
class ProcessParams:
    baseline_C_mg_per_L: float = 3.25
    per_load_cv: float = 2
    within_load_spike_minute: int = 10
    within_load_spike_duration_min: int = 15
    within_load_spike_add_mgL: float = 1.50
    duration_hours: float = 280.0
    dt_seconds: float = 10.0
    p_breakthrough_per_min: float = 0.01
    breakthrough_drop_frac: float = 0.25
    eff_jitter_sd: float = 0.02

def simulate_cleaning(f: FilterParams, p: ProcessParams, seed: None = None):
    rng = np.random.default_rng(seed)
    A, L, mu = f.area_m2, f.thickness_m, f.mu_Pa_s
    k0 = f.k0_m2; alpha = f.alpha_clog_per_g
    Q = f.Q_pump_m3s
    dp_base_clean_pa = mu*L*Q/(k0*A)

    T = int(np.ceil(p.duration_hours*3600)); dt = p.dt_seconds
    n = int(np.ceil(T/dt)); t = np.arange(n)*dt; hours_index=(t//3600).astype(int)

    H = int(np.ceil(p.duration_hours))
    if p.per_load_cv <= 0:
        multipliers = np.ones(H)
    else:
        sigma2 = np.log(1 + p.per_load_cv**2); mu_log = -0.5*sigma2
        multipliers = rng.lognormal(mean=mu_log, sigma=np.sqrt(sigma2), size=H)

    Cin = np.zeros(n)
    for i in range(n):
        h = hours_index[i]
        base = p.baseline_C_mg_per_L*multipliers[min(h,H-1)]
        minute = int((t[i]-h*3600)//60)
        if p.within_load_spike_duration_min>0:
            start=p.within_load_spike_minute; end=start+p.within_load_spike_duration_min
            if start<=minute<end: base += p.within_load_spike_add_mgL
        Cin[i] = max(base, 0.0)

    dP = np.zeros(n); eff = np.zeros(n)
    Mgen = np.zeros(n); Mcap = np.zeros(n); Mrel = np.zeros(n)
    cleanings=[]; replacements=[]; baseline_trace_pa=np.zeros(n)

    Mcap_loading=0.0
    Mgen_tot=Mcap_tot=Mrel_tot=0.0
    phi = 1.0

    def permeability(phi, M): return k0/(phi*(1.0 + alpha*M))
    def eff_det(M): return np.clip(f.eff0 + (f.eff_max-f.eff0)*(1 - np.exp(-f.gamma_eff_per_g*M)), 0.0, 1.0)

    for i in range(n):
        k = permeability(phi, Mcap_loading)
        e = eff_det(Mcap_loading)
        e = float(np.clip(e + rng.normal(0, p.eff_jitter_sd), 0.0, 1.0))
        if rng.random() < p.p_breakthrough_per_min*(dt/60.0):
            e *= (1.0 - p.breakthrough_drop_frac)
        eff[i] = e

        dP[i] = mu*L*Q/(k*A)

        m_in = Q*Cin[i]
        m_cap = e*m_in; m_rel=(1-e)*m_in
        Mgen_tot += m_in*dt; Mcap_tot += m_cap*dt; Mrel_tot += m_rel*dt
        Mcap_loading += m_cap*dt

        Mgen[i]=Mgen_tot; Mcap[i]=Mcap_tot; Mrel[i]=Mrel_tot
        baseline_trace_pa[i] = dp_base_clean_pa*phi

        if dP[i] >= f.dP_cleaning_threshold_Pa:
            Mcap_loading = 0.0
            phi *= (1.0 + f.residual_fouling_per_clean)
            cleanings.append(t[i])
            if dp_base_clean_pa*phi >= f.baseline_replacement_threshold_Pa:
                phi = 1.0
                replacements.append(t[i])

    return {
        "t_s": t, "dP_Pa": dP, "eff": eff, "Cin_gm3": Cin,
        "M_generated_g": Mgen, "M_captured_g": Mcap, "M_released_g": Mrel,
        "cleanings_s": cleanings, "replacements_s": replacements,
        "baseline_dp_pa_trace": baseline_trace_pa, "dp_base_clean_pa": dp_base_clean_pa
    }

if __name__ == "__main__":
    f = FilterParams(); p = ProcessParams()
    res = simulate_cleaning(f, p, seed=None)
    t_hr = res["t_s"]/3600

    plt.figure(); plt.plot(t_hr, res["M_generated_g"], label="Generated")
    plt.plot(t_hr, res["M_captured_g"], label="Captured")
    plt.plot(t_hr, res["M_released_g"], label="Released")
    plt.legend(); plt.xlabel("Loads of Laundry"); plt.ylabel("Cumulative [g]")
    plt.title("Mass Balances — cleaning + imperfect restoration"); plt.grid(True); plt.tight_layout()

    plt.figure(); plt.plot(t_hr, res["dP_Pa"], label="ΔP")
    plt.plot(t_hr, res["baseline_dp_pa_trace"], linestyle="--", label="Baseline ΔP if cleaned now")
    plt.xlabel("Loads of Laundry"); plt.ylabel("ΔP [Pa]")
    plt.title("Pressure Drop & Baseline Drift (cleaning events raise baseline)")
    plt.legend(); plt.grid(True); plt.tight_layout()

    plt.figure(); plt.plot(t_hr, res["eff"])
    plt.xlabel("Loads of Laundry"); plt.ylabel("Capture Efficiency")
    plt.title("Filter Capture Efficiency Over Time"); plt.grid(True); plt.tight_layout()

    print("Cleanings (hr):", [round(x/3600,2) for x in res["cleanings_s"]])
    print("Replacements (hr):", [round(x/3600,2) for x in res["replacements_s"]])
    print("Start baseline [psi]:", res["dp_base_clean_pa"]/PSI_TO_PA,
          "| End baseline [psi]:", res["baseline_dp_pa_trace"][-1]/PSI_TO_PA)
    plt.show()
