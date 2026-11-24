"""보고서용 그림과 수치표를 한 번에 생성하는 스크립트.

    python make_figures.py

task1~3의 함수를 불러와 figures/ 폴더에 그림(PNG)을, figures/results.json에 수치를 저장합니다.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuron_model import find_electric_field_threshold, simulate_hh_neuron
from tms_config import (
    DIDT_RISE,
    E_GOAL_TRADEOFF,
    FIG_DIR,
    H_NOMINAL,
    HEAD_LAYERS,
    I_PEAK_NOMINAL,
    MU0,
    N_TURNS,
    R_CIRC,
    R_FIG8,
    R_OFF_TARGET,
    R_TARGET,
    S_FIG8,
    T_MS,
    T_S,
    WIRE_RADIUS,
    Z_TARGET,
)
from task1_coil_field import (
    analytical_circular_on_axis_Bz,
    build_circular_coil,
    build_figure8_coil,
    compute_inductance,
    compute_unit_A_and_B,
    make_circular_loop,
    trapezoidal_pulse_current_and_didt,
    verify_curl_A,
)
from task2_tissue_induction import (
    compute_cortical_metrics,
    compute_depth_profile,
    compute_target_waveforms,
    sweep_lateral_offset,
    sweep_standoff,
    sweep_wing_radius,
)
from task3_neuron import compute_E_eff_waveform, compute_fig8_thresholds, find_max_standoff

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C_CIRC = "#1f77b4"
C_FIG8 = "#d62728"
RES: dict = {}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def shade_layers(ax, horizontal: bool = True) -> None:
    """깊이 축(x축, mm)에 4층 조직을 색으로 표시."""
    for layer in HEAD_LAYERS:
        top, bot = -layer["z_top"] * 1e3, -layer["z_bot"] * 1e3
        ax.axvspan(top, min(bot, 60.0), color=layer["color"], alpha=0.55, lw=0, zorder=0)
        ax.text(0.5 * (top + min(bot, 60.0)) if layer["name"] != "Brain" else 37.0, 0.97,
                layer["name"].split(" ")[0], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#444")


# ----------------------------------------------------------------------------------
circ_pts, circ_dl = build_circular_coil()
fig8_pts, fig8_dl = build_figure8_coil()
i_t, di_dt = trapezoidal_pulse_current_and_didt(T_S)

# ============================== Task 1.1 ==========================================
z_axis = np.linspace(-60e-3, 0.0, 121)
obs_axis = np.column_stack([np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis])
_, b_axis = compute_unit_A_and_B(circ_pts, circ_dl, obs_axis)
bz_num = b_axis[:, 2] * I_PEAK_NOMINAL
bz_ana = analytical_circular_on_axis_Bz(z_axis, current=I_PEAK_NOMINAL)
rel = np.abs(bz_num - bz_ana) / np.abs(bz_ana)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(z_axis * 1e3, bz_ana, "-", color="k", lw=1.5, label="Analytical")
ax1.plot(z_axis[::4] * 1e3, bz_num[::4], "o", color=C_CIRC, ms=4.5, mfc="none", label="Biot-Savart (K = 180)")
ax1.set_xlabel("z [mm]"); ax1.set_ylabel(r"$B_z(0,0,z)$ [T]  at  $I$ = 6 kA")
ax1.set_title("(a) On-axis $B_z$ of the circular coil"); ax1.legend()
ax2.semilogy(z_axis * 1e3, rel, color=C_CIRC)
ax2.axhline(1e-4, color="gray", ls="--", lw=1); ax2.text(-59, 1.1e-4, "goal $10^{-4}$", fontsize=8, color="gray")
ax2.set_ylim(1e-5, 3e-4)
ax2.set_xlabel("z [mm]"); ax2.set_ylabel("relative error"); ax2.set_title("(b) Relative error")
save(fig, "fig1_onaxis_Bz.png")

r_circ_check = np.array([R_CIRC, 0.0, Z_TARGET])
b_c, curl_c, err_c = verify_curl_A(circ_pts, circ_dl, r_circ_check)
b_c40, curl_c40, err_c40 = verify_curl_A(circ_pts, circ_dl, R_OFF_TARGET)
b_f, curl_f, err_f = verify_curl_A(fig8_pts, fig8_dl, R_TARGET)
RES["task1_1"] = {
    "max_rel_err": float(rel.max()),
    "Bz_scalp_T": float(bz_ana[-1]), "Bz_cortex_T": float(analytical_circular_on_axis_Bz(np.array([Z_TARGET]), current=I_PEAK_NOMINAL)[0]),
    "curl_circ_45": {"B": b_c.tolist(), "curl": curl_c.tolist(), "err": err_c},
    "curl_circ_40": {"B": b_c40.tolist(), "curl": curl_c40.tolist(), "err": err_c40},
    "curl_fig8": {"B": b_f.tolist(), "curl": curl_f.tolist(), "err": err_f},
}

# ============================== Task 1.2 ==========================================
wing1 = make_circular_loop(-S_FIG8, 0.0, H_NOMINAL, R_FIG8, +1.0)
wing2 = make_circular_loop(+S_FIG8, 0.0, H_NOMINAL, R_FIG8, -1.0)
l_circ = compute_inductance(circ_pts, circ_dl)
l1 = compute_inductance(*wing1)
l2 = compute_inductance(*wing2)
m12 = compute_inductance(*wing1, *wing2)
l_fig8 = compute_inductance(fig8_pts, fig8_dl)
l_thin = MU0 * N_TURNS ** 2 * R_CIRC * (np.log(8 * R_CIRC / WIRE_RADIUS) - 1.75)
l_thin_wing = MU0 * N_TURNS ** 2 * R_FIG8 * (np.log(8 * R_FIG8 / WIRE_RADIUS) - 1.75)
RES["task1_2"] = {
    "L_circ_uH": l_circ * 1e6, "L_thin_uH": l_thin * 1e6, "L_err_pct": (l_circ - l_thin) / l_thin * 100,
    "L1_uH": l1 * 1e6, "L2_uH": l2 * 1e6, "M12_uH": m12 * 1e6, "L_thin_wing_uH": l_thin_wing * 1e6,
    "L_sum_uH": (l1 + l2 + 2 * m12) * 1e6, "L_fig8_uH": l_fig8 * 1e6,
    "k_coupling": m12 / np.sqrt(l1 * l2),
    "V_circ_kV": l_circ * DIDT_RISE * 1e-3, "V_fig8_kV": l_fig8 * DIDT_RISE * 1e-3,
    "W_circ_J": 0.5 * l_circ * I_PEAK_NOMINAL ** 2, "W_fig8_J": 0.5 * l_fig8 * I_PEAK_NOMINAL ** 2,
}

win = T_MS <= 0.30
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
ax1.plot(T_MS[win], i_t[win] * 1e-3, color="k", lw=1.8)
ax1.set_ylabel("coil current $I(t)$ [kA]"); ax1.set_title("(a) Trapezoidal coil current")
ax2.step(T_MS[win], l_circ * di_dt[win] * 1e-3, where="post", color=C_CIRC, lw=1.8,
         label=f"Circular ($L$ = {l_circ*1e6:.2f} $\\mu$H)")
ax2.step(T_MS[win], l_fig8 * di_dt[win] * 1e-3, where="post", color=C_FIG8, lw=1.8,
         label=f"Figure-8 ($L$ = {l_fig8*1e6:.2f} $\\mu$H)")
ax2.set_xlabel("t [ms]"); ax2.set_ylabel(r"$V_{coil}(t) = L\,dI/dt$ [kV]")
ax2.set_title("(b) Driving voltage (bipolar rectangular)"); ax2.legend(loc="upper right")
save(fig, "fig2_current_voltage.png")

# ============================== Task 2.2 ==========================================
z_line, e_f8, j_f8 = compute_depth_profile(fig8_pts, fig8_dl, x_line=0.0)
_, e_cc, j_cc = compute_depth_profile(circ_pts, circ_dl, x_line=R_CIRC)
depth = -z_line * 1e3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax in (ax1, ax2):
    shade_layers(ax)
    ax.set_xlim(0, 60); ax.set_xlabel("depth below scalp, $-z$ [mm]")
ax1.plot(depth, e_cc, color=C_CIRC, lw=1.8, label="Circular (x = 45 mm)")
ax1.plot(depth, e_f8, color=C_FIG8, lw=1.8, label="Figure-8 (x = 0)")
ax1.set_ylabel(r"$|\mathbf{E}(z)|$ [V/m]"); ax1.set_title(r"(a) Induced field $|\mathbf{E}|$ (rise phase)")
ax1.legend(loc="center right")
ax2.plot(depth, j_cc, color=C_CIRC, lw=1.8, label="Circular (x = 45 mm)")
ax2.plot(depth, j_f8, color=C_FIG8, lw=1.8, label="Figure-8 (x = 0)")
ax2.set_ylabel(r"$|\mathbf{J}(z)| = \sigma(z)|\mathbf{E}(z)|$ [A/m$^2$]")
ax2.set_title(r"(b) Eddy current density $|\mathbf{J}|$"); ax2.legend(loc="center right")
save(fig, "fig3_depth_profile.png")


def at_depth(arr, z_mm, side):
    """층 경계 바로 위(side=-1) 또는 바로 아래(side=+1)의 값."""
    k = int(np.argmin(np.abs(depth - z_mm)))
    return float(arr[k + side])


RES["task2_2a"] = {
    name: {
        "E_scalp": float(e[0]), "E_skull_top": at_depth(e, 5.0, 0), "E_csf_top": at_depth(e, 12.0, 0),
        "E_brain_top": at_depth(e, 15.0, 0), "E_20mm": float(e[int(np.argmin(np.abs(depth - 20.0)))]),
        "J_scalp_max": float(j[depth <= 5.0].max()),
        "J_skull_max": float(j[(depth > 5.0) & (depth <= 12.0)].max()),
        "J_csf_max": float(j[(depth > 12.0) & (depth <= 15.0)].max()),
        "J_brain_max": float(j[depth > 15.0].max()),
        "J_peak": float(j.max()),
    }
    for name, e, j in [("fig8", e_f8, j_f8), ("circ", e_cc, j_cc)]
}

b_t, e_t = compute_target_waveforms(fig8_pts, fig8_dl, R_TARGET, i_t, di_dt)
b_tc, e_tc = compute_target_waveforms(circ_pts, circ_dl, R_TARGET, i_t, di_dt)
hold = (T_S >= 0.07e-3) & (T_S < 0.11e-3)
RES["task2_2b"] = {
    "fig8_B_hold_T": float(b_t[hold].max()), "fig8_E_hold": float(e_t[hold].max()), "fig8_E_rise": float(e_t.max()),
    "circ_B_hold_T": float(b_tc[hold].max()), "circ_E_rise": float(e_tc.max()),
}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, (bb, ee, ttl) in zip(axes, [(b_t, e_t, "(a) Figure-8 coil, target (0, 0, $-$20) mm"),
                                    (b_tc, e_tc, "(b) Circular coil, target (0, 0, $-$20) mm")]):
    for t0, t1, lab, col in [(0, 0.06, "rise", "#fde0dd"), (0.06, 0.12, "hold", "#eeeeee"), (0.12, 0.18, "fall", "#deebf7")]:
        ax.axvspan(t0, t1, color=col, lw=0, zorder=0)
        ax.text(0.5 * (t0 + t1), 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color="#555")
    ax.plot(T_MS[win], bb[win], color="#2c3e50", lw=1.8)
    ax.set_xlabel("t [ms]"); ax.set_ylabel(r"$|\mathbf{B}(t)|$ [T]", color="#2c3e50"); ax.set_ylim(0, 1.0)
    ax.set_title(ttl, pad=16); ax.grid(False)
    axr = ax.twinx(); axr.spines["right"].set_visible(True)
    axr.step(T_MS[win], ee[win], where="post", color="#e67e22", lw=1.8)
    axr.set_ylabel(r"$|\mathbf{E}(t)|$ [V/m]", color="#e67e22"); axr.set_ylim(0, 280); axr.grid(False)
save(fig, "fig4_B_E_waveforms.png")

xs_2d = np.linspace(-120e-3, 120e-3, 121)
ys_2d = np.linspace(-120e-3, 120e-3, 121)
m_c = compute_cortical_metrics(circ_pts, circ_dl, DIDT_RISE, xs_2d, ys_2d, R_CIRC, l_circ)
m_f = compute_cortical_metrics(fig8_pts, fig8_dl, DIDT_RISE, xs_2d, ys_2d, 0.0, l_fig8)
RES["task2_2c"] = {k: {kk: vv for kk, vv in m.items() if kk != "emag_2d"} for k, m in [("circ", m_c), ("fig8", m_f)]}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
vmax = max(m_c["E_max_cortex_Vm"], m_f["E_max_cortex_Vm"])
for ax, m, ttl, coil in zip(axes, [m_c, m_f], ["(a) Circular coil", "(b) Figure-8 coil"], ["c", "f"]):
    pc = ax.pcolormesh(xs_2d * 1e3, ys_2d * 1e3, m["emag_2d"].T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    ax.contour(xs_2d * 1e3, ys_2d * 1e3, m["emag_2d"].T, levels=[0.5 * m["E_max_cortex_Vm"]],
               colors="w", linewidths=1.2, linestyles="--")
    th = np.linspace(0, 2 * np.pi, 200)
    if coil == "c":
        ax.plot(R_CIRC * 1e3 * np.cos(th), R_CIRC * 1e3 * np.sin(th), color="cyan", lw=1)
    else:
        for sgn in (-1, 1):
            ax.plot((sgn * S_FIG8 + R_FIG8 * np.cos(th)) * 1e3, R_FIG8 * 1e3 * np.sin(th), color="cyan", lw=1)
    ax.plot(R_TARGET[0] * 1e3, R_TARGET[1] * 1e3, "*", color="w", mec="k", ms=13, label="target")
    ax.plot(R_OFF_TARGET[0] * 1e3, R_OFF_TARGET[1] * 1e3, "o", color="lime", mec="k", ms=7, label="off-target")
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.grid(False)
    ax.set_title(f"{ttl}:  $|\\mathbf{{E}}|$ at z = $-$20 mm\n"
                 f"$E_{{max}}$ = {m['E_max_cortex_Vm']:.0f} V/m,  $S_{{1/2}}$ = {m['S_half_cm2']:.0f} cm$^2$", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.colorbar(pc, ax=ax, label=r"$|\mathbf{E}|$ [V/m]", shrink=0.9)
save(fig, "fig5_cortical_maps.png")

# ============================== Task 2.3 ==========================================
radii = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
h_list = [2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
dx_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0]
rows_a = sweep_wing_radius(radii, xs_2d, ys_2d)
rows_h = sweep_standoff(h_list, l_fig8)
rows_x = sweep_lateral_offset(dx_list)
RES["task2_3"] = {"radius": rows_a, "standoff": rows_h, "lateral": rows_x}

col = lambda rows, k: np.array([r[k] for r in rows])
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax = axes[0]
ax.plot(radii, col(rows_a, "E_target_Vm"), "o-", color=C_FIG8)
ax.set_xlabel("wing radius $a$ [mm]"); ax.set_ylabel(r"$E_{target}$ [V/m]  (6 kA)", color=C_FIG8)
ax.set_title("(a) Wing-radius sweep ($s = a$ + 1 mm, $h$ = 5 mm)")
axr = ax.twinx(); axr.spines["right"].set_visible(True); axr.grid(False)
axr.plot(radii, col(rows_a, "d_half_scalp_mm"), "s--", color="#2ca02c", label=r"$d_{1/2}$ [mm]")
axr.plot(radii, col(rows_a, "S_half_cm2") / 10.0, "^--", color="#9467bd", label=r"$S_{1/2}$ [10 cm$^2$]")
axr.set_ylabel(r"$d_{1/2}$ [mm],  $S_{1/2}$ [10 cm$^2$]"); axr.legend(loc="upper left", fontsize=8)
ax = axes[1]
ax.plot(h_list, col(rows_h, "E_target_Vm"), "o-", color=C_FIG8, label=r"$E_{target}$")
ax.plot(h_list, col(rows_h, "E_off_Vm"), "s-", color="gray", label=r"$E_{off}$")
ax.axhline(E_GOAL_TRADEOFF, color="k", ls=":", lw=1); ax.text(29.5, E_GOAL_TRADEOFF + 5, "$E_{goal}$ = 150 V/m", ha="right", fontsize=8)
ax.set_xlabel("standoff $h$ [mm]"); ax.set_ylabel(r"$|\mathbf{E}|$ [V/m]  (6 kA)")
ax.set_title("(b) Standoff sweep"); ax.legend(loc="center left", fontsize=8)
axr = ax.twinx(); axr.spines["right"].set_visible(True); axr.grid(False)
axr.plot(h_list, col(rows_h, "V_req_150_kV"), "d--", color="#ff7f0e")
axr.set_ylabel(r"$V_{req}$ for 150 V/m [kV]", color="#ff7f0e")
ax = axes[2]
ax.plot(dx_list, col(rows_x, "E_target_Vm"), "o-", color=C_FIG8)
ax.axhline(0.5 * rows_x[0]["E_target_Vm"], color="k", ls=":", lw=1)
ax.text(59, 0.5 * rows_x[0]["E_target_Vm"] + 5, "50 %", ha="right", fontsize=8)
ax.set_xlabel(r"lateral offset $\Delta x$ [mm]"); ax.set_ylabel(r"$E_{target}$ [V/m]  (6 kA)")
ax.set_title("(c) Lateral-offset sweep")
save(fig, "fig6_sweeps.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax1.plot(radii, col(rows_a, "V_req_150_kV"), "o-", color="#ff7f0e", label=r"$V_{req}$ ($E_{target}$ = 150 V/m)")
ax1.plot(radii, col(rows_a, "V_coil_kV"), "s--", color="gray", label=r"$V_{coil}$ (nominal 6 kA)")
ax1.set_xlabel("wing radius $a$ [mm]"); ax1.set_ylabel("voltage [kV]")
ax1.set_title(r"(a) Required voltage vs. wing radius"); ax1.legend(fontsize=8, loc="upper center")
axr = ax1.twinx(); axr.spines["right"].set_visible(True); axr.grid(False)
axr.plot(radii, col(rows_a, "W_m_req_150_J"), "^:", color="#8c564b")
axr.set_ylabel(r"$W_{m,req}$ [J]", color="#8c564b")
sc = ax2.scatter(col(rows_a, "S_half_cm2"), col(rows_a, "d_half_scalp_mm"), c=col(rows_a, "W_m_req_150_J"),
                 cmap="viridis", s=90, edgecolor="k", zorder=3)
ax2.plot(col(rows_a, "S_half_cm2"), col(rows_a, "d_half_scalp_mm"), "-", color="gray", lw=1, zorder=2)
for r in rows_a:
    ax2.annotate(f"a={r['radius_mm']:.0f}", (r["S_half_cm2"], r["d_half_scalp_mm"]),
                 textcoords="offset points", xytext=(7, -10), fontsize=8)
ax2.annotate("ideal:\ndeep & focal", xy=(0.04, 0.95), xycoords="axes fraction", va="top", fontsize=9, color="#2ca02c")
ax2.set_xlabel(r"half-max area $S_{1/2}$ [cm$^2$]  (smaller = more focal)")
ax2.set_ylabel(r"half-max depth $d_{1/2}$ [mm]  (larger = deeper)")
ax2.set_title("(b) Depth-focality Pareto front")
fig.colorbar(sc, ax=ax2, label=r"$W_{m,req}$ [J]")
save(fig, "fig7_tradeoff.png")

# ============================== Task 3.1 ==========================================
cases = [
    ("Figure-8 / target", fig8_pts, fig8_dl, R_TARGET),
    ("Figure-8 / off-target", fig8_pts, fig8_dl, R_OFF_TARGET),
    ("Circular / target", circ_pts, circ_dl, R_TARGET),
    ("Circular / off-target", circ_pts, circ_dl, R_OFF_TARGET),
]
RES["task3_1a"] = []
fig, axes = plt.subplots(4, 4, figsize=(14, 10.5), sharex="col")
zoom = T_MS <= 0.4
for c, (name, pts, dl, r) in enumerate(cases):
    e_eff = compute_E_eff_waveform(pts, dl, r, di_dt)
    hh = simulate_hh_neuron(T_MS, e_eff)
    k_rise = int(np.argmax(np.abs(e_eff)))
    RES["task3_1a"].append({
        "case": name, "E_eff_rise": float(e_eff[k_rise]), "I_stim_rise": float(hh["I_stim"][k_rise]),
        "Vm_end_rise": float(np.interp(0.06, T_MS, hh["V_m"])), "Vm_end_pulse": float(np.interp(0.18, T_MS, hh["V_m"])),
        "peak_V_m": hh["peak_V_m"], "min_V_m": hh["min_V_m"], "AP": hh["ap_triggered"], "latency_ms": hh["ap_latency_ms"],
        "I_Na_min": float(hh["I_Na"].min()), "I_K_max": float(hh["I_K"].max()),
    })
    ax = axes[0, c]
    ax.plot(T_MS, hh["V_m"], color="k", lw=1.5); ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_ylim(-85, 50)
    ax.set_title(f"{name}\n$E_{{eff}}$ = {e_eff[k_rise] + 0.0:+.1f} V/m,  AP = {hh['ap_triggered']}", fontsize=10,
                 color=("#b22222" if hh["ap_triggered"] else "#333"))
    ax = axes[1, c]
    for g, colr in [("m", "#d62728"), ("h", "#1f77b4"), ("n", "#2ca02c")]:
        ax.plot(T_MS, hh[g], color=colr, lw=1.4, label=f"${g}$")
    ax.set_ylim(-0.03, 1.03); ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax = axes[2, c]
    ax.plot(T_MS, hh["I_Na"], color="#d62728", lw=1.4, label=r"$I_{Na}$")
    ax.plot(T_MS, hh["I_K"], color="#2ca02c", lw=1.4, label=r"$I_K$")
    ax.set_ylim(-1500, 1000); ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax = axes[3, c]
    ax.step(T_MS, hh["I_stim"], where="post", color="#ff7f0e", lw=1.4)
    ax.set_ylim(-1400, 1400); ax.set_xlabel("t [ms]"); ax.set_xlim(0, 6)
    ins = ax.inset_axes([0.42, 0.12, 0.54, 0.34])
    ins.step(T_MS[zoom], hh["I_stim"][zoom], where="post", color="#ff7f0e", lw=1.2)
    ins.set_xlim(0, 0.3); ins.set_ylim(-1400, 1400); ins.tick_params(labelsize=6); ins.grid(False)
    ins.set_title("zoom 0-0.3 ms", fontsize=7, pad=2)
for ax, lab in zip(axes[:, 0], [r"$V_m$ [mV]", "gating variables", r"ionic current [$\mu$A/cm$^2$]",
                                r"$I_{stim}$ [$\mu$A/cm$^2$]"]):
    ax.set_ylabel(lab)
save(fig, "fig8_neuron_responses.png")

unit = di_dt / DIDT_RISE
e_th = find_electric_field_threshold(T_MS, unit)
e_f8_peak = float(np.max(np.abs(compute_E_eff_waveform(fig8_pts, fig8_dl, R_TARGET, di_dt))))
th = compute_fig8_thresholds(e_th, e_f8_peak, l_fig8)
h_max = find_max_standoff(e_th)
RES["task3_1b"] = {"E_th": e_th, "E_fig8_target": e_f8_peak, **th, "h_max_mm": h_max * 1e3,
                   "margin": e_f8_peak / e_th}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
for scale, colr in [(0.90, "#1f77b4"), (0.99, "#2ca02c"), (1.01, "#ff7f0e"), (1.10, "#d62728")]:
    hh = simulate_hh_neuron(T_MS, scale * e_th * unit)
    ax1.plot(T_MS, hh["V_m"], color=colr, lw=1.5, label=f"{scale:.2f} $E_{{th}}$ ({scale*e_th:.1f} V/m)")
ax1.axhline(0, color="gray", ls=":", lw=1)
ax1.set_xlabel("t [ms]"); ax1.set_ylabel(r"$V_m$ [mV]"); ax1.legend(fontsize=8)
ax1.set_title(f"(a) All-or-none response around $E_{{th}}$ = {e_th:.2f} V/m")
hh_arr = np.linspace(2e-3, 45e-3, 60)
e_h = []
for hv in hh_arr:
    p, d = build_figure8_coil(h_standoff=hv)
    e_h.append(float(np.max(compute_E_eff_waveform(p, d, R_TARGET, di_dt))))
ax2.plot(hh_arr * 1e3, e_h, color=C_FIG8, lw=1.8, label=r"$E_{eff}$ at target (6 kA)")
ax2.axhline(e_th, color="k", ls="--", lw=1, label=f"$E_{{th}}$ = {e_th:.1f} V/m")
ax2.axvline(h_max * 1e3, color="gray", ls=":", lw=1)
ax2.plot([h_max * 1e3], [e_th], "o", color="k")
ax2.annotate(f"$h_{{max}}$ = {h_max*1e3:.1f} mm", (h_max * 1e3, e_th), textcoords="offset points", xytext=(-90, -22), fontsize=9)
ax2.fill_between(hh_arr * 1e3, e_th, e_h, where=np.array(e_h) >= e_th, color=C_FIG8, alpha=0.12)
ax2.set_xlabel("standoff $h$ [mm]"); ax2.set_ylabel(r"$E_{eff}$ [V/m]"); ax2.legend(fontsize=8)
ax2.set_title("(b) Maximum standoff for firing")
save(fig, "fig9_threshold.png")

# ============================== 저장 ==============================================
with open(FIG_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=2, ensure_ascii=False, default=float)
print(json.dumps(RES, indent=1, default=float))
