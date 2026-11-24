"""Task 2 — 조직 레이어에서 유도 전기장(induced E-field in tissue layers): Task 2.2, Task 2.3.

Task 2.1은 보고서로만 제출하는 증명입니다. 아래 함수들은 Task 2.1에서 유도한
유도 전기장 E(r, t)의 식을 사용해 구현합니다.

Task 1 함수(task1_coil_field.py)가 필요합니다.

    python task2_tissue_induction.py

를 실행하면 Task 2.2 / 2.3 결과가 출력됩니다.
"""

from __future__ import annotations

import numpy as np

from tms_config import (
    DIDT_RISE,
    E_GOAL_TRADEOFF,
    I_PEAK_NOMINAL,
    R_CIRC,
    R_OFF_TARGET,
    R_TARGET,
    T_S,
    Z_TARGET,
    get_layer_properties,
)
from task1_coil_field import (
    build_circular_coil,
    build_figure8_coil,
    compute_inductance,
    compute_unit_A_and_B,
    trapezoidal_pulse_current_and_didt,
)


# ==============================================================================
# Task 2.2 — 깊이 분포(depth profile), 시간 파형(time waveform), 피질 분포도(cortical map)
# ==============================================================================
def compute_depth_profile(
    pts: np.ndarray,
    dl: np.ndarray,
    x_line: float,
    didt: float = DIDT_RISE,
    z_line: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """수직선 (x_line, 0, z)를 따라 유도 전기장 |E(z)|와 와전류 밀도(eddy current density) |J(z)|를 계산합니다.

    인자(Parameters)
    ----------------
    pts, dl : ndarray, shape (K, 3)
        코일 선분 중점과 현 벡터 [m].
    x_line : float
        수직선의 x 위치 [m] (보통 코일 핫스팟(hotspot) 위치).
    didt : float
        전류의 시간 미분 dI/dt [A/s] (기본값: 상승 구간 값).
    z_line : ndarray or None
        평가할 깊이 z [m]. None이면 [0, −60 mm]를 601점으로 나눕니다.

    반환값(Returns)
    ---------------
    z_line : ndarray, shape (Z,)
        깊이 좌표 [m].
    e_mag : ndarray, shape (Z,)
        유도 전기장 크기 |E(z)| [V/m].
    j_mag : ndarray, shape (Z,)
        와전류 밀도 |J(z)| = σ(z)|E(z)| [A/m²].
    """
    if z_line is None:
        z_line = np.linspace(0.0, -60e-3, 601)
    # === [Task 2.2a] ===
    # Task 2.1 결과: 평면 층 + 수평 코일에서는 grad V = 0 이므로 E = -Ã dI/dt (전도도와 무관).
    z_line = np.asarray(z_line, dtype=float)
    obs = np.column_stack([np.full_like(z_line, x_line), np.zeros_like(z_line), z_line])
    a_tilde, _ = compute_unit_A_and_B(pts, dl, obs)
    e_mag = np.linalg.norm(-a_tilde * didt, axis=1)
    sigma = np.array([get_layer_properties(float(z))[1] for z in z_line])
    j_mag = sigma * e_mag
    return z_line, e_mag, j_mag


def compute_target_waveforms(
    pts: np.ndarray,
    dl: np.ndarray,
    r_obs: np.ndarray,
    i_t: np.ndarray,
    di_dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """한 점 r_obs에서 |B(t)|와 |E(t)|의 시간 파형(time waveform)을 계산합니다.

    인자(Parameters)
    ----------------
    pts, dl : ndarray, shape (K, 3)
        코일 선분 중점과 현 벡터 [m].
    r_obs : ndarray, shape (3,)
        관측점 위치 [m].
    i_t : ndarray, shape (T,)
        코일 전류 I(t) [A].
    di_dt : ndarray, shape (T,)
        전류의 시간 미분 dI/dt [A/s].

    반환값(Returns)
    ---------------
    b_mag_t : ndarray, shape (T,)
        자기선속밀도 크기 |B(t)| = |B̃|·|I(t)| [T].
    e_mag_t : ndarray, shape (T,)
        유도 전기장 크기 |E(t)| [V/m].
    """
    # === [Task 2.2b] ===
    a_tilde, b_tilde = compute_unit_A_and_B(pts, dl, r_obs)
    b_mag_t = np.linalg.norm(b_tilde[0]) * np.abs(np.asarray(i_t, dtype=float))
    e_mag_t = np.linalg.norm(a_tilde[0]) * np.abs(np.asarray(di_dt, dtype=float))
    return b_mag_t, e_mag_t


def compute_cortical_metrics(
    pts: np.ndarray,
    dl: np.ndarray,
    didt_rise: float,
    xs_2d: np.ndarray,
    ys_2d: np.ndarray,
    x_hotspot_for_depth: float,
    l_coil: float,
    i_peak: float = I_PEAK_NOMINAL,
    e_goal: float = E_GOAL_TRADEOFF,
) -> dict:
    """z = −20 mm 피질 평면(cortical plane)의 전기장 지표와 E_goal 달성 요구 사양을 계산합니다.

    인자(Parameters)
    ----------------
    pts, dl : ndarray, shape (K, 3)
        코일 선분 중점과 현 벡터 [m].
    didt_rise : float
        상승 구간 dI/dt [A/s].
    xs_2d, ys_2d : ndarray
        2차원 분포도(2D map)의 x, y 격자(grid) [m] (등간격).
    x_hotspot_for_depth : float
        반치 깊이 d_1/2를 구할 수직선의 x 위치 [m] (원형: a_circ, 8자형: 0).
    l_coil : float
        코일 인덕턴스 L [H].
    i_peak : float
        명목 첨두 전류 [A].
    e_goal : float
        목표 전기장 E_goal [V/m].

    반환값(Returns)
    ---------------
    metrics : dict
        최소한 다음 키(key)를 포함해야 합니다.
        "E_target_Vm", "E_off_target_Vm", "E_max_cortex_Vm" : 표적점 / 비표적점 / 분포도 최대 |E| [V/m]
        "target_to_off_target_ratio"                     : 선택비(selectivity ratio) E_target / E_off
        "S_half_cm2"                                     : |E| ≥ 0.5·E_max인 반치 면적(half-max area) [cm²]
        "d_half_from_scalp_mm"                           : |E|가 두피 값의 절반이 되는 반치 깊이(half-max depth) [mm]
        "L_uH", "V_coil_kV", "W_m_J"                     : 명목 L, V_coil = L·didt_rise, W_m = ½·L·i_peak²
        "I_req_150_kA", "V_req_150_kV", "W_m_req_150_J"  : 기준 전기장 E_ref를 e_goal로 맞출 때의 요구 사양
        "emag_2d"                                        : |E| 분포도, shape (len(xs_2d), len(ys_2d)) [V/m]
        E_ref = E_target이며, E_target ≈ 0이면(원형 코일) E_max(핫스팟)를 사용합니다.
    """
    # === [Task 2.2c] ===
    # 1. 피질 평면 |E| 분포도 (emag_2d[i, j] <-> (xs_2d[i], ys_2d[j]))
    xx, yy = np.meshgrid(xs_2d, ys_2d, indexing="ij")
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, Z_TARGET)])
    a_map, _ = compute_unit_A_and_B(pts, dl, obs)
    emag_2d = (np.linalg.norm(a_map, axis=1) * abs(didt_rise)).reshape(xx.shape)

    # 2. 표적 / 비표적 / 최대 전기장
    a_pts, _ = compute_unit_A_and_B(pts, dl, np.vstack([R_TARGET, R_OFF_TARGET]))
    e_target = float(np.linalg.norm(a_pts[0]) * abs(didt_rise))
    e_off = float(np.linalg.norm(a_pts[1]) * abs(didt_rise))
    e_max = float(np.max(emag_2d))

    # 3. 반치 면적 S_1/2 = (|E| >= 0.5 E_max 인 격자 칸 수) · dx · dy
    dx = float(xs_2d[1] - xs_2d[0])
    dy = float(ys_2d[1] - ys_2d[0])
    s_half_m2 = float(np.count_nonzero(emag_2d >= 0.5 * e_max)) * dx * dy

    # 4. 반치 깊이 d_1/2: 핫스팟 수직선에서 |E|가 두피(z = 0) 값의 절반이 되는 깊이
    z_fine = np.linspace(0.0, -200e-3, 4001)
    _, e_line, _ = compute_depth_profile(pts, dl, x_hotspot_for_depth, didt=didt_rise, z_line=z_fine)
    half = 0.5 * e_line[0]
    below = np.nonzero(e_line <= half)[0]
    if below.size == 0 or below[0] == 0:
        d_half_m = float("nan")
    else:
        k = int(below[0])
        frac = (e_line[k - 1] - half) / (e_line[k - 1] - e_line[k])   # 선형 보간
        d_half_m = float(-(z_fine[k - 1] + frac * (z_fine[k] - z_fine[k - 1])))

    # 5. E_goal 달성 요구 사양 (E ∝ I_peak ∝ dI/dt, T_rise 고정)
    e_ref = e_target if e_target > 1e-3 * e_max else e_max
    scale = e_goal / e_ref
    i_req = i_peak * scale
    v_coil = l_coil * abs(didt_rise)
    v_req = v_coil * scale
    w_m = 0.5 * l_coil * i_peak ** 2
    w_req = 0.5 * l_coil * i_req ** 2

    return {
        "E_target_Vm": e_target,
        "E_off_target_Vm": e_off,
        "E_max_cortex_Vm": e_max,
        "E_ref_Vm": e_ref,
        "target_to_off_target_ratio": e_target / e_off if e_off > 0.0 else float("inf"),
        "S_half_cm2": s_half_m2 * 1e4,
        "d_half_from_scalp_mm": d_half_m * 1e3,
        "L_uH": l_coil * 1e6,
        "V_coil_kV": v_coil * 1e-3,
        "W_m_J": w_m,
        "I_req_150_kA": i_req * 1e-3,
        "V_req_150_kV": v_req * 1e-3,
        "W_m_req_150_J": w_req,
        "emag_2d": emag_2d,
    }


# ==============================================================================
# Task 2.3 — 8자형 코일 설계 변수 스윕(parametric sweep)
# ==============================================================================
def sweep_wing_radius(
    radii_mm: list[float],
    xs_2d: np.ndarray,
    ys_2d: np.ndarray,
    didt_rise: float = DIDT_RISE,
) -> list[dict]:
    """날개 반경(wing radius) a를 바꾸어 가며 8자형 코일의 지표를 계산합니다 (s = a + 1 mm, h = 5 mm).

    인자(Parameters)
    ----------------
    radii_mm : list of float
        날개 반경 목록 [mm].
    xs_2d, ys_2d : ndarray
        compute_cortical_metrics에 넘길 피질 격자 [m].
    didt_rise : float
        상승 구간 dI/dt [A/s].

    반환값(Returns)
    ---------------
    rows : list of dict
        반경마다 dict 하나. 키: "radius_mm", "E_target_Vm", "d_half_scalp_mm",
        "S_half_cm2", "L_uH", "V_coil_kV", "V_req_150_kV", "W_m_req_150_J".
    """
    # === [Task 2.3a] ===
    rows = []
    for radius_mm in radii_mm:
        a = float(radius_mm) * 1e-3
        pts, dl = build_figure8_coil(wing_radius=a)        # s = a + 1 mm, h = 5 mm (기본값)
        l_coil = compute_inductance(pts, dl)
        m = compute_cortical_metrics(pts, dl, didt_rise, xs_2d, ys_2d, 0.0, l_coil)
        rows.append({
            "radius_mm": float(radius_mm),
            "E_target_Vm": m["E_target_Vm"],
            "d_half_scalp_mm": m["d_half_from_scalp_mm"],
            "S_half_cm2": m["S_half_cm2"],
            "L_uH": m["L_uH"],
            "V_coil_kV": m["V_coil_kV"],
            "V_req_150_kV": m["V_req_150_kV"],
            "W_m_req_150_J": m["W_m_req_150_J"],
        })
    return rows


def sweep_standoff(
    h_list_mm: list[float],
    l_fig8: float,
    didt_rise: float = DIDT_RISE,
) -> list[dict]:
    """코일–두피 이격 거리(standoff distance) h를 바꾸어 가며 명목 8자형 코일의 지표를 계산합니다.

    인자(Parameters)
    ----------------
    h_list_mm : list of float
        이격 거리 목록 [mm].
    l_fig8 : float
        8자형 코일 인덕턴스 [H] (h와 무관하므로 그대로 사용).
    didt_rise : float
        상승 구간 dI/dt [A/s].

    반환값(Returns)
    ---------------
    rows : list of dict
        h마다 dict 하나. 키: "h_mm", "E_target_Vm", "E_off_Vm",
        "V_req_150_kV", "W_m_req_150_J" (E_target = E_GOAL_TRADEOFF를 위한 요구 사양).
    """
    # === [Task 2.3b] ===
    rows = []
    for h_mm in h_list_mm:
        pts, dl = build_figure8_coil(h_standoff=float(h_mm) * 1e-3)
        a_pts, _ = compute_unit_A_and_B(pts, dl, np.vstack([R_TARGET, R_OFF_TARGET]))
        e_target = float(np.linalg.norm(a_pts[0]) * abs(didt_rise))
        e_off = float(np.linalg.norm(a_pts[1]) * abs(didt_rise))
        scale = E_GOAL_TRADEOFF / e_target
        i_req = I_PEAK_NOMINAL * scale
        rows.append({
            "h_mm": float(h_mm),
            "E_target_Vm": e_target,
            "E_off_Vm": e_off,
            "V_req_150_kV": l_fig8 * abs(didt_rise) * scale * 1e-3,
            "W_m_req_150_J": 0.5 * l_fig8 * i_req ** 2,
        })
    return rows


def sweep_lateral_offset(
    dx_list_mm: list[float],
    didt_rise: float = DIDT_RISE,
) -> list[dict]:
    """코일을 x 방향으로 Δx만큼 옮겼을 때(수평 정렬 오차, lateral offset) 고정된 표적점의 |E|를 계산합니다.

    인자(Parameters)
    ----------------
    dx_list_mm : list of float
        수평 이동량 Δx 목록 [mm] (build_figure8_coil의 offset_x로 사용).
    didt_rise : float
        상승 구간 dI/dt [A/s].

    반환값(Returns)
    ---------------
    rows : list of dict
        Δx마다 dict 하나. 키: "dx_mm", "E_target_Vm".
    """
    # === [Task 2.3c] ===
    rows = []
    for dx_mm in dx_list_mm:
        pts, dl = build_figure8_coil(offset_x=float(dx_mm) * 1e-3)
        a_pt, _ = compute_unit_A_and_B(pts, dl, R_TARGET)
        rows.append({
            "dx_mm": float(dx_mm),
            "E_target_Vm": float(np.linalg.norm(a_pt[0]) * abs(didt_rise)),
        })
    return rows


# ==============================================================================
# Task 2 검증 실행(이 파일을 직접 실행할 때)
# ==============================================================================
def main() -> None:
    circ_pts, circ_dl = build_circular_coil()
    fig8_pts, fig8_dl = build_figure8_coil()
    l_circ = compute_inductance(circ_pts, circ_dl)
    l_fig8 = compute_inductance(fig8_pts, fig8_dl)
    i_t, di_dt = trapezoidal_pulse_current_and_didt(T_S)

    # --- Task 2.2a: 핫스팟 아래 깊이 분포 ---
    _, e_f8, j_f8 = compute_depth_profile(fig8_pts, fig8_dl, x_line=0.0)
    _, e_cc, j_cc = compute_depth_profile(circ_pts, circ_dl, x_line=R_CIRC)
    print(f"[Task 2.2a] Peak |J| (CSF shunting): Figure-8 = {np.max(j_f8):.2f} A/m^2, Circular = {np.max(j_cc):.2f} A/m^2")

    # --- Task 2.2b: 표적점의 |B(t)| vs |E(t)| ---
    b_t, e_t = compute_target_waveforms(fig8_pts, fig8_dl, R_TARGET, i_t, di_dt)
    hold = (T_S >= 0.07e-3) & (T_S < 0.11e-3)
    print(f"[Task 2.2b] Figure-8 target on hold: |B| = {np.max(b_t[hold]):.3f} T, |E| = {np.max(e_t[hold]):.2f} V/m")

    # --- Task 2.2c: 피질 분포도 지표 ---
    xs_2d = np.linspace(-120e-3, 120e-3, 121)
    ys_2d = np.linspace(-120e-3, 120e-3, 121)
    m_c = compute_cortical_metrics(circ_pts, circ_dl, DIDT_RISE, xs_2d, ys_2d, R_CIRC, l_circ)
    m_f = compute_cortical_metrics(fig8_pts, fig8_dl, DIDT_RISE, xs_2d, ys_2d, 0.0, l_fig8)
    for name, m in [("Circular", m_c), ("Figure-8", m_f)]:
        print(f"[Task 2.2c] {name:8s} | E_target = {m['E_target_Vm']:.2e} V/m, E_off = {m['E_off_target_Vm']:.2f} V/m, "
              f"S_1/2 = {m['S_half_cm2']:.2f} cm^2, d_1/2 = {m['d_half_from_scalp_mm']:.2f} mm, "
              f"V_req = {m['V_req_150_kV']:.2f} kV, Wm_req = {m['W_m_req_150_J']:.1f} J")

    # --- Task 2.3: 설계 변수 스윕 ---
    for row in sweep_wing_radius([25.0, 30.0, 35.0, 45.0, 55.0], xs_2d, ys_2d):
        print(f"[Task 2.3a] a = {row['radius_mm']:4.0f} mm | d_1/2 = {row['d_half_scalp_mm']:.2f} mm, "
              f"S_1/2 = {row['S_half_cm2']:.2f} cm^2, V_req = {row['V_req_150_kV']:.2f} kV, Wm_req = {row['W_m_req_150_J']:.1f} J")
    for row in sweep_standoff([2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0], l_fig8):
        print(f"[Task 2.3b] h = {row['h_mm']:4.0f} mm | E_target = {row['E_target_Vm']:.2f} V/m, V_req = {row['V_req_150_kV']:.2f} kV")
    for row in sweep_lateral_offset([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]):
        print(f"[Task 2.3c] dx = {row['dx_mm']:4.0f} mm | E_target = {row['E_target_Vm']:.2f} V/m")


if __name__ == "__main__":
    main()
