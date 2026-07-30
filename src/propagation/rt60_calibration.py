"""
Single-channel RT60 self-consistency calibration.

Makes the simulator's *realised* RT60 (measured by Schroeder integration on the
produced RIR) equal the requested RT60, instead of trusting Sabine's open-loop
inversion (pra.inverse_sabine), which is biased for non-small absorption.
RT60 is essentially space-invariant, so a single source/mic pair in the target
room is enough and cheap.

Provides:
  - simulate_singlechannel_rir : forward model (one omni src, one omni mic)
  - maxorder_convergence       : realised RT60 vs ISM order (pick where it saturates)
  - sweep_alpha                : realised RT60 vs broadband absorption alpha
  - build_alpha_to_rt60        : monotone map + inverse (alpha_of_rt60 / rt60_of_alpha)
"""
import numpy as np
import pyroomacoustics as pra
from scipy.interpolate import interp1d

from propagation.acoustic_descriptors import measure_rt60_schroeder

C_SOUND = 343.0


def _default_geometry(room_dims, src_dist=1.0, height=None):
    """
    Representative single source/mic pair. RT60 is space-invariant, so we only
    avoid the exact centre plane (degenerate symmetric image overlaps / flutter
    that would bias the EDC).
    """
    room_dims = np.asarray(room_dims, float)
    if height is None:
        height = room_dims[2] / 2.0
    mic = np.array([room_dims[0] * 0.5 - 0.1, room_dims[1] * 0.5 + 0.1, height])
    src = mic + np.array([0.0, src_dist, 0.0])
    return src, mic


def simulate_singlechannel_rir(room_dims, alpha, fs, max_order,
                               src=None, mic=None, air_absorption=True,
                               use_rand_ism=False, max_rand_disp=0.08):
    """
    Forward model for one omni source and one omni mic in a shoebox with uniform
    broadband absorption `alpha`. use_rand_ism defaults to False here so the
    alpha->RT60 map is deterministic (RT60 is a gross statistic and barely moves
    with rand_ism anyway; randomisation matters later for spatial coherence).
    """
    room_dims = np.asarray(room_dims, float)
    if src is None or mic is None:
        src, mic = _default_geometry(room_dims)

    room = pra.ShoeBox(
        room_dims, fs=fs,
        materials=pra.Material(float(alpha)),
        max_order=int(max_order),
        air_absorption=air_absorption,
        use_rand_ism=use_rand_ism,
        max_rand_disp=max_rand_disp,
    )
    room.add_source(np.asarray(src, float))
    room.add_microphone(np.asarray(mic, float))
    room.compute_rir()
    return np.asarray(room.rir[0][0], float)


def maxorder_convergence(room_dims, alpha, fs, orders, method='T20', **sim_kwargs):
    """
    Realised RT60 as ISM order grows, for a fixed absorption. The order at which
    RT60 stops increasing is the truncation-safe minimum; below it the tail is
    cut and RT60 is underestimated. Returns array of (order, rt60).
    """
    rows = []
    for order in orders:
        rir = simulate_singlechannel_rir(room_dims, alpha, fs, order, **sim_kwargs)
        rt60 = measure_rt60_schroeder(rir, fs, method=method)
        rows.append((int(order), float(rt60)))
    return np.array(rows)


def sweep_alpha(room_dims, fs, alphas, max_order, method='T20', **sim_kwargs):
    """
    Realised RT60 vs broadband absorption alpha. Returns array of (alpha, rt60).
    """
    rows = []
    for a in alphas:
        rir = simulate_singlechannel_rir(room_dims, a, fs, max_order, **sim_kwargs)
        rt60 = measure_rt60_schroeder(rir, fs, method=method)
        rows.append((float(a), float(rt60)))
    return np.array(rows)


def build_alpha_to_rt60(sweep):
    """
    Monotone interpolants from a sweep array (alpha, rt60):
        rt60_of_alpha(alpha), alpha_of_rt60(rt60).
    NaN rows (failed fits) are dropped. RT60 decreases with alpha, so both maps
    are well defined after sorting.
    """
    sweep = np.asarray(sweep, float)
    sweep = sweep[~np.isnan(sweep[:, 1])]
    a, rt = sweep[:, 0], sweep[:, 1]

    oa = np.argsort(a)
    rt60_of_alpha = interp1d(a[oa], rt[oa], bounds_error=False, fill_value='extrapolate')
    ort = np.argsort(rt)
    alpha_of_rt60 = interp1d(rt[ort], a[ort], bounds_error=False, fill_value='extrapolate')
    return rt60_of_alpha, alpha_of_rt60


if __name__ == "__main__":
    # Calibrate on the MIRD room (Hadad et al. 2014): 6 x 6 x 2.4 m.
    # MIRD measured RT60s: 0.16, 0.36, 0.61 s. We check what the OPEN-LOOP Sabine
    # inversion actually realises vs what a closed-loop alpha delivers.
    room_dims = np.array([6.0, 6.0, 2.4])
    fs = 16000
    method = 'T20'
    mird_rts = [0.16, 0.36, 0.61]

    print("=" * 68)
    print(f"RT60 self-consistency calibration | room={room_dims} | fs={fs}")
    print("=" * 68)

    # --- 1) Order convergence at a mid absorption (pick truncation-safe order) ---
    print("\n[1] max_order convergence (alpha=0.3):")
    conv = maxorder_convergence(room_dims, 0.3, fs, orders=[10, 20, 30, 40, 50, 70])
    print(f"    {'order':>6} {'RT60[s]':>9}")
    for o, rt in conv:
        print(f"    {int(o):6d} {rt:9.3f}")

    MAX_ORDER = 50  # from convergence: RT60 should be saturated by here

    # --- 2) Open-loop Sabine bias: what does inverse_sabine actually realise? ---
    print(f"\n[2] Open-loop Sabine vs realised (max_order={MAX_ORDER}):")
    print(f"    {'RT_target':>9} {'alpha_Sabine':>13} {'order_Sabine':>13} {'RT_realised':>12}")
    for rt_t in mird_rts:
        a_sab, mo_sab = pra.inverse_sabine(rt_t, room_dims)
        rir = simulate_singlechannel_rir(room_dims, a_sab, fs, MAX_ORDER)
        rt_real = measure_rt60_schroeder(rir, fs, method=method)
        print(f"    {rt_t:9.3f} {a_sab:13.4f} {mo_sab:13d} {rt_real:12.3f}")

    # --- 3) Closed-loop: sweep alpha -> realised RT60, invert ---
    alphas = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90])
    print(f"\n[3] alpha sweep (max_order={MAX_ORDER}):")
    sweep = sweep_alpha(room_dims, fs, alphas, MAX_ORDER, method=method)
    print(f"    {'alpha':>6} {'RT60[s]':>9}")
    for a, rt in sweep:
        print(f"    {a:6.2f} {rt:9.3f}")

    rt60_of_alpha, alpha_of_rt60 = build_alpha_to_rt60(sweep)

    print("\n[4] Closed-loop alpha for each MIRD RT60 (calibrated inverse):")
    print(f"    {'RT_target':>9} {'alpha_cal':>10} {'RT_check':>9}")
    for rt_t in mird_rts:
        a_cal = float(alpha_of_rt60(rt_t))
        rir = simulate_singlechannel_rir(room_dims, a_cal, fs, MAX_ORDER)
        rt_chk = measure_rt60_schroeder(rir, fs, method=method)
        print(f"    {rt_t:9.3f} {a_cal:10.4f} {rt_chk:9.3f}")
