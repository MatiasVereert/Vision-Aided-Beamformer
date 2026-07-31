"""
test_wpe_numba_equivalence.py
=============================
Valida que el core recursivo del WPE online acelerado con Numba
(``process_wpe_online`` / ``process_wpe_online_with_components`` en
``dereverberation.nara_wrappers``) sea numericamente equivalente a la
implementacion de referencia en Python puro (``_process_wpe_online_ref`` /
``_process_wpe_online_with_components_ref``), dentro de tolerancia de fp.

Chequeos:
  a) Barrido de (taps, delay) sobre senales multicanal sinteticas (M=8, varias
     longitudes) comparando NUEVO vs REFERENCIA para:
        - process_wpe_online(mezcla)
        - process_wpe_online_with_components(mezcla, [target, ruido])
     Assert max|nuevo - ref| < 1e-6 (se reporta el valor real).
  b) Reejecuta tests/test_frontend_decomposition.py para confirmar que la
     identidad de descomposicion sigue en verde con la version Numba.
  c) Mide y reporta el SPEEDUP (ref vs nuevo) sobre una senal ~15 s,
     descontando la compilacion JIT del primer llamado.

USO:
    conda activate tesis_beam
    python tests/test_wpe_numba_equivalence.py
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np

from dereverberation.nara_wrappers import (
    process_wpe_online,
    process_wpe_online_with_components,
    _process_wpe_online_ref,
    _process_wpe_online_with_components_ref,
)

FS = 16000
M = 8
TOL = 1e-6


def _synthetic_component(rng, n, M, n_taps=1200, tail_scale=0.3):
    """Fuente 1/f coloreada convolucionada con RIRs cortas por canal (direct+cola)."""
    src = rng.standard_normal(n)
    src = np.cumsum(src) - np.cumsum(np.concatenate([[0.0], src[:-1]]))
    src = src / (np.std(src) + 1e-9)
    x = np.zeros((M, n))
    for m in range(M):
        h = np.zeros(n_taps)
        h[10 + m * 3] = 1.0
        h += rng.standard_normal(n_taps) * np.exp(-np.arange(n_taps) / 300.0) * tail_scale
        x[m] = np.convolve(src, h)[:n]
    return x


def _make_mix(rng, n):
    """target + interf con escala global comun (mezcla = suma exacta)."""
    target = _synthetic_component(rng, n, M, tail_scale=0.25)
    interf = _synthetic_component(rng, n, M, tail_scale=0.40)
    mix = target + interf
    g = 0.99 / (np.max(np.abs(mix)) + 1e-10)
    return target * g, interf * g, mix * g


# ---------------------------------------------------------------------
# a) Equivalencia numerica nuevo vs referencia
# ---------------------------------------------------------------------
def test_equivalence():
    print("=" * 70)
    print("a) EQUIVALENCIA NUMERICA (nuevo Numba vs referencia Python)")
    print("=" * 70)
    rng = np.random.default_rng(20260730)

    # Varias longitudes y STFT chicas (F chico) para barrer rapido.
    durations = [1.5, 3.0]
    stft_cfg = dict(stft_size=256, stft_shift=64)
    sweeps = [
        (3, 1),
        (5, 2),
        (7, 3),
    ]
    alpha = 0.9999

    ok = True
    worst = 0.0
    for dur in durations:
        n = int(FS * dur)
        target, interf, mix = _make_mix(rng, n)
        for taps, delay in sweeps:
            # --- process_wpe_online ---
            z_new = process_wpe_online(mix, taps=taps, delay=delay, alpha=alpha, **stft_cfg)
            z_ref = _process_wpe_online_ref(mix, taps=taps, delay=delay, alpha=alpha, **stft_cfg)
            e_mix = float(np.max(np.abs(z_new - z_ref)))

            # --- process_wpe_online_with_components ---
            zu_new, (ct_new, cn_new) = process_wpe_online_with_components(
                mix, [target, interf], taps=taps, delay=delay, alpha=alpha, **stft_cfg
            )
            zu_ref, (ct_ref, cn_ref) = _process_wpe_online_with_components_ref(
                mix, [target, interf], taps=taps, delay=delay, alpha=alpha, **stft_cfg
            )
            e_zu = float(np.max(np.abs(zu_new - zu_ref)))
            e_ct = float(np.max(np.abs(ct_new - ct_ref)))
            e_cn = float(np.max(np.abs(cn_new - cn_ref)))

            # Invariantes internas de la version nueva:
            #  - z_u (con componentes) == process_wpe_online (mezcla), bit-identico
            e_zu_vs_plain = float(np.max(np.abs(zu_new - z_new)))
            #  - WPE(target)+WPE(ruido) == WPE(mezcla)  (algebraico)
            e_decomp = float(np.max(np.abs((ct_new + cn_new) - zu_new)))

            emax = max(e_mix, e_zu, e_ct, e_cn)
            worst = max(worst, emax, e_zu_vs_plain, e_decomp)
            ok &= emax < TOL
            ok &= e_zu_vs_plain < 1e-9   # misma aritmetica -> bit-identico
            ok &= e_decomp < 1e-9        # descomposicion exacta

            print(f"  dur={dur:>4}s taps={taps} delay={delay} | "
                  f"online={e_mix:.2e} zu={e_zu:.2e} tgt={e_ct:.2e} noi={e_cn:.2e} "
                  f"|| zu_vs_plain={e_zu_vs_plain:.2e} decomp={e_decomp:.2e}")

    print(f"\n  -> peor caso global = {worst:.2e}  (TOL={TOL:.0e})")
    print("  ->", "[PASS]" if ok else "[FAIL]", "equivalencia numerica")
    return ok


# ---------------------------------------------------------------------
# b) Reejecutar el test de descomposicion del front-end
# ---------------------------------------------------------------------
def test_frontend_decomposition():
    print("\n" + "=" * 70)
    print("b) test_frontend_decomposition.py (con la version Numba)")
    print("=" * 70)
    path = os.path.join(SCRIPT_DIR, "test_frontend_decomposition.py")
    res = subprocess.run([sys.executable, path], capture_output=True, text=True)
    print(res.stdout.rstrip())
    if res.stderr.strip():
        print("  [stderr]", res.stderr.rstrip())
    ok = (res.returncode == 0)
    print("  ->", "[PASS]" if ok else "[FAIL]", "front-end decomposition")
    return ok


# ---------------------------------------------------------------------
# c) Speedup (descontando compilacion JIT)
# ---------------------------------------------------------------------
def bench_speedup():
    print("\n" + "=" * 70)
    print("c) SPEEDUP (senal ~15 s, JIT ya compilado)")
    print("=" * 70)
    rng = np.random.default_rng(7)
    n = int(FS * 15.0)
    _, _, mix = _make_mix(rng, n)
    taps, delay, alpha = 7, 3, 0.9999
    cfg = dict(stft_size=512, stft_shift=128)

    # Warm-up JIT (compila ambos cores) sobre una senal corta -> no se cronometra.
    _, _, small = _make_mix(rng, int(FS * 0.5))
    t0 = time.perf_counter()
    process_wpe_online(small, taps=taps, delay=delay, alpha=alpha, **cfg)
    process_wpe_online_with_components(small, [small, small], taps=taps, delay=delay,
                                       alpha=alpha, **cfg)
    jit_compile_s = time.perf_counter() - t0
    print(f"  (compilacion JIT + warm-up: {jit_compile_s:.2f} s, no cronometrado)")

    # Referencia
    t0 = time.perf_counter()
    z_ref = _process_wpe_online_ref(mix, taps=taps, delay=delay, alpha=alpha, **cfg)
    t_ref = time.perf_counter() - t0

    # Nuevo (ya compilado)
    t0 = time.perf_counter()
    z_new = process_wpe_online(mix, taps=taps, delay=delay, alpha=alpha, **cfg)
    t_new = time.perf_counter() - t0

    e = float(np.max(np.abs(z_new - z_ref)))
    print(f"  referencia (Python loop) : {t_ref:8.3f} s")
    print(f"  nuevo (Numba)            : {t_new:8.3f} s")
    print(f"  speedup                  : {t_ref / t_new:8.2f}x")
    print(f"  max|nuevo-ref| (15 s)    : {e:.2e}")
    return e < TOL


def main():
    ok = True
    ok &= test_equivalence()
    ok &= test_frontend_decomposition()
    ok &= bench_speedup()
    print("\n" + ("[PASS] WPE Numba equivalente y mas rapido" if ok
                  else "[FAIL] revisar diferencias arriba"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
