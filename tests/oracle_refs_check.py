"""
oracle_refs_check.py
====================
Prueba BEFORE/AFTER del fix de dominio de las referencias del ORACLE.

El fix cambia SOLO como se construyen scene_config['oracle_target'/'oracle_noise']
(ahora en el MISMO dominio que mic_signals_ready: HW mismatch + WPE). Por lo tanto:

  * Procesadores NO-oracle (DS, NM-MVDR, ...) NO leen esas refs -> su salida y sus
    metricas deben quedar IDENTICAS before/after (no-regresion del pipeline).
  * Procesadores ORACLE (oracle-mask, Souden-SCM) cambian: pasan a ser una cota
    superior CONSISTENTE -> deben quedar >= NM-MVDR en todas las celdas.

USO:
    conda activate tesis_beam
    # 1) baseline con el codigo ACTUAL (pre-fix):
    python tests/oracle_refs_check.py run  <before_dir>
    # 2) aplicar el fix, luego:
    python tests/oracle_refs_check.py run  <after_dir>
    # 3) comparar:
    python tests/oracle_refs_check.py compare <before_dir> <after_dir>

Grid chico (rapido): rt60=360ms, use_wpe {on,off} x mismatch_gain {0, 0.6},
1 interferente. spacing 8-8-8 (existe para 160/360/610).
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

SIGNALS = os.path.join(REPO_ROOT, "tools", "data", "signals")
ROOT_MIRD_DIR = os.path.join(REPO_ROOT, "tools", "data", "rirs", "mird")

# Witnesses de NO-regresion (no leen oracle refs) + baseline de comparacion.
NON_ORACLE = ["DS", "NM-MVDR"]
ORACLE = ["Oracle_mask_soft", "Oracle_mask_hard", "Souden_Oracle_SCM"]
# Celda -> claves que la identifican (para el merge before/after).
CELL_KEYS = ["rt60", "use_wpe", "mismatch_gain", "mismatch_phase", "isir_db",
             "N_interferences"]
CHECK_METRICS = ["Delta_tot_PESQ_early", "Delta_tot_STOI_early",
                 "Delta_tot_SDR_early", "Delta_tot_SIR_early"]


def _build():
    base_config = {
        'fs': 16000,
        'duration': 6,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "8-8-8-8-8-8-8",
        'snr_db': 60.0,
        'source_path': os.path.join(SIGNALS, "p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(SIGNALS, "techno_gated commune.wav")],
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['early'],
        'dtln_model_path': os.path.join(REPO_ROOT, "src", "dnn_denoise", "models", "model_quant_1.tflite"),
    }
    param_grid = {
        'rt60': [0.360],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]],
        'isir_db': [3],
        'mismatch_gain': [0.0, 0.6],
        'mismatch_phase': [0.0],
        'use_wpe': [True, False],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0],
    }
    return base_config, param_grid


def _build_verify():
    """Grid amplio pedido para P1 (WPE on/off) y P2 (mismatch): RT60 {160,360,610}
    x use_wpe {T,F} x {1,2 interferentes} x mismatch_gain {0, 0.6}."""
    base_config, _ = _build()
    param_grid = {
        'rt60': [0.160, 0.360, 0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)], [(45, 1.0), (315, 1.0)]],
        'isir_db': [3],
        'mismatch_gain': [0.0, 0.6],
        'mismatch_phase': [0.0],
        'use_wpe': [True, False],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0],
    }
    return base_config, param_grid


def verify(outdir):
    """Corre el grid amplio (solo codigo ARREGLADO) y chequea oracle>=NM-MVDR en
    TODAS las celdas, desglosado por use_wpe y por mismatch."""
    from propagation.mird_loader import MirdDatasetProvider
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
    from evaluation.bf_wrappers import NM_MVDR, ORACLE_MB_MVDR_SOUDEN, SOUDEN_ORACLE_SCM

    os.makedirs(outdir, exist_ok=True)
    provider = MirdDatasetProvider(root_dir=os.path.abspath(ROOT_MIRD_DIR))
    base_config, param_grid = _build_verify()
    processors = {
        "NM-MVDR": NM_MVDR(min_loading=1e-6, alpha=0.99, sharpen_exp=4.0),
        "Oracle_mask_soft": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=1.0),
        "Oracle_mask_hard": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=4.0),
        "Souden_Oracle_SCM": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99),
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=processors,
        scene_base_config=base_config, output_dir=outdir,
        interpreter_1=None, interpreter_2=None,
    )
    df.to_parquet(os.path.join(outdir, "verify_metrics.parquet"))
    return _report_upper_bound(df)


def _report_upper_bound(df, tol=-0.02):
    """oracle >= NM-MVDR por celda, desglose de margenes; devuelve 0 si PASS."""
    nm = df[df["processor"] == "NM-MVDR"].set_index(CELL_KEYS)
    assert nm.index.is_unique, "CELL_KEYS no identifican celdas unicas para NM-MVDR"
    print("\n" + "=" * 78)
    print("VERIFY (grid amplio) — oracle >= NM-MVDR por celda  [tol=-0.02]")
    print("=" * 78)
    all_ok = True
    for proc in ORACLE:
        po = df[df["processor"] == proc].set_index(CELL_KEYS)
        viol = []
        margins = {c: [] for c in CHECK_METRICS}
        for cell, row in po.iterrows():
            if cell not in nm.index:
                continue
            for c in CHECK_METRICS:
                dv = float(row[c]) - float(nm.loc[cell, c])
                if np.isfinite(dv):
                    margins[c].append(dv)
                    if dv < tol:
                        viol.append((cell, c, dv))
        n_cells = len(po)
        if viol:
            all_ok = False
            print(f"  {proc:20s} [FAIL] {len(viol)} viol / {n_cells} celdas:")
            for cell, c, dv in viol[:12]:
                print(f"      {dict(zip(CELL_KEYS, cell))} {c} Δ={dv:+.3f}")
        else:
            print(f"  {proc:20s} [OK] {n_cells} celdas, min margen sobre NM-MVDR:")
            for c in CHECK_METRICS:
                if margins[c]:
                    print(f"        {c:22s} min Δ = {min(margins[c]):+.3f}")
    print("=" * 78)
    print(f"VERIFY VEREDICTO: {'[PASS]' if all_ok else '[FAIL]'}")
    print("=" * 78)
    return 0 if all_ok else 1


def run(outdir):
    from propagation.mird_loader import MirdDatasetProvider
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
    from evaluation.bf_wrappers import DS, NM_MVDR, ORACLE_MB_MVDR_SOUDEN, SOUDEN_ORACLE_SCM

    os.makedirs(outdir, exist_ok=True)
    provider = MirdDatasetProvider(root_dir=os.path.abspath(ROOT_MIRD_DIR))
    base_config, param_grid = _build()

    processors = {
        "DS": DS(),
        "NM-MVDR": NM_MVDR(min_loading=1e-6, alpha=0.99, sharpen_exp=4.0),
        "Oracle_mask_soft": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=1.0),
        "Oracle_mask_hard": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=4.0),
        "Souden_Oracle_SCM": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99),
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=processors,
        scene_base_config=base_config, output_dir=outdir,
        interpreter_1=None, interpreter_2=None,
    )
    print(f"\n[*] parquet -> {os.path.join(outdir, 'mird_benchmark_metrics.parquet')}")
    return df


def _load(d):
    return pd.read_parquet(os.path.join(d, "mird_benchmark_metrics.parquet"))


def compare(before_dir, after_dir):
    b = _load(before_dir)
    a = _load(after_dir)
    keys = CELL_KEYS + ["processor"]
    metric_cols = [c for c in a.columns if c.startswith(("Delta_tot_", "proc_"))
                   and (c.endswith("_early"))]

    m = pd.merge(b[keys + metric_cols], a[keys + metric_cols],
                 on=keys, suffixes=("_before", "_after"))

    print("=" * 70)
    print("1) NO-REGRESION: procesadores NO-oracle deben ser IDENTICOS")
    print("=" * 70)
    reg_ok = True
    for proc in NON_ORACLE:
        sub = m[m["processor"] == proc]
        if sub.empty:
            print(f"  [!] {proc}: sin filas"); continue
        max_abs = 0.0
        for c in metric_cols:
            diff = (sub[f"{c}_before"] - sub[f"{c}_after"]).abs()
            diff = diff[np.isfinite(diff)]
            if len(diff):
                max_abs = max(max_abs, float(diff.max()))
        status = "OK" if max_abs < 1e-9 else "CAMBIO!"
        reg_ok &= max_abs < 1e-9
        print(f"  {proc:20s} max|Δ metrica| = {max_abs:.2e}   [{status}]")

    print("\n" + "=" * 70)
    print("2) ORACLE cambio (esperado) — resumen de Delta_tot_*_early")
    print("=" * 70)
    for proc in ORACLE:
        sub = m[m["processor"] == proc]
        if sub.empty:
            continue
        moved = 0.0
        for c in CHECK_METRICS:
            if f"{c}_before" in sub:
                d = (sub[f"{c}_before"] - sub[f"{c}_after"]).abs()
                d = d[np.isfinite(d)]
                if len(d):
                    moved = max(moved, float(d.max()))
        print(f"  {proc:20s} max|Δ before->after| = {moved:.3f}")

    print("\n" + "=" * 70)
    print("3) COTA SUPERIOR: oracle >= NM-MVDR por celda (AFTER)  [tol=-0.02]")
    print("=" * 70)
    tol = -0.02
    nm = a[a["processor"] == "NM-MVDR"].set_index(CELL_KEYS)
    ub_ok = True
    for proc in ORACLE:
        po = a[a["processor"] == proc].set_index(CELL_KEYS)
        viol = []
        for cell, row in po.iterrows():
            if cell not in nm.index:
                continue
            nmrow = nm.loc[cell]
            for c in CHECK_METRICS:
                dv = row.get(c, np.nan) - nmrow.get(c, np.nan)
                if np.isfinite(dv) and dv < tol:
                    viol.append((cell, c, dv))
        # tambien reportar el mismo check en BEFORE (para ver el bug original)
        pob = b[b["processor"] == proc].set_index(CELL_KEYS)
        nmb = b[b["processor"] == "NM-MVDR"].set_index(CELL_KEYS)
        viol_b = 0
        for cell, row in pob.iterrows():
            if cell not in nmb.index:
                continue
            for c in CHECK_METRICS:
                dv = row.get(c, np.nan) - nmb.loc[cell].get(c, np.nan)
                if np.isfinite(dv) and dv < tol:
                    viol_b += 1
        if viol:
            ub_ok = False
            print(f"  {proc:20s} [FAIL] {len(viol)} violaciones AFTER "
                  f"(before tenia {viol_b}):")
            for cell, c, dv in viol[:8]:
                print(f"        cell={dict(zip(CELL_KEYS, cell))}  {c}  Δ={dv:+.3f}")
        else:
            print(f"  {proc:20s} [OK] oracle>=NM-MVDR en todas las celdas "
                  f"(before violaba {viol_b})")

    print("\n" + "=" * 70)
    verdict = reg_ok and ub_ok
    print(f"VEREDICTO: {'[PASS]' if verdict else '[FAIL]'}  "
          f"(no-regresion={'ok' if reg_ok else 'FALLA'}, "
          f"cota-superior={'ok' if ub_ok else 'FALLA'})")
    print("=" * 70)
    return 0 if verdict else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "verify":
        sys.exit(verify(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "recheck":
        sys.exit(_report_upper_bound(
            pd.read_parquet(os.path.join(sys.argv[2], "verify_metrics.parquet"))))
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
