"""
Re-corre SOLO NM_MVDR_OFB sobre el barrido de iSIR ya existente (los otros
procesadores no cambiaron de codigo; no hace falta recalcularlos). Actualiza
IN-PLACE las filas 'NM_MVDR_OFB' de isir_sweep_metrics.csv y
nonintrusive_sweep_metrics.csv (el resto de las filas queda intacto) y
regenera los 4 PNG del barrido.

Uso
---
    python tests/dsm_blind_rerun_ofb.py
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from evaluation.intrusive_benchmark_real import (
    run_intrusive_benchmark, default_base_config, DTLN_MODEL_1, DTLN_MODEL_2,
)
from evaluation.bf_wrappers import NM_MVDR_OFB

from dsm_blind_isir_sweep import (
    DEFAULT_SENAL, DEFAULT_RUIDO, DEFAULT_OUT, METRIC_COLS as INTR_METRIC_COLS,
    plot_global_bars as plot_global_bars_intr, plot_isir_curves as plot_isir_curves_intr,
)
from dsm_blind_isir_nonintrusive_plots import (
    METRIC_COLS as NONI_METRIC_COLS, plot_global_bars as plot_global_bars_noni,
    plot_isir_curves as plot_isir_curves_noni,
)

PROC_NAME = "NM_MVDR_OFB"
ISIRS = [-10, -5, 0, 5, 10]


def main():
    interp1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1); interp1.allocate_tensors()
    interp2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2); interp2.allocate_tensors()

    proc = NM_MVDR_OFB(win_type='rect', synth='hann', sharpen_exp=8.0, alpha=0.99,
                       block_update=1, leak=0.0, smooth=0.5, fuse='mean', fuse_src='ref')

    intr_csv = os.path.join(DEFAULT_OUT, "isir_sweep_metrics.csv")
    noni_csv = os.path.join(DEFAULT_OUT, "nonintrusive_sweep_metrics.csv")
    df_intr = pd.read_csv(intr_csv)
    df_noni = pd.read_csv(noni_csv)
    df_intr = df_intr[df_intr["processor"] != PROC_NAME]
    df_noni = df_noni[df_noni["processor"] != PROC_NAME]

    new_intr_rows, new_noni_rows = [], []
    for isir in ISIRS:
        cell_out = os.path.join(DEFAULT_OUT, f"isir{isir:g}")
        print(f"\n[*] === iSIR {isir:+g} dB -> {cell_out}  (solo {PROC_NAME}) ===")
        base_config = default_base_config(fs=16000)
        res = run_intrusive_benchmark(
            senal_path=DEFAULT_SENAL, ruido_path=DEFAULT_RUIDO, output_dir=cell_out,
            base_config=base_config, interpreter_1=interp1, interpreter_2=interp2,
            snr=isir, ref_mic=None, eval_start_s=5.0, use_wpe=False,
            extra_processors={PROC_NAME: proc},
        )
        metrics = dict(res["rows"])[PROC_NAME]
        row = {"isir_db": isir, "processor": PROC_NAME}
        for c in INTR_METRIC_COLS + ["SAR"]:
            row[c] = metrics.get(c, np.nan)
        new_intr_rows.append(row)

        diag = pd.read_csv(os.path.join(cell_out, "diagnostics_real.csv"))
        diag_row = diag[diag["senal"] == PROC_NAME].iloc[0].to_dict()
        nrow = {"isir_db": isir, "processor": PROC_NAME}
        for c in NONI_METRIC_COLS:
            nrow[c] = diag_row.get(c, np.nan)
        new_noni_rows.append(nrow)

    df_intr = pd.concat([df_intr, pd.DataFrame(new_intr_rows)], ignore_index=True)
    df_noni = pd.concat([df_noni, pd.DataFrame(new_noni_rows)], ignore_index=True)
    df_intr.to_csv(intr_csv, index=False)
    df_noni.to_csv(noni_csv, index=False)
    print(f"\n[*] CSVs actualizados: {intr_csv}  {noni_csv}")

    plot_global_bars_intr(df_intr, os.path.join(DEFAULT_OUT, "global_bars.png"))
    plot_isir_curves_intr(df_intr, os.path.join(DEFAULT_OUT, "isir_curves.png"))
    plot_global_bars_noni(df_noni, os.path.join(DEFAULT_OUT, "global_bars_nonintrusive.png"))
    plot_isir_curves_noni(df_noni, os.path.join(DEFAULT_OUT, "isir_curves_nonintrusive.png"))
    print(f"\n[ok] {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
