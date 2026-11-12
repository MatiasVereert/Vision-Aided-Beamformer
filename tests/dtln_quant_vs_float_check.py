"""
¿CUANTO PESA HABER COMPARADO CONTRA EL DTLN CUANTIZADO?

La Fase 2 (Fase_2_seleccion_algoritmo.ipynb) corre el baseline mono y el sistema
con `model_quant_{1,2}.tflite`. En hardware los modelos SIN cuantizar
(tools/data/models/model_{1,2}.tflite) suenan bastante mejor, sobre todo la
etapa 2, que cuantizada mete un ruido molesto (probable discontinuidad de fase
entre bloques). Este test acotado mide si eso cambia la conclusion de la fase
antes de tocar el notebook.

OJO CON LA ASIMETRIA: el OFB usa SOLO la etapa 1 del DTLN (la mascara). La
etapa 2 la usa unicamente el mono. Si el artefacto es de la etapa 2, la
cuantizacion le pego al baseline y no al sistema, y la brecha de la Fase 2
esta inflada a favor del NM-MVDR.

FILAS (todas sobre las MISMAS escenas, una sola corrida del motor)
-----
    NM-MVDR[Q]   NM_MVDR_OFB_AUTO con model_quant_1  (lo que corrio la Fase 2)
    NM-MVDR[F]   NM_MVDR_OFB_AUTO con model_1 float
    mono[QQ]     DTLN mono, etapa1 quant + etapa2 quant (el baseline de la Fase 2)
    mono[FF]     DTLN mono, todo float
    mono[FQ]     etapa1 float + etapa2 quant  \\ atribuyen la diferencia a
    mono[QF]     etapa1 quant + etapa2 float  /  cada etapa

Control: el motor sigue calculando su propio mono (columnas dtln_alone_*) con
los interpretes cuantizados; mono[QQ] tiene que coincidir con eso exactamente.

Misma escena base, mismo SMOOTH_STAR/SYS y mismos locutores/interferentes que
la Fase 2; solo se achica la grilla.

USO
---
    conda activate tesis_beam
    python tests/dtln_quant_vs_float_check.py --quick       # 2 celdas (plomeria)
    python tests/dtln_quant_vs_float_check.py               # 24 celdas (~6 min)
    python tests/dtln_quant_vs_float_check.py --audio       # + wavs para escuchar
    python tests/dtln_quant_vs_float_check.py --nonintrusive  # + DNSMOS/SQUIM del motor
"""

import os
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR_OFB_AUTO
from evaluation.nonintrusive import NONINTRUSIVE_KEYS
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "dtln_quant_vs_float")
MODELS = os.path.join(PROJECT_ROOT, "tools", "data", "models")
SIGNALS = os.path.join(PROJECT_ROOT, "tools", "data", "signals")

M1 = {"Q": f"{MODELS}/model_quant_1.tflite", "F": f"{MODELS}/model_1.tflite"}
M2 = {"Q": f"{MODELS}/model_quant_2.tflite", "F": f"{MODELS}/model_2.tflite"}

# Lo que fijo la Fase 2 (celda de Config del notebook).
SMOOTH_STAR = 0.2
SYS = dict(sharpen_exp=8.0, alpha=0.99)

ORDER = ["NM-MVDR[Q]", "NM-MVDR[F]", "mono[QQ]", "mono[FF]", "mono[FQ]", "mono[QF]"]
METRICS = ["PESQ", "STOI", "SDR", "SIR", "SAR"]


def interpreter(path):
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    return itp


class Tap:
    """Guarda la salida de cada celda en wav (si hay audio_dir). Las celdas se
    numeran en el orden en que las corre el motor, que es el mismo orden de las
    filas de ese procesador en el CSV."""

    def __init__(self, name, audio_dir, save_input=False):
        self.name, self.audio_dir, self.save_input = name, audio_dir, save_input
        self.n = 0

    def __call__(self, y, mic_signals, scene_config):
        if self.audio_dir is not None:
            fs = scene_config['fs']
            tag = f"c{self.n:02d}_isir{int(scene_config['isir_db']):+d}"
            sf.write(os.path.join(self.audio_dir, f"{tag}_{self.name}.wav"),
                     np.asarray(y, dtype=np.float32), fs, subtype="FLOAT")
            if self.save_input:
                ref = mic_signals[int(scene_config['ref_mic_idx'])]
                sf.write(os.path.join(self.audio_dir, f"{tag}_input_ref.wav"),
                         np.asarray(ref, dtype=np.float32), fs, subtype="FLOAT")
        self.n += 1


class WithModel:
    """El OFB toma el modelo de scene_config['dtln_model_path']; esto lo pisa
    por procesador para poder correr Q y F en la misma corrida."""

    def __init__(self, proc, model_path, tap):
        self.proc, self.model_path, self.tap = proc, model_path, tap

    def process(self, mic_signals, scene_config):
        cfg = dict(scene_config, dtln_model_path=self.model_path)
        y, w = self.proc.process(mic_signals, cfg)
        self.tap(y, mic_signals, scene_config)
        return y, w


class MonoDTLN:
    """DTLN mono sobre el mic de referencia, la MISMA funcion que usa el motor
    para dtln_alone_*, pero con el par de modelos elegido."""

    def __init__(self, path1, path2, tap):
        self.itp1, self.itp2, self.tap = interpreter(path1), interpreter(path2), tap

    def process(self, mic_signals, scene_config):
        ref = mic_signals[int(scene_config['ref_mic_idx'])]
        y = apply_dtln_post_tflite_realtime(self.itp1, self.itp2, ref)
        self.tap(y, mic_signals, scene_config)
        return y, None


def build_processors(audio_dir):
    procs = {}
    for q in ("Q", "F"):
        name = f"NM-MVDR[{q}]"
        procs[name] = WithModel(NM_MVDR_OFB_AUTO(smooth=SMOOTH_STAR, **SYS), M1[q],
                                Tap(name, audio_dir, save_input=(q == "Q")))
    for q1, q2 in (("Q", "Q"), ("F", "F"), ("F", "Q"), ("Q", "F")):
        name = f"mono[{q1}{q2}]"
        procs[name] = MonoDTLN(M1[q1], M2[q2], Tap(name, audio_dir))
    return procs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="2 celdas (plomeria)")
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--audio", action="store_true",
                    help="guardar los wavs de cada celda en <out-dir>/audio")
    ap.add_argument("--nonintrusive", action="store_true",
                    help="que el motor agregue DNSMOS/SQUIM (columnas *_DNSMOS_*, *_SQUIM_*)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    audio_dir = None
    if args.audio:
        audio_dir = os.path.join(args.out_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

    # Los del motor quedan CUANTIZADOS: son el control de mono[QQ].
    interpreter_1, interpreter_2 = interpreter(M1["Q"]), interpreter(M2["Q"])
    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    targets = [f"{SIGNALS}/p002_emo_adoration_sentences.wav",
               f"{SIGNALS}/p008_emo_contentment_sentences.wav"]
    interf = [f"{SIGNALS}/techno_gated commune.wav",
              f"{SIGNALS}/hairdryer_07_SH_MKH800.wav",
              f"{SIGNALS}/drill_07_RHODE_NT1.wav"]

    # La escena base de la Fase 2.
    base_config = {
        'fs': 16000, 'duration': args.duration, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': targets[0], 'interf_paths': interf,
        'wpe_taps': 5, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'stft_window': 512, 'stft_overlap': 384,
        'dtln_model_path': M1["Q"],
        'eval_references': ['early'],
    }

    # La grilla P1 de la Fase 2, recortada: las 3 salas, los 2 locutores, los
    # layouts de 1 (techno) y 3 interferentes (techno+secador+taladro) y 2 iSIR
    # en los dos lados del cruce de la agenda (2.2 dB).
    layouts = [[(45, 1.0, 0)],
               [(30, 1.0, 0), (-30, 1.0, 1), (60, 1.0, 2)]]
    if args.quick:
        grid = dict(rt60=[0.360], source_path=targets[:1], isir_db=[0, 10],
                    interf_configs=layouts[:1])
    else:
        grid = dict(rt60=[0.160, 0.360, 0.610], source_path=targets, isir_db=[-5, 5],
                    interf_configs=layouts)
    grid.update(
        target_angle=[0], target_dist=[1.0],
        use_wpe=[False],
        mismatch_gain=[0], mismatch_phase=[0],
        error_angle_deg=[0.0], error_distance_m=[0.0],
    )

    procs = build_processors(audio_dir)
    print(f"[*] procesadores: {list(procs)}")

    df = run_mird_grid_search(
        grid_params=grid, dataset_provider=provider, processors=procs,
        scene_base_config=base_config, output_dir=args.out_dir,
        interpreter_1=interpreter_1, interpreter_2=interpreter_2,
        apply_dtln_post=False, save_catalog=False,
        nonintrusive=args.nonintrusive,
    )
    df.to_csv(os.path.join(args.out_dir, "dtln_quant_vs_float.csv"), index=False)
    summarize(df, args.out_dir)


def summarize(df, out_dir):
    key = [c for c in ("rt60", "source", "interf_configs", "isir_db") if c in df.columns]
    cols = {m: f"Delta_tot_{m}_early" for m in METRICS if f"Delta_tot_{m}_early" in df.columns}
    order = [p for p in ORDER if p in df["processor"].unique()]
    by = {p: df[df["processor"] == p].set_index(key) for p in order}

    # --- control: mono[QQ] == el dtln_alone del motor -----------------------
    if "mono[QQ]" in by and "Delta_dtln_alone_PESQ_early" in df.columns:
        eng = df.drop_duplicates(subset=key).set_index(key)
        err = max(np.nanmax(np.abs(by["mono[QQ]"][c].values -
                                   eng[c.replace("Delta_tot_", "Delta_dtln_alone_")]
                                   .reindex(by["mono[QQ]"].index).values))
                  for c in cols.values())
        print(f"\n[control] |mono[QQ] - dtln_alone del motor| max = {err:.2e}"
              + ("  OK" if err < 1e-6 else "  <<< NO COINCIDE"))

    print("\n" + "=" * 78)
    print(f"MEDIANA de Delta_tot sobre {len(by[order[0]])} celdas (referencia early)")
    print("=" * 78)
    t = pd.DataFrame({m: {p: by[p][c].median() for p in order} for m, c in cols.items()})
    print(t.round(3).to_string())

    print("\n--- Efecto de des-cuantizar (F - Q, pareado por celda): mediana [min, max] ---")
    pairs = [("NM-MVDR", "NM-MVDR[F]", "NM-MVDR[Q]"),
             ("mono FF-QQ", "mono[FF]", "mono[QQ]"),
             ("  solo etapa1 (FQ-QQ)", "mono[FQ]", "mono[QQ]"),
             ("  solo etapa2 (QF-QQ)", "mono[QF]", "mono[QQ]")]
    for lbl, a, b in pairs:
        if a not in by or b not in by:
            continue
        d = {m: by[a][c] - by[b][c].reindex(by[a].index) for m, c in cols.items()}
        print(f"  {lbl:22s} " + "  ".join(
            f"{m}:{v.median():+.3f} [{v.min():+.2f},{v.max():+.2f}]" for m, v in d.items()))

    # --- LO QUE IMPORTA: la brecha sistema - mono, antes y despues ----------
    print("\n--- Brecha NM-MVDR - mono (lo que reporta la Fase 2) ---")
    gaps = {}
    for lbl, s, m in (("Fase 2 actual  (Q vs QQ)", "NM-MVDR[Q]", "mono[QQ]"),
                      ("todo float     (F vs FF)", "NM-MVDR[F]", "mono[FF]"),
                      ("OFB quant vs mono float (Q vs FF)", "NM-MVDR[Q]", "mono[FF]")):
        if s not in by or m not in by:
            continue
        g = {k: by[s][c] - by[m][c].reindex(by[s].index) for k, c in cols.items()}
        gaps[lbl] = g
        print(f"  {lbl:34s} " + "  ".join(
            f"{k}:{v.median():+.3f} (gana {int((v > 0).sum())}/{len(v)})"
            for k, v in g.items()))

    if "PESQ" in cols:
        print("\n--- Delta PESQ por celda ---")
        cell = pd.DataFrame({p: by[p][cols["PESQ"]] for p in order})
        print(cell.round(3).to_string())

        print("\n--- Delta PESQ por iSIR (mediana) ---")
        print(df.pivot_table(index="processor", columns="isir_db",
                             values=cols["PESQ"], aggfunc="median")
                .reindex(order).round(3).to_string())

    # --- no intrusivas del motor (solo con --nonintrusive) ------------------
    ni = [k for k in NONINTRUSIVE_KEYS if f"proc_{k}" in df.columns]
    if ni:
        print("\n--- No intrusivas (motor, -26 dBFS): mediana absoluta ---")
        eng = df.drop_duplicates(subset=key)
        t = {"input_ref": {k: eng[f"base_{k}"].median() for k in ni}}
        if f"dtln_alone_{ni[0]}" in df.columns:
            t["dtln_alone (motor)"] = {k: eng[f"dtln_alone_{k}"].median() for k in ni}
        t.update({p: {k: by[p][f"proc_{k}"].median() for k in ni} for p in order})
        print(pd.DataFrame(t).T.round(3).to_string())

    print(f"\n[ok] {out_dir}")


if __name__ == "__main__":
    main()
