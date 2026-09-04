"""
Exporta a WAV los audios del catalogo H5 del benchmark, para escuchar.

El catalogo guarda el MEJOR y el PEOR caso por (procesador, metrica). Para
comparar de oido lo que importa es escuchar la MISMA celda pasada por todos los
procesadores, asi que este script busca la celda que aparece en el catalogo para
la mayor cantidad de procesadores y exporta esa.

    ref_early.wav          la referencia (target early) = lo que hay que estimar
    input_mic<N>.wav       el canal de referencia crudo = el punto de partida
    dtln_alone.wav         el DTLN mono, baseline sin arreglo
    <PROCESADOR>.wav       la salida de cada beamformer

Todos normalizados con el MISMO factor (el pico del conjunto) para que las
diferencias de nivel entre procesadores se escuchen tal cual son y no se pierdan
en una normalizacion por archivo.

USO
---
    python tests/ds_mask_export_audio.py
    python tests/ds_mask_export_audio.py --metric Delta_tot_SIR --case worst_case
    python tests/ds_mask_export_audio.py --list
"""

import os
import argparse
from collections import Counter

import numpy as np
import h5py
from scipy.io import wavfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BENCH = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ds_mask_bench")


def _num(attrs, key):
    """El catalogo guarda los NaN como la CADENA "NaN" (ver
    full_benchmark_test_dtln_mird.save_extreme_case_to_master)."""
    v = attrs.get(key, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def cell_key(md):
    return f"rt{md['rt60']:g}_ang{md['interf_configs']}_isir{md['isir_db']:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=os.path.join(BENCH, "mird_benchmark_catalog.h5"))
    ap.add_argument("--out-dir", default=os.path.join(BENCH, "audio"))
    ap.add_argument("--metric", default="Delta_tot_PESQ")
    ap.add_argument("--case", default="best_case", choices=["best_case", "worst_case"])
    ap.add_argument("--ref-mic", type=int, default=None,
                    help="canal de referencia a exportar (default M//2)")
    ap.add_argument("--list", action="store_true", help="solo listar el catalogo")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        procs = list(f.keys())
        if args.list:
            for p in procs:
                for m in f[p]:
                    for c in f[p][m]:
                        md = f[f"{p}/{m}/{c}"]["metadata"].attrs
                        print(f"  {p:16s} {m:16s} {c:10s} {cell_key(md)}")
            return

        # celda mas representada para (metrica, caso)
        have = {}
        for p in procs:
            path = f"{p}/{args.metric}/{args.case}"
            if path in f:
                have[p] = cell_key(f[path]["metadata"].attrs)
        if not have:
            raise SystemExit(f"no hay {args.metric}/{args.case} en el catalogo")
        cell, n = Counter(have.values()).most_common(1)[0]
        keep = [p for p in procs if have.get(p) == cell]
        print(f"[*] celda {cell}  ({n} de {len(procs)} procesadores la comparten)")
        if len(keep) < len(procs):
            print(f"    fuera: {[p for p in procs if p not in keep]} "
                  f"(su {args.case} de {args.metric} cayo en otra celda)")

        os.makedirs(args.out_dir, exist_ok=True)
        g0 = f[f"{keep[0]}/{args.metric}/{args.case}"]
        fs = int(g0["metadata"].attrs["fs"])
        mics = np.asarray(g0["audio"]["mic_signals"])
        # El catalogo NO guarda ref_mic_idx en metadata (va en el CSV, no en el
        # exp_config). Se cae al MISMO default de toda la familia -- M//2, que es
        # el canal sobre el que proyectan los beamformers y contra el que se
        # miden las metricas intrusivas. Escuchar otro canal seria comparar
        # contra algo que el sistema no estima.
        ref_mic = mics.shape[0] // 2 if args.ref_mic is None else args.ref_mic

        tracks = {"ref_early": np.asarray(g0["audio"]["target_reference"]),
                  f"input_mic{ref_mic}": mics[ref_mic]}
        if "processed_dtln_alone" in g0["audio"]:
            tracks["dtln_alone"] = np.asarray(g0["audio"]["processed_dtln_alone"])
        for p in keep:
            g = f[f"{p}/{args.metric}/{args.case}"]
            tracks[p] = np.asarray(g["audio"][f"processed_{p}"])
            print(f"    {p:16s} dPESQ={_num(g['metrics'].attrs, 'Delta_tot_PESQ_early'):+.3f}"
                  f"  dSIR={_num(g['metrics'].attrs, 'Delta_tot_SIR_early'):+.2f}")

    # normalizacion COMUN: preserva las diferencias de nivel entre procesadores
    peak = max(float(np.max(np.abs(v))) for v in tracks.values())
    for name, v in tracks.items():
        out = os.path.join(args.out_dir, f"{name}.wav")
        wavfile.write(out, fs, (0.95 * v / (peak + 1e-12)).astype(np.float32))
    print(f"\n[ok] {len(tracks)} wav en {args.out_dir}")


if __name__ == "__main__":
    main()
