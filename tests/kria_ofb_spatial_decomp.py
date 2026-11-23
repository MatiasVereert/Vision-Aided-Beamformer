#!/usr/bin/env python3
"""kria_ofb_spatial_decomp.py -- que hace la etapa ESPACIAL del OFB, por banda.

PREGUNTA. El OFB_AUTO tiene dos etapas: el MVDR guiado por mascara (w^H x) y el
post-filtro espectral (G = smooth + (1-smooth) m). Las metricas del banco miden
las dos juntas. La teoria dice que contra un interferente DIFUSO (CDR bajo) el
MVDR pierde capacidad de cancelar; esto mide si eso pasa en las RIRs medidas.

COMO. El filtro espacial es LINEAL dados sus pesos. Se corre el OFB con
smooth=None (sin post: los pesos son identicos a los de la version con post,
porque el lazo alimenta la red con la salida ANTES del post) sobre la MEZCLA,
y despues se aplican esos mismos pesos frame a frame a cada COMPONENTE por
separado (objetivo solo, interferentes solos, ruido de sensor). Por banda de
octava, sobre la ventana de metricas del banco:

  NR_int   = 10 log10( |X_int,ref|^2 / |w^H X_int|^2 )   atenuacion del interferente
  TG_tgt   = 10 log10( |w^H X_tgt|^2 / |X_tgt,ref|^2 )   ganancia sobre el objetivo
  D_tgt    = 10 log10( |w^H X_tgt - X_tgt,ref|^2 / |X_tgt,ref|^2 )
             distorsion del objetivo respecto de como llega al mic de referencia
             (0 dB = tanto error como senal; MVDR ideal -> -inf)
  SIRg     = NR_int + TG_tgt                               mejora de SIR espacial
  NR_noise = idem para el ruido de sensor (ganancia de ruido blanco del arreglo)

Y del INTERFERENTE solo, la estructura de su covarianza espacial por bin
(promedio largo sobre la misma ventana), promediada en la banda:

  int_ev1, int_ev3 = fraccion de energia en el 1er / los 3 primeros autovalores
  int_erank        = rango efectivo exp(entropia de los autovalores normalizados)

Es lo que de verdad limita al MVDR: el CDR mide coherencia DE A PARES contra un
campo difuso isotropo (infinitas ondas independientes); una sola fuente en una
sala da una covarianza de rango BAJO aunque su CDR sea negativo, y con M grados
de libertad el MVDR puede anular sus direcciones dominantes.

Las escenas son EXACTAMENTE las del banco (full_benchmark_test_dtln_kria.py:
mismo plan, misma mezcla, misma semilla de la emulacion de hardware), asi que
se cruzan fila a fila con su parquet.

USO
    conda run -n tesis_beam python tests/kria_ofb_spatial_decomp.py \\
        --room habitacion_pre_test --n-interf 1 2 --isir -5 0 5 10 15 --workers 5
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, os.path.join(SRC, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BANDS = [125, 250, 500, 1000, 2000, 4000]
VOZ = (300.0, 4000.0)


def _band_masks(freqs):
    out = {b: (freqs >= b / np.sqrt(2)) & (freqs < b * np.sqrt(2)) for b in BANDS}
    out["voz"] = (freqs >= VOZ[0]) & (freqs <= VOZ[1])
    return out


def _db(a, b):
    return float(10 * np.log10((a + 1e-30) / (b + 1e-30)))


_W = {}


def _init(root, room, audio, duration):
    from propagation.kria_rir_loader import KriaRirProvider
    _W["prov"] = KriaRirProvider(**({} if root is None else {"root": root}), verbose=False)
    _W.update(room=room, audio=audio, duration=duration)


def build_scene(prov, room, array, target, combo, isir, audio, duration, seed_plan):
    """La escena del banco: (base, exp, x_degradada, componentes)."""
    import contextlib
    import io
    from full_benchmark_test_dtln_kria import plan_array, build_config, build_grid
    from full_benchmark_test_dtln_mird import compute_scene_seed
    from propagation.simulate_acoustics_v1 import SimAcoustic
    from beamforming.array.microphone import Microphone

    with contextlib.redirect_stdout(io.StringIO()):
        pl = plan_array(prov, room, array, target, [len(combo)], None, seed_plan)
        pl["combos"] = [tuple(combo)]
        base = build_config(prov, room, array, target, audio, duration=duration)
        grid, info = build_grid(room, pl, [isir], prov=prov, interf_audio=audio,
                                source_path=base["source_path"])
        cfg = grid["interf_configs"][0]
        exp = {"rt60": room, "target_src": target,
               "target_angle": grid["target_angle"][0],
               "target_dist": grid["target_dist"][0],
               "interf_configs": cfg, "isir_db": isir,
               "source_path": base["source_path"]}
        center = np.asarray(base["array_center"], dtype=float)
        mic = prov.array_geometry(room, array) + center
        sa = SimAcoustic(array_geometry=mic, array_mismatch=0.0,
                         duration=base["duration"], fs=base["fs"])
        prov.load_rir(room, array, source=target)
        sa.set_source(base["source_path"], gain=1.0,
                      position=(center + prov.export_position("cartesian").squeeze()).reshape(1, 3))
        for (_, _, aidx, name) in cfg:
            prov.load_rir(room, array, source=name)
            sa.set_interference(audio_path=base["interf_paths"][aidx], gain=1.0,
                                position=(center + prov.export_position("cartesian").squeeze()).reshape(1, 3))
        sa.import_rirs(dataset_provider=prov, target_t60=room, array_center=center,
                       spacing_cfg=array, source_names=[target] + [c[3] for c in cfg])
        sa.convolve_signals(t_early=base["t_early"])
        sd = sa.mix_and_normalize(iSIR_dB=isir)
        base["mic_coords"] = mic
        ms = Microphone(fs=base["fs"])
        ms.set_seed(compute_scene_seed(exp, base))
        ms.set_custom_errors(std_gain_dB=0, std_phase_deg=0, snr_dB=base["snr_db"])
        x = ms.emulate(sd["mic_signals"])
    tgt = sd["target_early"] + sd["target_late"]
    itf = sd["interference_early"] + sd["interference_late"]
    comps = {"tgt": tgt, "int": itf, "noise": x - tgt - itf}
    return base, exp, info[str(cfg)], x, comps


def decompose(base, x, comps, isir, ban):
    import contextlib
    import io
    import scipy.signal as sig
    from bf_wrappers import NM_MVDR_OFB_AUTO, resolve_stft_window, ola_taper

    n = base["stft_window"]
    hop = n - base["stft_overlap"]
    fs = base["fs"]
    ref = int(base["ref_mic_mode"])
    scfg = dict(base, ref_mic_idx=ref, isir_db=isir)
    proc = NM_MVDR_OFB_AUTO(smooth=None, sharpen_exp=8.0, ban=ban)
    with contextlib.redirect_stdout(io.StringIO()):
        y, W = proc.process(x, scfg)                       # W: (K, T, M)
    win = resolve_stft_window(scfg, "rect", n)

    def stft(s):
        f, _, Z = sig.stft(s, fs=fs, window=win, nperseg=n, noverlap=base["stft_overlap"], nfft=n)
        return f, np.transpose(Z, (1, 2, 0))               # (K, T, M)

    freqs, X = stft(x)
    Ymix = np.einsum("ktm,ktm->kt", W.conj(), X)
    # CHEQUEO: la salida reconstruida con los pesos devueltos tiene que ser la
    # del procesador (mismo frame <-> mismo peso). Si no, todo lo demas no vale.
    y_re = ola_taper(Ymix, n, hop, "hann", x.shape[1])
    recon_err = float(np.max(np.abs(y_re - y)) / (np.max(np.abs(y)) + 1e-30))

    t0 = int(np.ceil(min(5.0, base["duration"] * 0.3) * fs / hop))   # = eval_start_s del banco
    bm = _band_masks(freqs)
    out = {}
    Yc, Xr = {}, {}
    for k, s in comps.items():
        _, Xc = stft(s)
        Yc[k] = np.einsum("ktm,ktm->kt", W.conj(), Xc)[:, t0:]
        Xr[k] = Xc[:, t0:, ref]
    # estructura espacial del interferente: R(k) = <x x^H> sobre la ventana
    _, Xi = stft(comps["int"])
    Xi = Xi[:, t0:, :]
    R = np.einsum("ktm,ktn->kmn", Xi, Xi.conj()) / Xi.shape[1]
    ev = np.clip(np.linalg.eigvalsh(R)[:, ::-1], 0, None)       # (K, M) desc
    evn = ev / (ev.sum(axis=1, keepdims=True) + 1e-30)
    ev1, ev3 = evn[:, 0], evn[:, :3].sum(axis=1)
    erank = np.exp(-np.sum(evn * np.log(evn + 1e-30), axis=1))
    pw = ev.sum(axis=1)                                         # peso por energia
    for b, m in bm.items():
        P = lambda Z: float(np.sum(np.abs(Z[m]) ** 2))
        nr_int = _db(P(Xr["int"]), P(Yc["int"]))
        tg = _db(P(Yc["tgt"]), P(Xr["tgt"]))
        out[b] = {"NR_int": nr_int, "TG_tgt": tg,
                  "D_tgt": _db(P(Yc["tgt"] - Xr["tgt"]), P(Xr["tgt"])),
                  "SIRg": nr_int + tg,
                  "NR_noise": _db(P(Xr["noise"]), P(Yc["noise"])),
                  "in_SIR": _db(P(Xr["tgt"]), P(Xr["int"])),
                  "int_ev1": float(np.average(ev1[m], weights=pw[m] + 1e-30)),
                  "int_ev3": float(np.average(ev3[m], weights=pw[m] + 1e-30)),
                  "int_erank": float(np.average(erank[m], weights=pw[m] + 1e-30))}
    return out, recon_err


def _run(job):
    idx, array, target, combo, isir, seed_plan = job
    t = time.time()
    prov = _W["prov"]
    base, exp, info, x, comps = build_scene(prov, _W["room"], array, target, combo, isir,
                                            _W["audio"], _W["duration"], seed_plan)
    rows = []
    for ban in (False, True):
        res, err = decompose(base, x, comps, isir, ban)
        for b, r in res.items():
            rows.append({"processor": "OFB_AUTO_BAN" if ban else "OFB_AUTO",
                         "scene_id": info["scene_id"], "array": array,
                         "interf_srcs": ",".join(combo), "K": len(combo),
                         "isir_db": isir, "band": str(b), "recon_err": err, **r,
                         "sep_min_deg": info["sep_min_deg"],
                         **{f"int{j}_CDR_{bb}": info.get(f"int{j}_CDR_{bb}")
                            for j in (1, 2) for bb in BANDS},
                         **{f"int{j}_DRR_{bb}": info.get(f"int{j}_DRR_{bb}")
                            for j in (1, 2) for bb in BANDS},
                         **{f"tgt_CDR_{bb}": info.get(f"tgt_CDR_{bb}") for bb in BANDS}})
    return idx, rows, time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--room", required=True)
    ap.add_argument("--target", default="1")
    ap.add_argument("--n-interf", type=int, nargs="+", default=[1])
    ap.add_argument("--isir", type=float, nargs="+", default=[0.0])
    ap.add_argument("--min-sep", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="solo los primeros N jobs (prueba)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ablacion_nopf"))
    a = ap.parse_args(argv)

    import multiprocessing as mp
    import pandas as pd
    from propagation.kria_rir_loader import KriaRirProvider
    from full_benchmark_test_dtln_kria import plan_array, DEFAULT_INTERF_AUDIO, SIGNALS

    prov = KriaRirProvider(**({} if a.root is None else {"root": a.root}), verbose=False)
    audio = [os.path.join(SIGNALS, p) for p in DEFAULT_INTERF_AUDIO]
    jobs = []
    for arr in prov.arrays(a.room):
        pl = plan_array(prov, a.room, arr, a.target, a.n_interf, None, a.seed, a.min_sep)
        if not pl:
            continue
        for c in pl["combos"]:
            for isir in a.isir:
                jobs.append((len(jobs), arr, a.target, list(c), isir, a.seed))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"[decomp] {len(jobs)} jobs (escena x iSIR, cada uno con y sin BAN) en {a.workers} procesos",
          flush=True)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(max(1, (os.cpu_count() or 2) // max(1, a.workers)))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    rows, t0 = [], time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, a.workers), initializer=_init,
                  initargs=(a.root, a.room, audio, a.duration)) as pool:
        for k, (idx, r, dt) in enumerate(pool.imap_unordered(_run, jobs), 1):
            rows += r
            print(f"[decomp] {k}/{len(jobs)} {jobs[idx][3]} iSIR {jobs[idx][4]:+g} "
                  f"({dt:.0f} s, total {time.time() - t0:.0f} s, recon {r[0]['recon_err']:.1e})",
                  flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"kria_{a.room}_spatial_decomp")
    df.to_parquet(stem + ".parquet", index=False)
    df.to_csv(stem + ".csv", index=False)
    print(f"[decomp] {len(df)} filas -> {stem}.parquet  (recon_err max {df.recon_err.max():.1e})")
    v = df[df.band == "voz"]
    print(v.pivot_table(index=["processor", "interf_srcs"], columns="isir_db",
                        values="NR_int").round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
