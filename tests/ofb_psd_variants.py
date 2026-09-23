"""
ofb_psd_variants.py
===================
QUE PARTE DE LA PROYECCION PSD HACE FALTA, Y CON QUE SE PUEDE REEMPLAZAR.

`tests/ofb_psd_ablation.py` cerro que sacar la proyeccion a lo bruto rompe el
sistema (Delta PESQ +1.02 -> -0.09 en MIRD; la salida del aro se va a +0.7 dBFS).
El mecanismo medido: Phi_SS = Phi_XX - Phi_NN sale del cono PSD en el 99.6 % de
los bins, y aunque los autovalores negativos son chicos (~4 % del mayor), la
TRAZA los suma y lambda_S = tr(Phi_NN^-1 Phi_SS) termina NEGATIVO en el 27 % de
los bins. Ahi `max(lambda_S, 1e-3)` divide por el piso -> ganancia x1000, y de
paso la columna del numerador conserva el signo negativo -> se invierte la fase
contra el mic de referencia. Se rompe la restriccion distortionless.

Pero la matriz proyectada entra en los DOS lados de

    w = B e_ref / lambda_S ,   B = Phi_NN^-1 Phi_SS ,  lambda_S = tr(B)

asi que "hace falta para la normalizacion" es una HIPOTESIS mientras no se
separen los dos roles. Eso es lo primero que mide este barrido.

LAS VARIANTES
-------------
  psd       (referencia) la proyeccion completa: eigh, clip de negativos,
            recomposicion. Es el 81 % del `solve` y el ~77 % del sistema.
  none      (control) sin proyectar. Ya medido: se rompe.

  -- DIAGNOSTICO: separan los dos roles. Las dos usan el eigh, asi que NO son
     candidatas de port: son el instrumento que decide que hay que reemplazar.
  lam_only  numerador CRUDO, lambda_S de la matriz PROYECTADA. Si esto recupera
            la calidad, el valor de la proyeccion esta SOLO en la normalizacion.
  num_only  numerador PROYECTADO, lambda_S CRUDO. El complemento.

  -- CANDIDATAS DE PORT: ninguna descompone nada.
  fallback  sin proyectar, pero el bin cuyo lambda_S cae por debajo del piso se
            declara no estimable y se le da w = e_ref (pasa el mic de
            referencia) en vez de amplificar basura por mil. Cuesta una
            comparacion por bin.
  rank1     Phi_SS <- max(lambda_1, 0) v v^H con (lambda_1, v) del autovector
            dominante por ITERACION DE POTENCIA (pocas iteraciones, M^2 cada
            una, sin descomposicion). Dos argumentos a favor: (a) rank-1 es PSD
            por construccion, asi que lambda_S > 0 sale gratis; (b) con UNA
            fuente target Phi_SS DEBERIA ser rank-1, o sea que truncar no es
            una aproximacion sino imponer la estructura correcta.
            Ojo con la elegancia escondida: con Phi_SS = l1 v v^H, l1 se cancela
            entre numerador y traza y queda w = (Phi_NN^-1 v) v_ref^* /
            (v^H Phi_NN^-1 v), o sea el MVDR clasico con la RTF sacada del
            autovector principal. El mismo objeto que el front-end del lazo
            ciego estimaba con un eigh entero (ver blind_feedback.py).
            El arranque de la iteracion es la COLUMNA ref de Phi_SS, que si la
            matriz fuera exactamente rank-1 ya seria el autovector: se arranca
            practicamente convergido.

USO
---
    conda activate tesis_beam
    python tests/ofb_psd_variants.py --only-cost
    python tests/ofb_psd_variants.py
"""

import os
import sys
import time
import argparse

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for d in (os.path.join(PROJECT_ROOT, "src"), os.path.join(PROJECT_ROOT, "tests")):
    if d not in sys.path:
        sys.path.insert(0, d)

import beamforming.mask.ofb as ofb_mod                              # noqa: E402
from beamforming.mask.blind_feedback import SoudenSubtractCore      # noqa: E402
from beamforming.mask.ofb import SoudenCore as _ORIG_CORE           # noqa: E402
from evaluation.bf_wrappers import OFB_MVDR                         # noqa: E402
from propagation.mird_loader import MirdDatasetProvider             # noqa: E402
from ofb_refactor_equivalence import (si_sdr, FS, MIRD_ROOT, MODEL_1,
                                      SPACING, SOURCE_WAV, INTERF_WAV)  # noqa: E402
from ofb_block_update_sweep import ARO12_WAV                        # noqa: E402

VARIANTS = ("psd", "none", "lam_only", "num_only", "fallback", "rank1",
            "rank2", "rank3", "ns8", "ns16", "ns16nf", "nosub")
OUT_ARO = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "ofb_psd_variants")


class VariantCore(SoudenSubtractCore):
    """El nucleo con el tratamiento de Phi_SS intercambiable. `update` no cambia."""

    variant = "psd"
    rank1_iters = 4

    def _psd_project(self, Phi_SS):
        evals, evecs = np.linalg.eigh(Phi_SS)
        evals = np.maximum(evals, 0.0)
        return np.einsum("fmp,fnp->fmn", evecs * evals[:, None, :], evecs.conj())

    def _rank1(self, Phi_SS):
        """
        Autovector dominante por iteracion de potencia y truncamiento a rank-1.
        Arranca en la columna `ref`: si Phi_SS fuera exactamente rank-1 (una sola
        fuente), esa columna YA es el autovector salvo escala.
        """
        v = Phi_SS[:, :, self.ref].copy()
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-20)
        for _ in range(self.rank1_iters):
            v = np.einsum("fmn,fn->fm", Phi_SS, v)
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-20)
        lam1 = np.real(np.einsum("fm,fmn,fn->f", v.conj(), Phi_SS, v))
        lam1 = np.maximum(lam1, 0.0)
        return lam1[:, None, None] * np.einsum("fm,fn->fmn", v, v.conj())

    def _rank_r(self, Phi_SS, r):
        """
        Truncamiento a rank-r por ITERACION ORTOGONAL (block power): sin
        descomposicion de la matriz grande. El unico eigh que queda es el de la
        proyeccion r x r, que para r=2 o 3 es forma cerrada en C++.
        Arranca en r columnas de Phi_SS, entre ellas la `ref`.
        """
        M = self.M
        cols = [(self.ref + j) % M for j in range(r)]
        Q, _ = np.linalg.qr(Phi_SS[:, :, cols])
        for _ in range(self.rank1_iters):
            Q, _ = np.linalg.qr(np.einsum("fmn,fnr->fmr", Phi_SS, Q))
        S = np.einsum("fmr,fmn,fns->frs", Q.conj(), Phi_SS, Q)
        S = 0.5 * (S + np.conj(np.transpose(S, (0, 2, 1))))
        ev, V = np.linalg.eigh(S)
        Sp = np.einsum("frp,fsp->frs", V * np.maximum(ev, 0.0)[:, None, :], V.conj())
        return np.einsum("fmr,frs,fns->fmn", Q, Sp, Q.conj())

    def _newton_schulz(self, Phi_SS, iters):
        """
        Parte positiva SIN autovectores:  A_+ = (A + |A|)/2,  |A| = U^H A,
        con U (factor polar unitario) por Newton-Schulz,
            X_{k+1} = 0.5 X_k (3I - X_k^H X_k),   X_0 = A / ||A||_F
        El escalado por Frobenius garantiza sigma <= 1, que es la condicion de
        convergencia. Son PURAS multiplicaciones de matrices: sin raices, sin
        ramas, sin ordenar autovalores -- el kernel que mejor le sienta a NEON.
        """
        I = self.eye
        nrm = np.linalg.norm(Phi_SS, axis=(1, 2), keepdims=True) + 1e-30
        X = Phi_SS / nrm
        for _ in range(iters):
            X = 0.5 * (X @ (3.0 * I - np.conj(np.transpose(X, (0, 2, 1))) @ X))
        absA = np.conj(np.transpose(X, (0, 2, 1))) @ Phi_SS
        absA = 0.5 * (absA + np.conj(np.transpose(absA, (0, 2, 1))))
        return 0.5 * (Phi_SS + absA)

    def solve(self):
        M = self.M
        Phi_XX = self.Num_XX / (self.Den_XX + 1e-15)
        Phi_NN = self.Num_NN / (self.Den_NN + 1e-15)
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))
        Phi_SS = Phi_XX - Phi_NN
        Phi_SS = 0.5 * (Phi_SS + np.conj(np.transpose(Phi_SS, (0, 2, 1))))

        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        Phi_NN_s = Phi_NN + self.eye * ((self.min_loading * (tr_Phi / M))[:, None, None] + 1e-12)

        v = self.variant
        if v == "psd":
            B = np.linalg.solve(Phi_NN_s, self._psd_project(Phi_SS))
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        elif v == "none":
            B = np.linalg.solve(Phi_NN_s, Phi_SS)
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        elif v in ("lam_only", "num_only"):
            B_raw = np.linalg.solve(Phi_NN_s, Phi_SS)
            B_prj = np.linalg.solve(Phi_NN_s, self._psd_project(Phi_SS))
            lam_raw = np.real(np.trace(B_raw, axis1=1, axis2=2))
            lam_prj = np.real(np.trace(B_prj, axis1=1, axis2=2))
            B, lam = ((B_raw, lam_prj) if v == "lam_only" else (B_prj, lam_raw))
        elif v == "fallback":
            B = np.linalg.solve(Phi_NN_s, Phi_SS)
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        elif v == "rank1":
            B = np.linalg.solve(Phi_NN_s, self._rank1(Phi_SS))
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        elif v in ("rank2", "rank3"):
            B = np.linalg.solve(Phi_NN_s, self._rank_r(Phi_SS, int(v[-1])))
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        elif v == "nosub":
            # SIN SUSTRACCION: la covarianza de la mezcla enmascarada entra
            # directo en Souden. Phi_XX = sum w_t x x^H con w_t >= 0 es PSD POR
            # CONSTRUCCION, asi que no hay nada que proyectar y lambda = lambda_S
            # + M no puede acercarse a cero: se cae el eigh Y el piso.
            # El precio conocido (souden_mvdr.py): donde la mascara no encuentra
            # voz, Phi_XX -> Phi_NN y w -> u/M, o sea -20log10(M) dB con ganancia
            # de arreglo nula. PESQ no lo ve (P.862 es ciega debajo de 300 Hz):
            # ver el diagnostico por bandas.
            Phi_XX_h = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
            B = np.linalg.solve(Phi_NN_s, Phi_XX_h)
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        elif v.startswith("ns"):
            it = int(v[2:].replace("nf", ""))
            B = np.linalg.solve(Phi_NN_s, self._newton_schulz(Phi_SS, it))
            lam = np.real(np.trace(B, axis1=1, axis2=2))
        else:
            raise ValueError(f"variante desconocida: {v!r} {VARIANTS}")

        bad = lam < self.lambda_floor            # bins no estimables
        lam = np.maximum(lam, self.lambda_floor)
        weights = B[:, :, self.ref] / (lam[:, None] + self.mu + 1e-15)
        if v in ("fallback", "rank1", "rank2", "rank3") or (
                v.startswith("ns") and not v.endswith("nf")):
            # En vez de amplificar basura por 1/piso, el bin pasa el mic de
            # referencia: el sistema DEGRADA al canal crudo en vez de explotar.
            weights = np.where(bad[:, None], self.eye[0, :, self.ref][None, :], weights)
        return weights


def make_wrapper(variant, iters):
    """`OFB_MVDR` con el nucleo cambiado SOLO durante su propio `process`."""
    core_cls = type(f"Core_{variant}", (VariantCore,),
                    {"variant": variant, "rank1_iters": iters})

    class _W(OFB_MVDR):
        def process(self, mic_signals, scene_config):
            ofb_mod.SoudenCore = core_cls
            try:
                return super().process(mic_signals, scene_config)
            finally:
                ofb_mod.SoudenCore = _ORIG_CORE
    _W.__name__ = f"OFB_MVDR_{variant}"
    return _W


def part_cost(args):
    """Cuanto cuesta cada variante (x86; lo que importa es la RELACION)."""
    K, M, reps = 257, args.cost_m, args.cost_reps
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))) / np.sqrt(2)
    print(f"\n--- costo del `solve` (K={K}, M={M}, {reps} repeticiones) ---")
    base = None
    for v in args.variants:
        core = type("C", (VariantCore,), {"variant": v, "rank1_iters": args.rank1_iters})(
            K, M, M // 2)
        for _ in range(30):
            core.update(X, np.full(K, 0.3), np.full(K, 0.7))
        core.solve()
        t0 = time.perf_counter()
        for _ in range(reps):
            core.solve()
        ms = 1e3 * (time.perf_counter() - t0) / reps
        base = ms if base is None else base
        print(f"  {v:<10} {ms:7.3f} ms   ({base/ms:4.2f}x contra 'psd')")


def part_mird(args):
    import tensorflow as tf
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

    out_dir = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_psd_variants")
    os.makedirs(out_dir, exist_ok=True)
    itp = []
    for path in (MODEL_1, MODEL_1.replace("_1.tflite", "_2.tflite")):
        i = tf.lite.Interpreter(model_path=path)
        i.allocate_tensors()
        itp.append(i)

    base_config = {
        'fs': FS, 'duration': args.dur, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': args.spacing, 'snr_db': 60.0,
        'source_path': SOURCE_WAV, 'interf_paths': [INTERF_WAV],
        'wpe_taps': 7, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
        'dtln_model_path': MODEL_1,
    }
    param_grid = {
        'rt60': args.rt60, 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(a, 1.0)] for a in args.interf], 'isir_db': args.isir,
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    procs = {v: make_wrapper(v, args.rank1_iters)(smooth=args.smooth, return_weights=False)
             for v in args.variants}
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=MirdDatasetProvider(root_dir=MIRD_ROOT),
        processors=procs, scene_base_config=base_config, output_dir=out_dir,
        interpreter_1=itp[0], interpreter_2=itp[1],
        apply_dtln_post=False, save_catalog=False)

    order = [v for v in args.variants]
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early"] if c in df.columns]
    print("\n--- MIRD: Delta PESQ por iSIR (media sobre salas) ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").reindex(order).round(3).to_string())
    print("\n--- Delta STOI por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_STOI_early").reindex(order).round(3).to_string())
    print(f"\n--- mediana sobre las {len(df)//max(len(order),1)} celdas ---")
    print(df.groupby("processor")[cols].median().reindex(order).round(4).to_string())
    key = [c for c in ("rt60", "isir_db") if c in df.columns]
    ref = df[df.processor == "psd"].set_index(key)["Delta_bf_PESQ_early"]
    print("\n--- contra 'psd', Delta PESQ ---")
    for v in order:
        if v == "psd":
            continue
        cur = df[df.processor == v].set_index(key)["Delta_bf_PESQ_early"]
        d = cur - ref.reindex(cur.index)
        print(f"  {v:<10} media {d.mean():+.4f}   peor {d.min():+.4f}   "
              f"gana {int((d > 0).sum())}/{len(d)}")
    print("\n--- Delta PESQ por SALA ---")
    print(df.pivot_table(index="processor", columns="rt60",
                         values="Delta_bf_PESQ_early").reindex(order).round(3).to_string())
    print("\n--- Delta SIR por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_SIR_early").reindex(order).round(2).to_string())
    df.to_parquet(os.path.join(out_dir, f"psd_variants{args.tag}.parquet"))


def part_aro12(args):
    import soundfile as sf
    from evaluation.full_benchmark_real import (load_multichannel_wav, energy_vad,
                                                segmental_snr_estimate, rms_db)
    os.makedirs(OUT_ARO, exist_ok=True)
    mic_all, fs = load_multichannel_wav(args.aro_wav, expected_channels=12)
    i0 = int(round(args.aro_skip * fs))
    mic = np.ascontiguousarray(mic_all[:, i0:i0 + int(round(args.aro_dur * fs))])
    ref_mic = mic.shape[0] // 2
    cfg = {'fs': fs, 'stft_window': 512, 'stft_overlap': 384,
           'dtln_model_path': MODEL_1, 'ref_mic_idx': ref_mic}

    outs = {}
    for v in args.variants:
        t0 = time.perf_counter()
        y, _ = make_wrapper(v, args.rank1_iters)(smooth=args.smooth,
                                                 return_weights=False).process(mic, cfg)
        outs[v] = y
        print(f"[*] aro12 {v}: {time.perf_counter()-t0:.1f} s")

    x_ref = mic[ref_mic]
    peak = max(max(np.max(np.abs(y)) for y in outs.values()), np.max(np.abs(x_ref)))
    sc = 0.95 / (peak + 1e-12)
    sf.write(os.path.join(OUT_ARO, "ref_mic_raw.wav"),
             (x_ref * sc).astype(np.float32), fs, subtype="PCM_16")
    for v, y in outs.items():
        sf.write(os.path.join(OUT_ARO, f"ofb_{v}.wav"),
                 (y * sc).astype(np.float32), fs, subtype="PCM_16")
    try:
        from evaluation.nonintrusive import compute_nonintrusive, NONINTRUSIVE_KEYS
    except Exception:
        compute_nonintrusive, NONINTRUSIVE_KEYS = None, []
    print("\n--- aro12 (M=12) ---")
    print(f"{'variante':<10} {'RMS dBFS':>9} {'segSNR':>7}" +
          "".join(f"{k.replace('DNSMOS_','').replace('SQUIM_','SQ_'):>8}"
                  for k in NONINTRUSIVE_KEYS) + f"{'SI-SDR vs psd':>14}")
    for name, x in [("ref_mic", x_ref)] + list(outs.items()):
        ni = compute_nonintrusive(x, fs) if compute_nonintrusive else {}
        seg = segmental_snr_estimate(x, energy_vad(x, fs))
        d = si_sdr(x, outs["psd"]) if name in outs and name != "psd" else float('nan')
        print(f"{name:<10} {rms_db(x):9.1f} {seg:7.2f}" +
              "".join(f"{ni.get(k, np.nan):8.2f}" for k in NONINTRUSIVE_KEYS) +
              (f"{d:14.1f}" if np.isfinite(d) else f"{'--':>14}"))
    print(f"[*] WAVs: {OUT_ARO}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--rank1-iters", type=int, default=4)
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--isir", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 15.0])
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--spacing", default=SPACING,
                    help="OJO: rt60=0.160 solo existe con 8-8-8-8-8-8-8")
    ap.add_argument("--interf", type=float, nargs="+", default=[45.0])
    ap.add_argument("--tag", default="", help="sufijo del parquet de salida")
    ap.add_argument("--aro-wav", default=ARO12_WAV)
    ap.add_argument("--aro-skip", type=float, default=8.0)
    ap.add_argument("--aro-dur", type=float, default=15.0)
    ap.add_argument("--cost-m", type=int, default=12)
    ap.add_argument("--cost-reps", type=int, default=30)
    ap.add_argument("--only-cost", action="store_true")
    ap.add_argument("--only-aro12", action="store_true")
    ap.add_argument("--no-aro12", action="store_true")
    args = ap.parse_args()

    part_cost(args)
    if args.only_cost:
        return
    if args.only_aro12:
        part_aro12(args)
        return
    part_mird(args)
    if not args.no_aro12:
        part_aro12(args)


if __name__ == "__main__":
    main()
