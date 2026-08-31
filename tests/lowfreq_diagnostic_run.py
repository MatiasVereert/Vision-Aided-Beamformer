"""
Diagnostico de BAJA FRECUENCIA del NM-MVDR sobre una escena MIRD.

PREGUNTA
--------
La perdida en graves del beamformer mask-based, ¿es el limite FISICO de la
apertura del arreglo, o es error de ESTIMACION de las covarianzas (fuga de voz
dentro de Phi_NN -> auto-cancelacion del target)?

COMO LO RESPONDE
----------------
Corre la MISMA escena por tres filtros y compara curva contra curva, por bin:

    DS                 piso: filtro fijo, sin estadistica que estimar.
    NM_MVDR            el sistema real (mascara DTLN).
    SOUDEN_ORACLE_SCM  cota superior: covarianzas de las senales limpias.

Si el ORACLE tambien se cae en graves -> es apertura, y ningun cambio de
algoritmo lo arregla (abajo de f_c=c/2L la banda le toca al post-filtro).
Si el oracle esta sano y el mask-based no -> es estimacion, y el panel (f)
(fuga de target dentro de Phi_NN) dice si el mecanismo es self-nulling.

POR QUE NO ALCANZA CON EL BENCHMARK
-----------------------------------
PESQ (P.862) filtra por debajo de ~300 Hz: la banda sospechosa es justo la que
la metrica principal NO evalua. Todo lo de aca se mide directo sobre la senal.
La columna `pesq_blind` del CSV marca esa banda.

USO
---
    python tests/lowfreq_diagnostic_run.py
    python tests/lowfreq_diagnostic_run.py --rt60 0.360 --isir 0 --snr-db 30
    python tests/lowfreq_diagnostic_run.py --alpha 1.0 --min-loading 1e-2

Notas de barrido:
  --snr-db  controla el ruido propio de los microfonos. El default del benchmark
            (60 dB) deja el ruido termico 60 dB abajo, asi que la superdirectividad
            queda practicamente sin restriccion. Un arreglo MEMS real esta mas
            cerca de 25-35 dB; el panel de WNG muestra si los pesos se estan
            apoyando en ese margen que el hardware no va a tener.
  --alpha   con fuentes estacionarias alpha->1 es lo optimo. alpha=1.0 corre
            acumulacion estricta y aisla el limite de estimacion del de tracking.

Salida: tests/dataset_out/lowfreq/
    lowfreq_narrowband.csv   por bin de frecuencia y procesador
    lowfreq_thirdoctave.csv  agregado por tercio de octava
    lowfreq_bands.csv        resumen por bandas anchas (marca la banda ciega)
    lowfreq_diagnostic.png   panel de 6 graficos
"""

import os
import argparse

import numpy as np
import pandas as pd
import scipy.signal as sig

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import (
    MirdDatasetProvider, generate_mird_linear_array_from_spacing,
)
from beamforming.array.microphone import Microphone
from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
from evaluation.bf_wrappers import (
    DS, NM_MVDR, NM_MVDR_SUB, SOUDEN_ORACLE_SCM, MVDR_Recursive,
)
from evaluation.lowfreq_diagnostic import (
    narrowband_report, mask_leakage_report, scm_conditioning_report,
    souden_lambda_report,
    aggregate_bands, band_summary_rows, plot_diagnostic,
    array_aperture, critical_frequency, theoretical_dof, align_frames,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "lowfreq")


def build_scene(cfg, provider, rt60, target_angle, target_dist, interf_configs,
                isir_db, snr_db, mismatch_gain=0.0, mismatch_phase=0.0):
    """
    Reproduce los NODOS 1-3 de full_benchmark_test_dtln_mird.py (RIRs MIRD ->
    mezcla -> emulacion de hardware), SIN WPE (use_wpe=False en la config de
    trabajo actual). Devuelve la mezcla observada y las componentes oracle en el
    MISMO dominio, que es lo que exige el diagnostico.
    """
    array_center = np.array(cfg['array_center'])
    mic_coords = generate_mird_linear_array_from_spacing(cfg['mird_spacing']) + array_center
    cfg['mic_coords'] = mic_coords

    scene = SimAcoustic(array_geometry=mic_coords, array_mismatch=0.0,
                        duration=cfg['duration'], fs=cfg['fs'])

    _ = provider.load_rir(rt60, cfg['mird_spacing'], target_dist, target_angle)
    abs_pos_target = array_center + provider.export_position('cartesian').squeeze()
    cfg['source_pos'] = abs_pos_target.reshape(1, 3)
    scene.set_source(cfg['source_path'], gain=1.0, position=abs_pos_target.reshape(1, 3))

    for idx, (i_ang, i_dist) in enumerate(interf_configs):
        _ = provider.load_rir(rt60, cfg['mird_spacing'], i_dist, i_ang)
        abs_pos_interf = array_center + provider.export_position('cartesian').squeeze()
        scene.set_interference(
            audio_path=cfg['interf_paths'][idx % len(cfg['interf_paths'])],
            gain=1.0, position=abs_pos_interf.reshape(1, 3))

    scene.import_rirs(dataset_provider=provider, target_t60=rt60,
                      array_center=array_center, spacing_cfg=cfg['mird_spacing'])
    scene.convolve_signals(t_early=cfg['t_early'])
    scene_data = scene.mix_and_normalize(iSIR_dB=isir_db)

    # --- NODO 3: emulacion de hardware (mismatch=0, solo ruido termico) -------
    # Mismo desglose exacto que el benchmark:
    #   hw_target = mismatch(target_limpio)
    #   hw_noise  = degradada - hw_target  = mismatch(interf) + ruido termico
    # => hw_target + hw_noise == degradada  (exacto, el ruido termico cae entero
    # en el ruido, que es donde corresponde).
    mic_sim = Microphone(fs=cfg['fs'])
    mic_sim.set_seed(1234)
    mic_sim.set_custom_errors(std_gain_dB=mismatch_gain, std_phase_deg=mismatch_phase,
                              snr_dB=snr_db)
    degraded = mic_sim.emulate(scene_data["mic_signals"])

    target_clean = scene_data["target_early"] + scene_data["target_late"]
    hw_target = mic_sim._apply_mismatch(target_clean)
    hw_noise = degraded - hw_target

    return mic_coords, degraded, hw_target, hw_noise, scene_data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt60", type=float, default=0.610)
    ap.add_argument("--spacing", type=str, default="3-3-3-8-3-3-3")
    ap.add_argument("--target-angle", type=float, default=0)
    ap.add_argument("--target-dist", type=float, default=1.0)
    ap.add_argument("--interf-angle", type=float, default=45)
    ap.add_argument("--interf-dist", type=float, default=1.0)
    ap.add_argument("--isir", type=float, default=0.0)
    ap.add_argument("--snr-db", type=float, default=60.0,
                    help="ruido propio de los microfonos (60 = default benchmark)")
    ap.add_argument("--alpha", type=float, default=0.99,
                    help="factor de olvido; con fuentes estacionarias 1.0 es lo optimo")
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--mismatch-gain", type=float, default=0.0,
                    help="desvio de ganancia entre microfonos [dB] (emulacion de HW)")
    ap.add_argument("--mismatch-phase", type=float, default=0.0,
                    help="desvio de fase entre microfonos [grados]")
    ap.add_argument("--geometric", action="store_true",
                    help="agrega el MVDR GEOMETRICO (steering vector + VAD oracle)")
    ap.add_argument("--mu-sweep", type=float, nargs="*", default=[0.0, 0.25, 1.0],
                    help="valores de mu para NM_MVDR_SUB (sustraccion de covarianza). "
                         "0 = MVDR distortionless puro; vacio = no correr la variante.")
    ap.add_argument("--gate-sweep", type=float, nargs="*", default=[],
                    help="umbrales del gate por confianza (lambda_S/M) para NM_MVDR_SUB")
    ap.add_argument("--gate-fmax", type=float, default=None,
                    help="tope de frecuencia opcional del gate [Hz] (None = sin tope)")
    ap.add_argument("--alpha-lf-sweep", type=float, nargs="*", default=[],
                    help="alpha para la banda GRAVE (< --alpha-fsplit). Prueba si "
                         "promediar mas tiempo en graves reduce el error de estimacion.")
    ap.add_argument("--nm-alpha-lf-sweep", type=float, nargs="*", default=[],
                    help="alpha grave sobre el NM_MVDR BASE (sin sustraccion). "
                         "Ablacion limpia del efecto de alpha(f) solo.")
    ap.add_argument("--alpha-fsplit", type=float, default=300.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {
        'fs': 16000,
        'duration': args.duration,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': args.spacing,
        'snr_db': args.snr_db,
        'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],
        'stft_window': 512,
        'stft_overlap': 384,
        'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }
    nperseg = cfg['stft_window']
    noverlap = cfg['stft_overlap']
    hop = nperseg - noverlap
    fs = cfg['fs']

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    print(f"[*] Escena MIRD: rt60={args.rt60}s  spacing={args.spacing}  "
          f"iSIR={args.isir}dB  snr_mic={args.snr_db}dB")
    mic_coords, mixture, oracle_target, oracle_noise, scene_data = build_scene(
        cfg, provider, args.rt60, args.target_angle, args.target_dist,
        [(args.interf_angle, args.interf_dist)], args.isir, args.snr_db,
        mismatch_gain=args.mismatch_gain, mismatch_phase=args.mismatch_phase)

    M = mixture.shape[0]
    ref_ch = M // 2                      # ref_mic_mode=None del benchmark
    L = array_aperture(mic_coords)
    f_c = critical_frequency(L)
    cfg['ref_mic_idx'] = ref_ch
    cfg['oracle_target'] = oracle_target
    cfg['oracle_noise'] = oracle_noise
    cfg['VAD'] = scene_data["VAD"]          # lo consume el MVDR geometrico

    print(f"[*] M={M}  ref_mic={ref_ch}  apertura L={L*100:.1f} cm  ->  f_c=c/2L={f_c:.0f} Hz")
    print(f"    (debajo de f_c el arreglo tiene <2 grados de libertad espaciales)")

    # Frames a descartar: mismo criterio que el benchmark (eval_start_s) para
    # saltear el warm-up de los acumuladores recursivos.
    eval_start_s = min(5.0, cfg['duration'] * 0.3)
    start_frame = int(eval_start_s * fs / hop)
    print(f"[*] descartando los primeros {start_frame} frames ({eval_start_s:.1f}s de warm-up)")

    # --- STFT de las componentes oracle, misma ventana que los wrappers -------
    def _stft(x):
        f_, _, Z = sig.stft(x, fs=fs, window='hamming',
                            nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
        return f_, np.transpose(Z, (1, 2, 0))       # (K, T, M)

    freqs, S_stft = _stft(oracle_target)
    _, N_stft = _stft(oracle_noise)
    _, X_stft = _stft(mixture)

    # --- Mascaras DTLN, mismos parametros que NM_MVDR ------------------------
    print("[*] estimando mascaras DTLN (para el reporte de fuga)")
    mask_s, mask_n = get_dtln_masks_sharpen(
        mixture, ref_ch, cfg['dtln_model_path'],
        block_len=nperseg, block_shift=hop, sharpen_exp=4.0)

    # --- Procesadores --------------------------------------------------------
    processors = {
        "DS": DS(nperseg=nperseg, noverlap=noverlap),
        "NM_MVDR": NM_MVDR(nperseg=nperseg, noverlap=noverlap,
                           min_loading=args.min_loading, alpha=args.alpha),
        "ORACLE_SCM": SOUDEN_ORACLE_SCM(nperseg=nperseg, noverlap=noverlap,
                                        min_loading=args.min_loading, alpha=args.alpha),
    }
    # Variante con sustraccion de covarianza (Phi_SS = Phi_XX - Phi_NN), que
    # elimina el piso lambda >= M de la normalizacion de Souden. mu recorre el
    # trade-off PMWF: 0 = distortionless puro, M = mismo denominador que el core
    # actual.
    if args.geometric:
        # MVDR GEOMETRICO: steering vector desde source_pos + VAD ORACLE. Es la
        # cota superior optimista de la rama geometrica (posicion exacta, VAD
        # perfecto); lo unico que lo degrada es el mismatch de hardware.
        processors["GEO_MVDR"] = MVDR_Recursive(nperseg=nperseg, noverlap=noverlap,
                                                alpha=args.alpha)
    for mu in (args.mu_sweep or []):
        processors[f"NM_MVDR_SUB_mu{mu:g}"] = NM_MVDR_SUB(
            nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
            alpha=args.alpha, mu=mu)
    # Gate por confianza: blend suave hacia el passthrough donde lambda_S/M dice
    # que no hay informacion espacial util. mu=0 (el blend solo es coherente si
    # el beamformer es distortionless).
    for a_lf in (args.nm_alpha_lf_sweep or []):
        processors[f"NM_MVDR_aLF{a_lf:g}"] = NM_MVDR(
            nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
            alpha=args.alpha, alpha_lf=a_lf, alpha_fsplit_hz=args.alpha_fsplit)
    for a_lf in (args.alpha_lf_sweep or []):
        processors[f"SUB_aLF{a_lf:g}"] = NM_MVDR_SUB(
            nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
            alpha=args.alpha, mu=0.0, alpha_lf=a_lf,
            alpha_fsplit_hz=args.alpha_fsplit)
    for th in (args.gate_sweep or []):
        tag = f"SUB_GATE_{th:g}" + ("" if args.gate_fmax is None
                                    else f"_f{args.gate_fmax:g}")
        processors[tag] = NM_MVDR_SUB(
            nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
            alpha=args.alpha, mu=0.0, gate_thresh=th,
            gate_fmax_hz=args.gate_fmax)

    reports, narrow_rows, band_rows, third_rows = {}, [], [], []
    for name, proc in processors.items():
        print(f"\n[*] {name}")
        _, W = proc.process(mixture, cfg)
        print()
        rep = narrowband_report(W, S_stft, N_stft, ref_ch, start_frame=start_frame)
        reports[name] = rep

        for k, f_k in enumerate(freqs):
            narrow_rows.append({
                "processor": name, "freq_hz": float(f_k),
                "AG_dB": 10 * np.log10(max(rep["AG"][k], 1e-30)),
                "TR_dB": 10 * np.log10(max(rep["TR"][k], 1e-30)),
                "NR_dB": 10 * np.log10(max(rep["NR"][k], 1e-30)),
                "WNG_dB": 10 * np.log10(max(rep["WNG"][k], 1e-30)),
                "SNR_in_dB": 10 * np.log10(max(rep["SNR_in"][k], 1e-30)),
                "SNR_out_dB": 10 * np.log10(max(rep["SNR_out"][k], 1e-30)),
                "SD_coh": float(rep["SD_coh"][k]),
            })

        band_rows += band_summary_rows(freqs, rep, label=name)

        agg = aggregate_bands(freqs, rep)
        for i, fc_i in enumerate(agg["fc"]):
            third_rows.append({
                "processor": name, "fc_hz": float(fc_i),
                "AG_dB": 10 * np.log10(max(agg["AG"][i], 1e-30)),
                "TR_dB": 10 * np.log10(max(agg["TR"][i], 1e-30)),
                "NR_dB": 10 * np.log10(max(agg["NR"][i], 1e-30)),
                "WNG_dB": 10 * np.log10(max(agg["WNG"][i], 1e-30)),
                "SD_coh": float(agg["SD_coh"][i]),
            })

    # --- Fuga de mascara y condicionamiento de las covarianzas ---------------
    print("\n[*] fuga de mascara + escalera de autovalores")
    leak = mask_leakage_report(mask_s, mask_n, S_stft, N_stft, ref_ch,
                               start_frame=start_frame)
    Xa, mn = align_frames(X_stft, mask_n[:, :, None])
    cond = scm_conditioning_report(N_stft, X_stft=Xa, mask_n=mn[:, :, 0],
                                   start_frame=start_frame)
    lam = souden_lambda_report(X_stft, mask_s, mask_n,
                               min_loading=args.min_loading, start_frame=start_frame)

    dof = theoretical_dof(freqs, L, M=M)
    for r in narrow_rows:
        k = int(np.argmin(np.abs(freqs - r["freq_hz"])))
        r["dof_theory"] = float(dof[k])
        r["erank_NN_true"] = float(cond["erank_true"][k])
        r["erank_NN_est"] = float(cond["erank_est"][k])
        r["cond_NN_true_dB"] = float(10 * np.log10(max(cond["cond_true"][k], 1e-30)))
        r["leak_target_in_NN"] = float(leak["leak_NN"][k])
        r["contam_noise_in_XX"] = float(leak["cont_XX"][k])
        r["scm_err_rel"] = float(cond["scm_err_rel"][k])
        r["souden_lambda_over_M"] = float(lam["lambda_over_M"][k])
        r["souden_lambda_excess"] = float(lam["lambda_excess"][k])

    df_narrow = pd.DataFrame(narrow_rows)
    df_third = pd.DataFrame(third_rows)
    df_band = pd.DataFrame(band_rows)
    for df in (df_narrow, df_third, df_band):
        df["rt60"] = args.rt60
        df["isir_db"] = args.isir
        df["snr_db"] = args.snr_db
        df["alpha"] = args.alpha
        df["min_loading"] = args.min_loading
        df["aperture_m"] = L
        df["f_c_hz"] = f_c
        df["mismatch_gain"] = args.mismatch_gain
        df["mismatch_phase"] = args.mismatch_phase

    df_narrow.to_csv(os.path.join(args.out_dir, "lowfreq_narrowband.csv"), index=False)
    df_third.to_csv(os.path.join(args.out_dir, "lowfreq_thirdoctave.csv"), index=False)
    df_band.to_csv(os.path.join(args.out_dir, "lowfreq_bands.csv"), index=False)

    png = plot_diagnostic(
        freqs, reports, aperture=L, M=M,
        out_path=os.path.join(args.out_dir, "lowfreq_diagnostic.png"),
        leakage=leak, conditioning=cond, lambda_rep=lam,
        title=(f"NM-MVDR baja frecuencia | MIRD rt60={args.rt60}s  iSIR={args.isir}dB  "
               f"alpha={args.alpha}  loading={args.min_loading:g}  "
               f"L={L*100:.0f}cm  $f_c$={f_c:.0f}Hz  "
               f"mm={args.mismatch_gain:g}dB/{args.mismatch_phase:g}deg"))

    # --- Resumen en consola --------------------------------------------------
    print("\n=== RESUMEN POR BANDAS (dB) ===")
    print("  AG = TR + NR.  TR<0 dB = el filtro esta atenuando el TARGET.")
    print("  'pesq_blind' marca la banda que PESQ no evalua (<300 Hz).\n")
    cols = ["label", "band", "AG_dB", "TR_dB", "NR_dB", "SD_coh", "WNG_dB", "pesq_blind"]
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print(df_band[cols].to_string(index=False,
                                      float_format=lambda v: f"{v:7.2f}"))

    lo = freqs < 300
    print(f"\n=== BANDA <300 Hz ===")
    print(f"  DOF teorico a 200 Hz : {float(theoretical_dof(np.array([200.0]), L, M=M)[0]):.2f}"
          f"   (f_c = {f_c:.0f} Hz)")
    print(f"  rango efectivo Phi_NN verdadera : {cond['erank_true'][lo].mean():.2f}")
    print(f"  rango efectivo Phi_NN estimada  : {cond['erank_est'][lo].mean():.2f}")
    print(f"  target dentro de Phi_NN         : {100*leak['leak_NN'][lo].mean():.1f} %")
    print(f"  ruido dentro de Phi_XX          : {100*leak['cont_XX'][lo].mean():.1f} %")
    print(f"  error relativo de la SCM        : {cond['scm_err_rel'][lo].mean():.3f}")
    print(f"  piso de muestreo sqrt(M/N_eff)  : {cond['est_floor_sqrt_M_over_N']:.3f}"
          f"   <- comparar con la carga diagonal relativa ({args.min_loading:g})")
    print(f"  lambda/M de Souden              : {lam['lambda_over_M'][lo].mean():.3f}"
          f"   (1.0 = degenerado -> w=u/M -> TR={lam['degenerate_TR_dB']:.1f} dB)")
    print(f"  lambda_S/M (exceso sobre el piso): {lam['lambda_excess'][lo].mean():.3f}")

    print(f"\n[*] figura: {png}")
    print(f"[*] CSVs  : {args.out_dir}")


if __name__ == "__main__":
    main()
