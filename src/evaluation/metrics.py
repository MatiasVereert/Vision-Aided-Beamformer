import numpy as np
from scipy import signal
import pb_bss
from pb_bss.evaluation import OutputMetrics
import fast_bss_eval


def _fast_sdr_sir_sar(reference: np.ndarray, estimation: np.ndarray) -> tuple:
    """
    SDR/SIR/SAR del canal [0] via fast_bss_eval. Es el MISMO criterio BSS-Eval
    que mir_eval.separation.bss_eval_sources (filtro de 512 taps por LS), pero
    ~3x mas rapido.

    IMPORTANTE:
    - NO usar use_cg_iter (gradiente conjugado): con 10 iteraciones no converge y
      el SIR de los casos buenos sale +inf (ver comentario abajo). Se usa el
      solver directo via square_cosine_metrics(pairwise=True), porque
      bss_eval_sources(compute_permutation=False) con solver directo tira
      ValueError en np.linalg.solve con NumPy 2 (bug de fast_bss_eval 0.1.4).
    - Se asume que la estimacion ya esta en el orden de las fuentes (se toma [0, 0]).
    - La fila [0] es independiente de las demas filas de la estimacion (verificado),
      asi que un canal secundario dummy no afecta el resultado del target.
    - Para fuente unica, SIR sale inf (no hay interferencia); el llamador decide.
    """
    ref = np.ascontiguousarray(reference, dtype=np.float64)
    est = np.ascontiguousarray(estimation, dtype=np.float64)
    # Solver DIRECTO (exacto) en modo pairwise. Antes: use_cg_iter=10, que NO
    # converge: el SIR sale de coh_sdr/coh_sar (dos proyecciones casi iguales
    # cuando la interferencia residual es chica) y con 10 iteraciones de CG ese
    # cociente daba >=1 -> SIR=+inf -> NaN, justo en los MEJORES casos
    # (objetivo puntual, SIR real ~25-28 dB); tambien sesgaba SDR/SAR ~0.3 dB.
    # El solver directo de fast_bss_eval 0.1.4 esta roto con NumPy 2 SOLO en
    # modo no-pairwise (RHS vectorial en lote); en pairwise el RHS es matricial
    # y funciona. El elemento [0, 0] (ref 0 vs est 0) es el de mir_eval;
    # verificado vs minimos cuadrados explicitos (<=0.05 dB) e independiente de
    # la fila 1 de la estimacion. Mismo costo (~0.1 s por 10 s de audio).
    from fast_bss_eval.numpy.metrics import (square_cosine_metrics,
                                             _base_metrics_bss_eval)
    coh_sdr, coh_sar = square_cosine_metrics(
        ref, est, filter_length=512, use_cg_iter=None, pairwise=True)
    nsdr, nsir, nsar = _base_metrics_bss_eval(coh_sdr, coh_sar)
    return float(-nsdr[0, 0]), float(-nsir[0, 0]), float(-nsar[0, 0])

def precise_slice_alignment(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, max_shift_s: float = 0.5) -> tuple:
    """
    Aligns signals by slicing out the unaligned portions using cross-correlation.
    Preserved exactly to handle temporal processing delays before metric evaluation.
    """
    # Emphasize transients for better phase-alignment using a high-pass filter
    b, a = signal.butter(4, 300 / (fs / 2), btype='high')
    filt_ref = signal.filtfilt(b, a, ref_sig)
    filt_deg = signal.filtfilt(b, a, deg_sig)

    max_shift_samples = int(max_shift_s * fs)

    # Calculate cross-correlation on the actual waveforms
    corr = signal.correlate(filt_deg, filt_ref, mode='full', method='fft')
    lags = signal.correlation_lags(len(filt_deg), len(filt_ref), mode='full')

    valid_idx = np.where(np.abs(lags) <= max_shift_samples)[0]
    if len(valid_idx) == 0:
        shift = 0
    else:
        best_idx = valid_idx[np.argmax(corr[valid_idx])]
        shift = lags[best_idx]

    # Slice arrays to perfectly match without introducing zero-padding artifacts
    if shift > 0:
        aligned_deg = deg_sig[shift:]
        aligned_ref = ref_sig[:-shift]
    elif shift < 0:
        shift_abs = abs(shift)
        aligned_ref = ref_sig[shift_abs:]
        aligned_deg = deg_sig[:-shift_abs]
    else:
        aligned_ref = ref_sig.copy()
        aligned_deg = deg_sig.copy()

    min_len = min(len(aligned_ref), len(aligned_deg))
    return aligned_ref[:min_len], aligned_deg[:min_len], shift


def _pesq_single(ref, deg, fs):
    """PESQ de UN par referencia/degradada (mismo modo que el facade de
    pb_bss: 'wb' a 16 kHz, 'nb' a 8 kHz)."""
    from pb_bss.evaluation.module_pesq import pesq as _pb_pesq
    return float(np.squeeze(_pb_pesq(np.asarray(ref), np.asarray(deg), fs)))


def _safe_metric(getter):
    """
    Evalua una metrica de forma AISLADA. Si su calculo lanza una excepcion
    (p.ej. PESQ 'NoUtterancesError' cuando la señal esta demasiado degradada y su
    VAD interno no detecta habla), devuelve NaN sin tumbar al resto de las
    metricas de la misma llamada.
    """
    try:
        return float(getter())
    except Exception:
        return np.nan


def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int,
                           interf_early: np.ndarray = None,
                           interf_late: np.ndarray = None,
                           target_late: np.ndarray = None,
                           eval_start_s: float = 5.0,
                           **kwargs) -> dict:
    """
    Master evaluation function.
    Uses pb_bss for PESQ, STOI, SI-SDR, and traditional SDR/SIR/SAR via multi-source evaluation.
    Absorbs extra kwargs to maintain benchmark compatibility.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    results = {}

    # 1. Strict physical slice alignment
    aligned_ref, aligned_deg, shift = precise_slice_alignment(ref_sig, deg_sig, fs)

    # 2. Extract steady-state signals to avoid penalizing algorithm convergence
    start_idx = int(eval_start_s * fs)
    min_len = len(aligned_ref)

    if start_idx < min_len:
        ref_crop = aligned_ref[start_idx:]
        deg_crop = aligned_deg[start_idx:]
    else:
        # Fallback just in case the signal is shorter than the crop time
        ref_crop = aligned_ref
        deg_crop = aligned_deg

    # 3. Core multi-source Metrics via pb_bss OutputMetrics Facade
    try:
        # Check if secondary noise components are available to build the full subspace
        if interf_early is not None and interf_late is not None and target_late is not None:

            # Align secondary paths using the same shift parameter.
            # The noise components (interf/target_late) live in the CLEAN reference
            # domain (same time base as ref_sig, no processing delay), so they must
            # be sliced EXACTLY like aligned_ref in precise_slice_alignment -- NOT
            # like deg. Using deg's slicing offsets them by `shift` samples, which
            # (for shift beyond mir_eval's 512-tap filter) corrupts SIR/SAR.
            def align_secondary(sig):
                if sig is None: return None
                sig_sq = np.squeeze(sig)
                if shift > 0:
                    arr = sig_sq[:-shift]
                elif shift < 0:
                    arr = sig_sq[abs(shift):]
                else:
                    arr = sig_sq.copy()
                return arr[:min_len]

            aligned_ie = align_secondary(interf_early)
            aligned_il = align_secondary(interf_late)
            aligned_tl = align_secondary(target_late)

            if start_idx < min_len:
                ie_crop = aligned_ie[start_idx:]
                il_crop = aligned_il[start_idx:]
                tl_crop = aligned_tl[start_idx:]
                # Composite noise is everything spatial/reverberant we do not want
                noise_total_crop = ie_crop + il_crop + tl_crop
            else:
                noise_total_crop = np.zeros_like(ref_crop)

            # Stack target and noise as separate reference sources: shape (2, samples)
            speech_source = np.stack([ref_crop, noise_total_crop], axis=0)

            # Stack the degraded signal and a low-power dummy noise floor to match shapes: shape (2, samples)
            # The dummy only satisfies the K_target shape constraint of the facade; it
            # does NOT influence any channel-[0] metric. It is drawn from a LOCAL seeded
            # RNG (not np.random) so the result is deterministic and does not perturb the
            # global numpy RNG state used elsewhere in the benchmark.
            dummy_noise_floor = np.random.default_rng(0).standard_normal(len(deg_crop)) * 1e-10
            speech_prediction = np.stack([deg_crop, dummy_noise_floor], axis=0)

            # Facade only for PESQ/STOI/SI-SDR (all lazy; the slow mir_eval bss_eval
            # is never triggered because we no longer read mir_eval_* from it).
            metrics_facade = OutputMetrics(
                speech_source=speech_source,
                speech_prediction=speech_prediction,
                sample_rate=fs,
                enable_si_sdr=True,
                compute_permutation=False
            )

            # Extract index 0 metrics which correspond strictly to the target speech channel.
            # Each metric is isolated so a PESQ 'NoUtterances' failure does not discard the rest.
            # PESQ SOLO del canal 0. `metrics_facade.pesq` calcula PESQ de TODOS
            # los canales, y el canal 1 es [ruido_total vs relleno ~silencio]:
            # con interferencias tipo ruido el VAD de PESQ no encuentra habla en
            # ese par, tira NoUtterancesError, y _safe_metric perdia TAMBIEN el
            # PESQ del objetivo (canal 0), que era valido. Los NaN dependian del
            # tipo de interferente -> perdida de datos sesgada. Con la llamada de
            # un solo canal los valores que ya salian no cambian (el canal 0 no
            # depende del 1); solo se recuperan los que se perdian.
            results['PESQ'] = _safe_metric(lambda: _pesq_single(ref_crop, deg_crop, fs))
            results['STOI'] = _safe_metric(lambda: metrics_facade.stoi[0])
            results['SI-SDR'] = _safe_metric(lambda: metrics_facade.si_sdr[0])

            # SDR/SIR/SAR via fast_bss_eval (mir_eval-equivalent, ~3x faster). The three
            # come from a single call, so on failure they all fall back to NaN together.
            try:
                sdr, sir, sar = _fast_sdr_sir_sar(speech_source, speech_prediction)
            except Exception:
                sdr = sir = sar = np.nan
            # BSS-Eval degenera a +/-inf cuando la interferencia (SIR) o los artefactos
            # (SAR) se anulan exactamente -- casos casi perfectos a RT bajo. inf rompe
            # las agregaciones (media/Delta), asi que se colapsan a NaN en la fuente.
            results['SDR'] = sdr if np.isfinite(sdr) else np.nan
            results['SIR'] = sir if np.isfinite(sir) else np.nan
            results['SAR'] = sar if np.isfinite(sar) else np.nan

        else:
            # Fallback to single-source evaluation if secondary components are missing
            metrics_facade = OutputMetrics(
                speech_source=ref_crop[np.newaxis, :],
                speech_prediction=deg_crop[np.newaxis, :],
                sample_rate=fs,
                enable_si_sdr=True,
                compute_permutation=False
            )

            results['PESQ'] = _safe_metric(lambda: metrics_facade.pesq[0])
            results['STOI'] = _safe_metric(lambda: metrics_facade.stoi[0])
            results['SI-SDR'] = _safe_metric(lambda: metrics_facade.si_sdr[0])

            # Single-source: fast_bss_eval returns SIR=inf (no interference), which we
            # drop to NaN to keep the historical contract. SDR/SAR are still valid.
            try:
                sdr, _sir_inf, sar = _fast_sdr_sir_sar(ref_crop[np.newaxis, :], deg_crop[np.newaxis, :])
            except Exception:
                sdr = sar = np.nan
            results['SDR'] = sdr if np.isfinite(sdr) else np.nan
            results['SAR'] = sar if np.isfinite(sar) else np.nan
            results['SIR'] = np.nan

    except Exception:
        for key in ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']:
            results[key] = np.nan

    # Fill deprecated keys with NaN to prevent benchmark DataFrame from shifting
    results['SINR'] = np.nan

    return results