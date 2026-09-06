"""
DINAMICA DEL LAZO DE REALIMENTACION DE NM_MVDR_DSM_BLIND (cadena causal).

LA HIPOTESIS QUE SE PONE A PRUEBA
---------------------------------
Cuando el archivo ARRANCA CON RUIDO y la voz entra tarde, Phi_SS = Phi_XX - Phi_NN
se estima con frames que NO contienen voz. La conjetura es que la direccion
dominante que queda ahi es la del INTERFERENTE, con lo cual el front-end ciego
apunta AL RUIDO justo cuando menos conviene, le empeora el SNR de entrada al
DTLN, y el sistema tarda en recuperarse cuando la voz finalmente aparece.

Hay una hipotesis RIVAL, que es la que dice el diseno (ver el docstring de
`estimate_rtf_recursive`): en un tramo sin voz, Phi_XX y Phi_NN se estiman sobre
LOS MISMOS frames, asi que con ruido ESTACIONARIO ambas convergen a la misma
matriz, Phi_SS -> 0, la carga `rtf_loading` domina la columna de referencia y
d -> e_ref. O sea: el front-end no apunta a ningun lado, degrada al canal crudo
y el sistema se porta como el de hoy. Nada de apuntar al interferente.

Las dos no pueden ser ciertas a la vez, y la diferencia es MEDIBLE:

    prefijo sin voz  ->  cos2(d, d_interf) alto   => H_lock   (apunta al ruido)
                     ->  cos2(d, e_ref)   alto    => H_shrink (degrada a e_ref)

La prediccion afinada es que quien decide es la ESTACIONARIEDAD del interferente:
si es estacionario, Phi_XX - Phi_NN se cancela y gana H_shrink; si es GATEADO
(el techno del benchmark lo es), la mascara fuga preferentemente en los ataques,
Phi_XX queda pesada hacia los frames de interferente fuerte y Phi_NN hacia los
flojos, la resta NO se cancela y su direccion dominante es la del interferente:
gana H_lock. Por eso el banco corre las dos senales de interferencia.

QUE SE MIDE, FRAME A FRAME
--------------------------
    cos2_tgt / cos2_int / cos2_ref : hacia donde apunta la RTF estimada d(k,t),
        contra la RTF ORACLE del target, la del interferente, y e_ref. Suman
        informacion, no probabilidad: son tres proyecciones normalizadas.
    snr_gain_hyp_db : lo unico que le importa al DTLN, pero medido con una
        SONDA HIPOTETICA: el W del frame t aplicado al target COMPLETO (sin la
        compuerta de onset), contra el ruido real. Contesta "si la voz estuviera
        sonando AHORA, ¿cuanto ganaria el front-end?". Hace falta porque durante
        el prefijo el SNR real es 0/0 y no dice nada: es la unica manera de
        medir el APUNTAMIENTO en dB mientras la fuente esta apagada -- justo la
        pregunta de si el filtro fijo deberia seguir apuntando a la fuente.
        `snr_gain_db` es el SNR real (solo interpretable con voz presente).
        `snr_gain_rtfor_db` es la MISMA arquitectura (DS) apuntada con la RTF
        oracle desde el frame 0: no es una cota superior, es el mismo sistema
        sin transitorio, asi que la brecha ES el costo de la convergencia.
    sig_ratio_db : tr(Phi_SS)/tr(Phi_NN), el SNR a-posteriori del propio
        estimador. Es la senal de confianza candidata para gatear el lazo, y
        sale gratis porque la recursion ya la calcula. `sig_ratio_neg` es la
        fraccion de bins donde la traza da NEGATIVA: ahi la sustraccion no
        estima nada, y lo que sobrevive a la proyeccion PSD es residuo puro.
    m_corr2 / m_corr1 : correlacion por frame de la mascara (pasada 2 y pasada
        1) contra la IRM oracle del canal de referencia. Cierra el lazo:
        muestra si el error de apuntamiento se traduce en peor mascara.

POR QUE EL BENCHMARK NO PODIA VER ESTO
--------------------------------------
run_mird_grid_search descarta los primeros eval_start_s = min(5, 0.3*duration)
segundos antes de medir. Todo el transitorio de convergencia cae DENTRO de esa
ventana descartada. Este banco mide justamente ahi.

USO
---
    python tests/window_mismatch/dsm_blind_feedback_diag.py
    python tests/window_mismatch/dsm_blind_feedback_diag.py --quick
    python tests/window_mismatch/dsm_blind_feedback_diag.py \
        --onsets 0 4 8 --interf gated pink --rtf-alpha 0.999 0.99

Salida en tests/dataset_out/dsm_blind_feedback/:
    feedback_frames.csv   una fila por (condicion, frame)
    feedback_summary.csv  una fila por condicion (prefijo, convergencia, meseta)
    feedback_diag.png     panel
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import scipy.signal as sig

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from propagation.mird_loader import MirdDatasetProvider                      # noqa: E402
from beamforming.mask.dtln_masks import (get_dtln_masks_sharpen,             # noqa: E402
                                         get_dtln_masks_soft,
                                         align_mask_frames)
from beamforming.mask.ds_mask import estimate_rtf_recursive                  # noqa: E402
from evaluation.bf_wrappers import resolve_stft_window                       # noqa: E402
from lowfreq_diagnostic_run import build_scene                               # noqa: E402

OUT_DIR = os.path.join(ROOT, "tests", "dataset_out", "dsm_blind_feedback")
MODEL_1 = os.path.join(ROOT, "src/dnn_denoise/models/model_quant_1.tflite")
SIGNALS = os.path.join(ROOT, "tools/data/signals")

# Las TRES PIEZAS de arranque en frio, cada una AISLADA (ver `estimate_rtf_recursive`).
# Cada variante es un dict de kwargs que se le pasa al estimador; "base" = el
# sistema de hoy, y es el control contra el que se comparan las demas.
# La sombra que mide la confianza es un DETECTOR: con el alpha nominal (8 s)
# hereda el mismo anclaje que hay que detectar, asi que las variantes con gate
# la corren corta (conf_alpha=0.99, ~0.8 s).
_CA = 0.99
VARIANTS = {
    "base":      {},
    "gate":      {"conf_gate": 0.35, "conf_alpha": _CA},
    # ablaciones del umbral y de la ventana de la sombra
    "gate_th20": {"conf_gate": 0.20, "conf_alpha": _CA},
    "gate_th50": {"conf_gate": 0.50, "conf_alpha": _CA},
    "gate_slow": {"conf_gate": 0.35},
}

INTERF_WAVS = {
    "gated": os.path.join(SIGNALS, "techno_gated commune.wav"),
    "pink": os.path.join(SIGNALS, "ruido_rosa_16k.wav"),
}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def fit_T(m, T):
    """Mismo criterio que blind_bf_signal: repite el ultimo frame, no recorta."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape[1] >= T:
        return m[:, :T]
    return np.concatenate([m, np.repeat(m[:, -1:], T - m.shape[1], axis=1)], axis=1)


def onset_gate(n_samples, fs, t_on, fade_ms=50.0):
    """Compuerta 0->1 en t_on con rampa coseno (evita el click del corte)."""
    g = np.ones(n_samples)
    if t_on <= 0:
        return g
    n_on = min(int(t_on * fs), n_samples)
    n_f = max(int(fade_ms * 1e-3 * fs), 1)
    g[:n_on] = 0.0
    n_f = min(n_f, n_samples - n_on)
    if n_f > 0:
        g[n_on:n_on + n_f] = 0.5 * (1 - np.cos(np.pi * np.arange(n_f) / n_f))
    return g


def oracle_rtf(X, ref, frac=0.5):
    """
    RTF oracle por bin: autovector principal de la SCM de la componente LIMPIA,
    normalizado a d[ref] = 1. Se promedia solo sobre el `frac` de frames de mas
    energia del bin, para que los tramos en silencio (que solo aportan el ruido
    termico) no ensucien la direccion.

    X: (K, T, M) STFT de la componente aislada.  Devuelve (K, M).
    """
    K, T, M = X.shape
    pw = np.sum(np.abs(X) ** 2, axis=2)                       # (K, T)
    n_keep = max(int(T * frac), 1)
    idx = np.argsort(-pw, axis=1)[:, :n_keep]                 # (K, n_keep)
    Xs = np.take_along_axis(X, idx[:, :, None], axis=1)       # (K, n_keep, M)
    R = np.einsum("ktm,ktn->kmn", Xs, Xs.conj()) / n_keep
    _, V = np.linalg.eigh(0.5 * (R + np.conj(np.transpose(R, (0, 2, 1)))))
    v = V[:, :, -1]
    return v / (v[:, ref][:, None] + 1e-30)


def cos2(D, d_ref):
    """|d^H d_ref|^2 / (||d||^2 ||d_ref||^2) por (K, T).  D:(K,T,M) d_ref:(K,M)"""
    num = np.abs(np.einsum("ktm,km->kt", D.conj(), d_ref)) ** 2
    den = (np.sum(np.abs(D) ** 2, axis=2) *
           np.sum(np.abs(d_ref) ** 2, axis=1)[:, None])
    return num / (den + 1e-30)


def band_mean(A, freqs, f_lo, f_hi):
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    return np.mean(A[sel], axis=0)


def band_snr_db(S, N, freqs, f_lo, f_hi):
    """SNR por frame integrando la potencia en la banda. S, N: (K, T)."""
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    return 10 * np.log10((np.sum(S[sel], axis=0) + 1e-30) /
                         (np.sum(N[sel], axis=0) + 1e-30))


def frame_corr(A, B):
    """Correlacion de Pearson por frame (columna) entre dos (K, T)."""
    a = A - A.mean(axis=0, keepdims=True)
    b = B - B.mean(axis=0, keepdims=True)
    return (np.sum(a * b, axis=0) /
            (np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0)) + 1e-30))


def smooth_frames(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


# ---------------------------------------------------------------------------
# una condicion
# ---------------------------------------------------------------------------

def run_condition(cfg, mix_full, o_tgt, o_noi, ref, args, t_on, tag,
                  rtf_alpha=None, rtf_loading=None, variant="base"):
    fs = cfg["fs"]
    L, nov = cfg["stft_window"], cfg["stft_overlap"]
    hop = L - nov
    win = resolve_stft_window(cfg, args.win, L)
    rtf_alpha = args.rtf_alpha[0] if rtf_alpha is None else rtf_alpha
    rtf_loading = args.rtf_loading[0] if rtf_loading is None else rtf_loading
    vkw = dict(VARIANTS[variant])
    rtf_alpha = vkw.pop("rtf_alpha", rtf_alpha)
    rtf_loading = vkw.pop("rtf_loading", rtf_loading)

    # --- escena con la voz entrando en t_on --------------------------------
    g = onset_gate(mix_full.shape[1], fs, t_on)
    tgt = o_tgt * g[None, :]
    mix = tgt + o_noi

    # --- PASADA 1: la mascara del canal crudo (cadena CAUSAL) --------------
    m_s, m_n = get_dtln_masks_sharpen(
        mix, ref, MODEL_1, block_len=L, block_shift=hop,
        sharpen_exp=args.sharpen_exp, peak_norm=1.0, stretch=False)
    m_s, m_n = align_mask_frames((m_s, m_n), 1)

    # --- STFTs (mezcla y componentes oracle, mismo dominio) ---------------
    def _stft(x):
        f, _, Z = sig.stft(x, fs=fs, window=win, nperseg=L, noverlap=nov, nfft=L)
        return f, np.transpose(Z, (1, 2, 0))                  # (K, T, M)

    freqs, X = _stft(mix)
    _, X_t = _stft(tgt)              # target REAL (apagado antes de t_on)
    _, X_tf = _stft(o_tgt)           # target COMPLETO: la sonda hipotetica
    _, X_n = _stft(o_noi)
    K, T, M = X.shape
    m_s, m_n = fit_T(m_s, T), fit_T(m_n, T)
    m1_raw = np.clip(m_s ** (1.0 / args.sharpen_exp), 0.0, 1.0)

    # --- referencias oracle -------------------------------------------------
    d_tgt = oracle_rtf(X_tf, ref)
    d_int = oracle_rtf(X_n, ref)
    e_ref = np.zeros((K, M), dtype=np.complex128)
    e_ref[:, ref] = 1.0

    # Misma familia de front-end (DS) apuntado con la RTF oracle desde el frame
    # 0: NO es una cota superior, es la MISMA arquitectura sin transitorio. La
    # brecha contra ella es exactamente el costo de la convergencia.
    W_or = d_tgt / (np.sum(np.abs(d_tgt) ** 2, axis=1)[:, None] + 1e-30)

    S_rf, N_rf = np.abs(X_t[:, :, ref]) ** 2, np.abs(X_n[:, :, ref]) ** 2
    S_rff = np.abs(X_tf[:, :, ref]) ** 2
    irm = S_rf / (S_rf + N_rf + 1e-30)
    lo, hi = args.band
    times = np.arange(T) * hop / fs
    snr_ref = band_snr_db(S_rf, N_rf, freqs, lo, hi)
    snr_ref_h = band_snr_db(S_rff, N_rf, freqs, lo, hi)
    S_or = np.abs(np.einsum("km,ktm->kt", W_or.conj(), X_tf)) ** 2
    N_or = np.abs(np.einsum("km,ktm->kt", W_or.conj(), X_n)) ** 2
    gain_or_h = band_snr_db(S_or, N_or, freqs, lo, hi) - snr_ref_h
    # Frames con voz REAL presente (para no promediar sobre las pausas).
    tgt_pw = band_mean(S_rf, freqs, lo, hi)
    active = tgt_pw > (np.max(tgt_pw) * 1e-4)

    out = []
    # --- LAZO: se repite n_iter veces para ver si la realimentacion SUMA ----
    for it in range(1, max(args.n_iter, 1) + 1):
        W, D, diag = estimate_rtf_recursive(
            X, m_s, m_n, ref_mic_idx=ref, rtf_alpha=rtf_alpha,
            rtf_loading=rtf_loading, rtf_mode=args.rtf_mode, w_mode="ds",
            bf_loading=1e-6, return_diag=True,
            conf_bins=(freqs >= args.conf_band[0]) & (freqs <= args.conf_band[1]),
            **vkw)

        Y = np.einsum("ktm,ktm->kt", W.conj(), X)
        _, y_fix = sig.istft(Y, fs=fs, window=win, nperseg=L, noverlap=nov, nfft=L)
        y_fix = y_fix[:mix.shape[1]]

        # --- PASADA 2: la mascara que realmente alimenta al beamformer -----
        m_raw, _ = get_dtln_masks_soft(y_fix[None, :], 0, MODEL_1, block_len=L,
                                       block_shift=hop, peak_norm=1.0)
        m_raw = fit_T(align_mask_frames(m_raw, 1), T)

        c_tgt, c_int, c_ref = cos2(D, d_tgt), cos2(D, d_int), cos2(D, e_ref)
        S_out = np.abs(np.einsum("ktm,ktm->kt", W.conj(), X_t)) ** 2
        N_out = np.abs(np.einsum("ktm,ktm->kt", W.conj(), X_n)) ** 2
        # Sonda HIPOTETICA: el mismo W aplicado al target COMPLETO. Contesta
        # "si la voz estuviera sonando AHORA, ¿cuanto ganaria el front-end?",
        # que es la unica forma de medir el apuntamiento en dB durante el
        # prefijo sin voz (donde el SNR real es 0/0).
        S_hyp = np.abs(np.einsum("ktm,ktm->kt", W.conj(), X_tf)) ** 2

        df = pd.DataFrame({
            "tag": tag, "t_on": t_on, "iter": it, "variant": variant,
            "rtf_alpha": rtf_alpha, "rtf_loading": rtf_loading,
            "frame": np.arange(T), "time_s": times,
            "cos2_tgt": band_mean(c_tgt, freqs, lo, hi),
            "cos2_int": band_mean(c_int, freqs, lo, hi),
            "cos2_ref": band_mean(c_ref, freqs, lo, hi),
            "snr_out_db": band_snr_db(S_out, N_out, freqs, lo, hi),
            "snr_ref_db": snr_ref,
            "sig_ratio_db": 10 * np.log10(
                np.maximum(band_mean(diag["sig_ratio"], freqs, lo, hi), 1e-12)),
            "sig_ratio_neg": band_mean((diag["sig_ratio"] <= 0).astype(float),
                                       freqs, lo, hi),
            "den_s": band_mean(diag["den_s"], freqs, lo, hi),
            "load_ratio_db": 10 * np.log10(
                np.maximum(band_mean(diag["load_ratio"], freqs, lo, hi), 1e-12)),
            "conf": band_mean(diag["conf"], freqs, lo, hi),
            "gate": band_mean(diag["gate"], freqs, lo, hi),
            "m_corr1": frame_corr(m1_raw, irm),
            "m_corr2": frame_corr(m_raw, irm),
            "tgt_active": active,
            "snr_gain_hyp_db": band_snr_db(S_hyp, N_out, freqs, lo, hi) - snr_ref_h,
            "snr_gain_rtfor_db": gain_or_h,
        })
        df["snr_gain_db"] = df["snr_out_db"] - df["snr_ref_db"]
        out.append(df)

        # realimentacion: la mascara de la pasada 2 pasa a estimar la RTF
        m = np.clip(m_raw, 0.0, 1.0)
        m_s, m_n = m ** args.sharpen_exp, (1.0 - m) ** args.sharpen_exp

    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# resumen
# ---------------------------------------------------------------------------

def sustained_time(t, x, thr, hold_s=1.0):
    """Primer instante desde el que `x` se queda >= thr al menos `hold_s`."""
    if len(t) < 2:
        return np.nan
    n_hold = max(int(hold_s / (t[1] - t[0])), 1)
    ok = x >= thr
    for i in range(len(t) - n_hold):
        if ok[i:i + n_hold].all():
            return t[i]
    return np.nan


def summarize(df, fs, hop, plateau_s=3.0, cos_thr=0.9):
    """
    Prefijo sin voz, apuntamiento en meseta y tiempo de convergencia.

    La convergencia se mide sobre `cos2_tgt`, NO sobre el SNR: la ganancia por
    frame la modula el contenido espectral de la voz (en la escena convergida
    oscila entre -1 y +2.5 dB de un segmento a otro), asi que cualquier umbral
    sobre ella mide el material, no el estimador. cos2_tgt es monotona y llega a
    ~0.98 en todas las condiciones que convergen, asi que un umbral ABSOLUTO de
    0.9 sostenido 1 s es comparable entre condiciones sin normalizar por nada.

    `den_s_at_onset` es la variable explicativa: la masa de mascara YA acumulada
    en el estimador cuando entra la voz. Es el ancla que la evidencia nueva tiene
    que vencer, y el horizonte para vencerla es 1/(1-rtf_alpha) frames.
    """
    rows = []
    n_sm = max(int(0.5 * fs / hop), 1)
    for (tag, t_on, it, var, a_, ld_), g in df.groupby(["tag", "t_on", "iter", "variant", "rtf_alpha",
                                 "rtf_loading"], sort=False):
        g = g.sort_values("frame")
        t = g["time_s"].to_numpy()
        pre = (t > 0.5) & (t < max(t_on - 0.2, 0.0))          # prefijo sin voz
        tail = t >= (t.max() - plateau_s)
        c_tgt = g["cos2_tgt"].to_numpy()
        gain = smooth_frames(g["snr_gain_hyp_db"].to_numpy(), n_sm)

        post = t >= t_on
        t_conv = sustained_time(t[post], c_tgt[post], cos_thr)
        t_conv = np.nan if np.isnan(t_conv) else t_conv - t_on
        i_on = int(np.argmax(post))

        rows.append({
            "tag": tag, "t_on": t_on, "iter": it, "variant": var,
            "alpha": a_, "load": ld_,
            "pre_cos2_tgt": float(np.mean(c_tgt[pre])) if pre.any() else np.nan,
            "pre_cos2_int": float(np.mean(g["cos2_int"].to_numpy()[pre])) if pre.any() else np.nan,
            "pre_cos2_ref": float(np.mean(g["cos2_ref"].to_numpy()[pre])) if pre.any() else np.nan,
            "pre_gain_hyp_db": float(np.mean(gain[pre])) if pre.any() else np.nan,
            "pre_signeg": float(np.mean(g["sig_ratio_neg"].to_numpy()[pre])) if pre.any() else np.nan,
            "den_s_at_onset": float(g["den_s"].to_numpy()[i_on]),
            "pre_conf": float(np.mean(g["conf"].to_numpy()[pre])) if pre.any() else np.nan,
            "pre_gate": float(np.mean(g["gate"].to_numpy()[pre])) if pre.any() else np.nan,
            "post_gate": float(np.mean(g["gate"].to_numpy()[t >= t_on])),
            "t_conv_s": t_conv,
            "cos2_tgt_end": float(np.mean(c_tgt[tail])),
            "plateau_gain_db": float(np.median(gain[tail])),
            "rtf_oracle_gain_db": float(np.median(g["snr_gain_rtfor_db"].to_numpy()[tail])),
            "m_corr1_tail": float(np.mean(g["m_corr1"].to_numpy()[tail])),
            "m_corr2_tail": float(np.mean(g["m_corr2"].to_numpy()[tail])),
        })
    return pd.DataFrame(rows)


def plot_panel(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = df["iter"].max()
    df = df[df["iter"] == it]
    conds = list(dict.fromkeys(zip(df["tag"], df["t_on"], df["variant"],
                                   df["rtf_alpha"], df["rtf_loading"])))
    n = len(conds)
    fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 11.5), sharex=True,
                             squeeze=False)
    for j, (tag, t_on, var, a_, ld_) in enumerate(conds):
        g = df[(df.tag == tag) & (df.t_on == t_on) & (df.variant == var) &
               (df.rtf_alpha == a_) & (df.rtf_loading == ld_)].sort_values("frame")
        t = g["time_s"]

        ax = axes[0][j]
        ax.plot(t, g["cos2_tgt"], label=r"$\cos^2(\hat d, d_{tgt})$", lw=1.1)
        ax.plot(t, g["cos2_int"], label=r"$\cos^2(\hat d, d_{int})$", lw=1.1)
        ax.plot(t, g["cos2_ref"], label=r"$\cos^2(\hat d, e_{ref})$", lw=1.1)
        ax.set_ylim(-0.02, 1.02); ax.set_title(f"{tag}  voz en t={t_on}s\n"
                     rf"{var}   $\alpha$={a_:g}  load={ld_:g}", fontsize=9)
        ax.set_ylabel("apuntamiento")
        if j == 0:
            ax.legend(fontsize=7, loc="center right")

        # 0.5 s de promedio: la ganancia por frame la domina el contenido
        # espectral de la voz y sin suavizar tapa la tendencia.
        n_sm = max(int(0.5 / max(t.iloc[1] - t.iloc[0], 1e-9)), 1)
        ax = axes[1][j]
        ax.plot(t, smooth_frames(g["snr_gain_hyp_db"].to_numpy(), n_sm), lw=1.1,
                label="ciego (sonda hipotetica)")
        ax.plot(t, smooth_frames(g["snr_gain_rtfor_db"].to_numpy(), n_sm), lw=1.1,
                ls="--", label="RTF oracle")
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.set_ylabel("ganancia de SNR del front-end [dB]")
        if j == 0:
            ax.legend(fontsize=7)

        ax = axes[2][j]
        ax.plot(t, g["sig_ratio_db"], lw=1.1, label=r"tr$\Phi_{SS}$/tr$\Phi_{NN}$")
        ax.plot(t, g["den_s"], lw=1.1, label=r"masa acumulada $\Sigma\alpha^k m_s$")
        ax.plot(t, 10.0 * g["gate"], lw=1.4, alpha=0.6, label="gate x10")
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.set_ylim(-25, 40)
        ax.set_ylabel("confianza [dB]  /  masa")
        if j == 0:
            ax.legend(fontsize=7)

        ax = axes[3][j]
        ax.plot(t, smooth_frames(g["m_corr1"].to_numpy(), n_sm), lw=1.1,
                label="mascara pasada 1 (x_ref)")
        ax.plot(t, smooth_frames(g["m_corr2"].to_numpy(), n_sm), lw=1.1,
                label="mascara pasada 2 (y_fix)")
        ax.set_ylabel("corr con IRM oracle"); ax.set_xlabel("tiempo [s]")
        if j == 0:
            ax.legend(fontsize=7)

        for ax in axes[:, j]:
            if t_on > 0:
                ax.axvline(t_on, color="crimson", lw=1.0, alpha=0.6)
            ax.grid(alpha=0.25)
    fig.suptitle("NM_MVDR_DSM_BLIND causal: dinamica del lazo de la mascara", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[*] {path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt60", type=float, default=0.610)
    ap.add_argument("--spacing", default="3-3-3-8-3-3-3")
    ap.add_argument("--target-angle", type=float, default=0)
    ap.add_argument("--target-dist", type=float, default=1.0)
    ap.add_argument("--interf-angle", type=float, default=45)
    ap.add_argument("--interf-dist", type=float, default=1.0)
    ap.add_argument("--isir", type=float, default=0.0)
    ap.add_argument("--snr-db", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--onsets", type=float, nargs="+", default=[0.0, 4.0, 8.0],
                    help="instantes de entrada de la voz [s]")
    ap.add_argument("--interf", nargs="+", default=["gated", "pink"],
                    choices=sorted(INTERF_WAVS), help="tipo de interferente")
    ap.add_argument("--rtf-alpha", type=float, nargs="+", default=[0.999],
                    help="factor de olvido del estimador de RTF (acepta varios: se barre)")
    ap.add_argument("--rtf-loading", type=float, nargs="+", default=[1e-2],
                    help="carga relativa al ruido del estimador (acepta varios: se barre)")
    ap.add_argument("--rtf-mode", default="cs", choices=["cs", "evd", "cw"])
    ap.add_argument("--n-iter", type=int, default=1,
                    help="vueltas del lazo mascara->RTF->mascara (n_iter del wrapper). "
                         "2+ permite ver si la realimentacion suma o deriva.")
    ap.add_argument("--sharpen-exp", type=float, default=8.0)
    ap.add_argument("--win", default="rect", help="ventana de analisis (cadena causal: rect)")
    ap.add_argument("--band", type=float, nargs=2, default=[300.0, 3400.0],
                    help="banda donde se integran SNR y apuntamiento [Hz]")
    ap.add_argument("--variants", nargs="+", default=["base"],
                    choices=sorted(VARIANTS),
                    help="piezas de arranque en frio a comparar (aisladas)")
    ap.add_argument("--conf-band", type=float, nargs=2, default=[300.0, 3400.0],
                    help="banda donde se cuenta la fraccion de bins con senal [Hz]")
    ap.add_argument("--quick", action="store_true",
                    help="una sola condicion por interferente (onset 4 s), 12 s")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.quick:
        args.onsets, args.duration = [4.0], 12.0

    os.makedirs(args.out_dir, exist_ok=True)
    provider = MirdDatasetProvider(root_dir=os.path.join(ROOT, "tools/data/rirs/mird"))

    parts = []
    for kind in args.interf:
        cfg = {
            'fs': 16000, 'duration': args.duration, 't_early': 0.050,
            'array_center': [3.0, 3.0, 1.2], 'mird_spacing': args.spacing,
            'snr_db': args.snr_db,
            'source_path': os.path.join(SIGNALS, "p002_emo_adoration_sentences.wav"),
            'interf_paths': [INTERF_WAVS[kind]],
            'stft_window': 512, 'stft_overlap': 384,
            'dtln_model_path': MODEL_1,
        }
        print(f"\n[*] escena MIRD  interf={kind}  rt60={args.rt60}  iSIR={args.isir} dB")
        _, mix_full, o_tgt, o_noi, _ = build_scene(
            cfg, provider, args.rt60, args.target_angle, args.target_dist,
            [(args.interf_angle, args.interf_dist)], args.isir, args.snr_db)
        ref = mix_full.shape[0] // 2

        for t_on in args.onsets:
            for a in args.rtf_alpha:
                for ld in args.rtf_loading:
                    for var in args.variants:
                        print(f"    - onset {t_on:.1f} s  {var}  alpha={a:g} "
                              f"load={ld:g} ...", flush=True)
                        parts.append(run_condition(
                            cfg, mix_full, o_tgt, o_noi, ref, args, float(t_on),
                            kind, rtf_alpha=a, rtf_loading=ld, variant=var))

    df = pd.concat(parts, ignore_index=True)
    hop = 512 - 384
    summ = summarize(df, 16000, hop)

    f_frames = os.path.join(args.out_dir, "feedback_frames.csv")
    f_summ = os.path.join(args.out_dir, "feedback_summary.csv")
    df.to_csv(f_frames, index=False)
    summ.to_csv(f_summ, index=False)

    pd.set_option("display.width", 230)
    print("\n=== PREFIJO SIN VOZ: ¿a donde apunta la RTF ciega? ===")
    print("   (cos2_int >> cos2_ref  => H_lock;  cos2_ref alto => H_shrink)")
    print(summ[["tag", "t_on", "variant", "pre_cos2_tgt", "pre_cos2_int",
                "pre_cos2_ref", "pre_gain_hyp_db", "pre_conf",
                "pre_gate"]].round(3).to_string(index=False))
    print("\n=== CONVERGENCIA DESPUES DE LA ENTRADA DE VOZ ===")
    print("   (t_conv_s = cos2_tgt >= 0.9 sostenido 1 s, medido desde t_on)")
    print(summ[["tag", "t_on", "variant", "den_s_at_onset", "post_gate",
                "t_conv_s", "cos2_tgt_end", "plateau_gain_db",
                "rtf_oracle_gain_db", "m_corr2_tail"]].round(3).to_string(index=False))
    print(f"\n[*] {f_frames}\n[*] {f_summ}")
    plot_panel(df, os.path.join(args.out_dir, "feedback_diag.png"))
    return df, summ


if __name__ == "__main__":
    main()
