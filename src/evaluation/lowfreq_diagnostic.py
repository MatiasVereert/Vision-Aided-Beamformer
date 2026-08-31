"""
lowfreq_diagnostic.py
=====================
Diagnostico NARROWBAND (por bin de frecuencia) de un beamformer de la familia
Souden. Responde una sola pregunta:

    la perdida a baja frecuencia, ¿es un limite FISICO de apertura, o es error
    de ESTIMACION de las covarianzas (y en particular auto-cancelacion del
    target por fuga de voz dentro de Phi_NN)?

MOTIVACION
----------
PESQ (P.862) filtra por debajo de ~300 Hz: la banda donde sospechamos el
problema es justo la que la metrica principal del benchmark NO ve. Todo lo que
hay aca se mide directamente sobre la senal, sin PESQ/STOI, para que la banda
grave sea observable.

IDEA CENTRAL (exacta, no aproximada)
------------------------------------
El beamformer es LINEAL por frame: Y = w^H x. Si tenemos las componentes oracle
multicanal (target-solo y ruido-solo, en el MISMO dominio que la mezcla que se
filtro), entonces

    Y_S(k,t) = w(k,t)^H s(k,t)      (target a la salida)
    Y_N(k,t) = w(k,t)^H n(k,t)      (ruido  a la salida)

y  Y = Y_S + Y_N  EXACTAMENTE. No hay que estimar nada: se aplican los pesos
guardados (save_weights=True) a cada componente por separado. De ahi salen, por
bin:

    TR(k) = P_S_out / P_S_in        respuesta al target ("¿me como la voz?")
    NR(k) = P_N_in  / P_N_out       reduccion de ruido
    AG(k) = TR * NR                 ganancia de arreglo (SNR_out / SNR_in)

La descomposicion AG = TR * NR es el punto: si a 200 Hz hay AG alta pero TR muy
por debajo de 0 dB, el beamformer NO esta separando, esta apagando la banda
entera (y PESQ, ciega ahi, ni se entera).

QUE MAS SE MIDE
---------------
  - SD_coh : indice de distorsion invariante a escala, 1 - |coh(Y_S, S_ref)|^2.
             Comparable entre variantes con normalizacion distinta (p.ej. BAN).
  - WNG    : white noise gain = 1 / ||w||^2. Cuanto amplifica el filtro el ruido
             propio de los microfonos. Muy negativo = regimen superdirectivo,
             la solucion vive en los autovectores que no se pueden estimar.
  - leak_NN: fraccion de la energia acumulada en Phi_NN que era TARGET. Es el
             test directo de auto-cancelacion (self-nulling): si a baja
             frecuencia la mascara de ruido deja pasar voz, Phi_NN^-1 la nulea.
  - cont_XX: dual del anterior, ruido contaminando Phi_XX.
  - escalera de autovalores de Phi_NN (verdadera y estimada), rango efectivo y
             numero de condicion por bin.
  - DOF teorico del arreglo, 1 + 2L/lambda, y f_c = c/(2L).

COMO SE LEE
-----------
Corriendo el MISMO diagnostico sobre el beamformer mask-based y sobre el oracle
(SOUDEN_ORACLE_SCM, covarianzas de las senales limpias) se separan las dos
hipotesis:

  oracle tambien cae en grave  -> limite de APERTURA. Ningun cambio de algoritmo
                                  lo arregla; abajo de f_c la banda le toca al
                                  post-filtro espectral.
  oracle sano, mask cae        -> ESTIMACION. Ahi valen los levers: gating del
                                  update de Phi_NN, carga diagonal por bin, etc.
                                  Mirar leak_NN para confirmar self-nulling.

LIMITACION DE ALCANCE
---------------------
Los pesos describen SOLO la etapa espacial. Las variantes con post-filtro
(NM_MVDR_PF, *_specsub, *_MWF via ganancia sobre la salida) aplican una ganancia
real por bin DESPUES del filtro espacial, que no esta en `weights`. Para esas,
o se pasa la ganancia por `post_gain`, o se interpreta el reporte como
"diagnostico del beamformer subyacente" y nada mas.
"""

import numpy as np

C_SOUND = 343.0


# =====================================================================
# Utilidades
# =====================================================================
def _to_db(x, floor=1e-30):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor))


def align_frames(*arrays):
    """Trunca todos los arrays al minimo numero de frames (eje 1)."""
    T = min(a.shape[1] for a in arrays if a is not None)
    return tuple(None if a is None else a[:, :T, ...] for a in arrays)


def array_aperture(mic_coords):
    """Apertura maxima (distancia entre los dos micros mas lejanos), en metros."""
    P = np.asarray(mic_coords, dtype=np.float64)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return float(d.max())


def theoretical_dof(freqs, aperture, c=C_SOUND, M=None):
    """
    Grados de libertad espaciales de una apertura L: DOF(f) = 1 + 2L/lambda,
    saturados en M. Es el producto espacio-ancho de banda (la region visible del
    espectro de numero de onda tiene ancho 2k sobre una apertura L).

    DOF < 2 significa que el arreglo NO puede colocar un nulo a esa frecuencia,
    sin importar cuantos microfonos tenga adentro de la misma apertura.
    """
    dof = 1.0 + 2.0 * aperture * np.asarray(freqs, dtype=np.float64) / c
    return np.minimum(dof, M) if M is not None else dof


def critical_frequency(aperture, c=C_SOUND):
    """f_c = c / (2L): debajo de esto el arreglo tiene menos de 2 DOF."""
    return c / (2.0 * aperture)


def effective_rank(evals, eps=1e-30):
    """
    Rango efectivo por entropia (Roy & Vetterli): exp(-sum p log p) con
    p = lambda / sum(lambda). Version suave de "cuantos autovalores importan".
    Entrada (K, M) de autovalores >= 0. Salida (K,).
    """
    ev = np.maximum(np.asarray(evals, dtype=np.float64), 0.0)
    s = ev.sum(axis=-1, keepdims=True)
    p = ev / np.maximum(s, eps)
    ent = -np.sum(np.where(p > eps, p * np.log(np.maximum(p, eps)), 0.0), axis=-1)
    return np.exp(ent)


# =====================================================================
# 1. Reporte narrowband a partir de los pesos guardados
# =====================================================================
def narrowband_report(W, S_stft, N_stft, ref_mic_idx, start_frame=0,
                      post_gain=None):
    """
    Aplica los pesos guardados a las componentes oracle por separado.

    W          : (K, T, M) pesos del beamformer (save_weights=True). La salida se
                 forma como Y = w^H x, igual que en souden_mvdr.py.
    S_stft     : (K, T, M) target-solo, MISMO dominio y MISMA STFT que la mezcla.
    N_stft     : (K, T, M) ruido+interferencia-solo, idem.
    ref_mic_idx: canal de referencia (el mismo que proyecta el beamformer).
    start_frame: frames iniciales a descartar (warm-up recursivo + eval_start_s).
    post_gain  : (K, T) real opcional, ganancia del post-filtro aplicada a la
                 SALIDA. Si se pasa, se incluye en Y_S / Y_N.

    Devuelve un dict de arrays (K,) con potencias y ratios lineales (no dB).
    Las potencias se devuelven crudas para poder agregar por bandas SUMANDO
    potencias antes de dividir (promediar ratios por banda estaria mal).
    """
    W, S_stft, N_stft = align_frames(W, S_stft, N_stft)
    if post_gain is not None:
        post_gain = post_gain[:, :W.shape[1]]

    sl = slice(start_frame, None)

    # Salida por componente. Y = w^H x  (conjugado en w, igual que el core).
    Y_S = np.einsum("ktm,ktm->kt", W.conj(), S_stft)
    Y_N = np.einsum("ktm,ktm->kt", W.conj(), N_stft)
    if post_gain is not None:
        Y_S = Y_S * post_gain
        Y_N = Y_N * post_gain

    S_ref = S_stft[:, :, ref_mic_idx]
    N_ref = N_stft[:, :, ref_mic_idx]

    P_S_out = np.mean(np.abs(Y_S[:, sl]) ** 2, axis=1)
    P_N_out = np.mean(np.abs(Y_N[:, sl]) ** 2, axis=1)
    P_S_in = np.mean(np.abs(S_ref[:, sl]) ** 2, axis=1)
    P_N_in = np.mean(np.abs(N_ref[:, sl]) ** 2, axis=1)

    # Distorsion invariante a escala: 1 - |coherencia|^2 entre Y_S y S_ref.
    # Robusto a variantes con normalizacion distinta (BAN reescala w).
    num = np.abs(np.sum(Y_S[:, sl] * S_ref[:, sl].conj(), axis=1)) ** 2
    den = (np.sum(np.abs(Y_S[:, sl]) ** 2, axis=1)
           * np.sum(np.abs(S_ref[:, sl]) ** 2, axis=1))
    sd_coh = 1.0 - num / np.maximum(den, 1e-30)

    # Distorsion cruda (sin reescalar): penaliza tambien el error de ganancia.
    sd_raw = (np.mean(np.abs(Y_S[:, sl] - S_ref[:, sl]) ** 2, axis=1)
              / np.maximum(P_S_in, 1e-30))

    # White noise gain = 1 / ||w||^2, promediado en frames.
    w_norm2 = np.mean(np.sum(np.abs(W[:, sl, :]) ** 2, axis=2), axis=1)
    wng = 1.0 / np.maximum(w_norm2, 1e-30)

    return {
        "P_S_in": P_S_in, "P_N_in": P_N_in,
        "P_S_out": P_S_out, "P_N_out": P_N_out,
        # ratios lineales por bin
        "TR": P_S_out / np.maximum(P_S_in, 1e-30),      # respuesta al target
        "NR": P_N_in / np.maximum(P_N_out, 1e-30),      # reduccion de ruido
        "AG": ((P_S_out / np.maximum(P_N_out, 1e-30))
               / np.maximum(P_S_in / np.maximum(P_N_in, 1e-30), 1e-30)),
        "SNR_in": P_S_in / np.maximum(P_N_in, 1e-30),
        "SNR_out": P_S_out / np.maximum(P_N_out, 1e-30),
        "SD_coh": np.clip(sd_coh, 0.0, 1.0),
        "SD_raw": sd_raw,
        "WNG": wng,
        "w_norm2": w_norm2,
    }


# =====================================================================
# 2. Fuga de target dentro de Phi_NN (test de auto-cancelacion)
# =====================================================================
def mask_leakage_report(mask_s, mask_n, S_stft, N_stft, ref_mic_idx, start_frame=0):
    """
    Cuanto de lo que el beamformer METIO en cada covarianza era la componente
    equivocada. Se mide sobre el canal de referencia, con los MISMOS pesos de
    mascara que usan los acumuladores del core.

        leak_NN(k) = sum_t m_n |S_ref|^2 / sum_t m_n (|S_ref|^2 + |N_ref|^2)
        cont_XX(k) = sum_t m_s |N_ref|^2 / sum_t m_s (|S_ref|^2 + |N_ref|^2)

    leak_NN alto a baja frecuencia = la mascara de ruido deja pasar voz ->
    Phi_NN contiene la firma espacial del target -> Phi_NN^-1 la nulea. Ese es
    el mecanismo clasico de signal cancellation, y explica "se come la voz".

    NOTA: se usa el factor de olvido implicito de la acumulacion plana (todos
    los frames), no el alpha del core. Con fuentes estacionarias y alpha->1 son
    equivalentes; el objetivo aca es el BALANCE de energia, no la dinamica.
    """
    mask_s, mask_n, S_stft, N_stft = align_frames(
        mask_s[:, :, None], mask_n[:, :, None], S_stft, N_stft)
    mask_s = mask_s[:, :, 0]
    mask_n = mask_n[:, :, 0]
    sl = slice(start_frame, None)

    Ps = np.abs(S_stft[:, sl, ref_mic_idx]) ** 2
    Pn = np.abs(N_stft[:, sl, ref_mic_idx]) ** 2
    ms = mask_s[:, sl]
    mn = mask_n[:, sl]

    num_leak = np.sum(mn * Ps, axis=1)
    den_leak = np.sum(mn * (Ps + Pn), axis=1)
    num_cont = np.sum(ms * Pn, axis=1)
    den_cont = np.sum(ms * (Ps + Pn), axis=1)

    return {
        "leak_NN": num_leak / np.maximum(den_leak, 1e-30),
        "cont_XX": num_cont / np.maximum(den_cont, 1e-30),
        "mask_s_mean": np.mean(ms, axis=1),
        "mask_n_mean": np.mean(mn, axis=1),
    }


# =====================================================================
# 3. Condicionamiento de las covarianzas de ruido
# =====================================================================
def _weighted_scm(Z_stft, weights_t=None, start_frame=0):
    """SCM promediada en el tiempo, opcionalmente ponderada por mascara. (K,M,M)."""
    Z = Z_stft[:, start_frame:, :]
    if weights_t is None:
        R = np.einsum("ktm,ktn->kmn", Z, Z.conj()) / max(Z.shape[1], 1)
    else:
        w = weights_t[:, start_frame:][:, :, None]
        R = (np.einsum("ktm,ktn->kmn", w * Z, Z.conj())
             / np.maximum(np.sum(weights_t[:, start_frame:], axis=1)[:, None, None], 1e-15))
    return 0.5 * (R + np.conj(np.transpose(R, (0, 2, 1))))


def scm_conditioning_report(N_stft, X_stft=None, mask_n=None, start_frame=0):
    """
    Escalera de autovalores de la covarianza de ruido, por bin.

    Se calculan dos versiones:
      - VERDADERA : de la componente de ruido oracle, sin mascara. Es la que fija
                    el limite fisico (en campo difuso lambda_p/lambda_1 ~ (kL)^(2(p-1)),
                    la caida factorial que hace superdirectivo al problema).
      - ESTIMADA  : la que el beamformer realmente invirtio, es decir la mezcla
                    ponderada por la mascara de ruido. La distancia entre las dos
                    es el error de estimacion que la inversion amplifica por 1/lambda_p.

    Devuelve autovalores normalizados a lambda_1, rango efectivo y numero de
    condicion. Comparar `erank_true` con el DOF teorico 1+2L/lambda.
    """
    out = {}

    R_true = _weighted_scm(N_stft, None, start_frame)
    ev_true = np.linalg.eigvalsh(R_true)[:, ::-1]            # descendente (K,M)
    out["evals_true"] = ev_true
    out["evals_true_norm"] = ev_true / np.maximum(ev_true[:, :1], 1e-30)
    out["erank_true"] = effective_rank(ev_true)
    out["cond_true"] = ev_true[:, 0] / np.maximum(ev_true[:, -1], 1e-30)

    if X_stft is not None and mask_n is not None:
        Xa, mn = align_frames(X_stft, mask_n[:, :, None])
        R_est = _weighted_scm(Xa, mn[:, :, 0], start_frame)
        ev_est = np.linalg.eigvalsh(R_est)[:, ::-1]
        out["evals_est"] = ev_est
        out["evals_est_norm"] = ev_est / np.maximum(ev_est[:, :1], 1e-30)
        out["erank_est"] = effective_rank(ev_est)
        out["cond_est"] = ev_est[:, 0] / np.maximum(ev_est[:, -1], 1e-30)

        # Error relativo de estimacion en norma de Frobenius, escala-invariante.
        # Se compara contra la traza para que no dependa del nivel de senal.
        Kd = R_true.shape[0]
        sc_t = np.real(np.trace(R_true, axis1=1, axis2=2)).reshape(Kd, 1, 1)
        sc_e = np.real(np.trace(R_est, axis1=1, axis2=2)).reshape(Kd, 1, 1)
        A = R_true / np.maximum(sc_t, 1e-30)
        B = R_est / np.maximum(sc_e, 1e-30)
        out["scm_err_rel"] = (np.linalg.norm(A - B, axis=(1, 2))
                              / np.maximum(np.linalg.norm(A, axis=(1, 2)), 1e-30))

        # Piso de error de estimacion esperado por muestreo finito, sqrt(M/N_eff).
        # Es la referencia contra la que hay que comparar la carga diagonal: si
        # eps << este piso, la carga no esta regularizando nada.
        M = R_true.shape[1]
        N_eff = max(Xa.shape[1] - start_frame, 1)
        out["est_floor_sqrt_M_over_N"] = float(np.sqrt(M / N_eff))

    return out


# =====================================================================
# 3b. Degeneracion de la normalizacion de Souden
# =====================================================================
def souden_lambda_report(X_stft, mask_s, mask_n, min_loading=1e-9, start_frame=0):
    """
    Mide lambda = tr(Phi_NN^-1 Phi_XX), el denominador de la normalizacion de
    Souden, por bin.

    POR QUE IMPORTA. En el core, Phi_XX NO es la covarianza del target: es la
    covarianza de la MEZCLA ponderada por la mascara de voz, o sea
    Phi_XX ~ Phi_SS + Phi_NN. Entonces

        lambda = tr(Phi_NN^-1 Phi_SS) + M = lambda_S + M

    y lambda esta acotado por abajo por M. En un bin donde la mascara no
    encontro target (lambda_S -> 0) queda Phi_XX ~ Phi_NN, con lo cual

        Phi_NN^-1 Phi_XX -> I ,  lambda -> M ,  w -> u / M

    es decir, el filtro DEGENERA al microfono de referencia dividido por M: la
    banda entera sale atenuada ~1/M^2 (para M=8, -18 dB) con ganancia de arreglo
    NULA. No es un nulo espacial sobre la voz, es un colapso de ESCALA.

    lambda_excess = lambda/M - 1 = lambda_S/M es entonces un detector de
    confianza por bin, gratis (el core ya calcula lambda): ~0 significa "aca la
    mascara no vio target".

    Se usa acumulacion plana (alpha=1), que con fuentes estacionarias es el
    regimen optimo y aisla el efecto de la mascara del de tracking.
    """
    Xa, ms, mn = align_frames(X_stft, mask_s[:, :, None], mask_n[:, :, None])
    R_XX = _weighted_scm(Xa, ms[:, :, 0], start_frame)
    R_NN = _weighted_scm(Xa, mn[:, :, 0], start_frame)

    M = R_NN.shape[1]
    tr = np.real(np.trace(R_NN, axis1=1, axis2=2))
    eye = np.eye(M)[None, :, :]
    R_NN_st = R_NN + eye * (min_loading * (tr / M)[:, None, None] + 1e-12)

    lam = np.real(np.trace(np.linalg.solve(R_NN_st, R_XX), axis1=1, axis2=2))
    return {
        "lambda": lam,
        "lambda_over_M": lam / M,
        # lambda_S/M: exceso sobre el piso degenerado. ->0 = mascara sin target.
        "lambda_excess": np.maximum(lam / M - 1.0, 0.0),
        # atenuacion de escala que impone la normalizacion en el caso degenerado
        "degenerate_TR_dB": float(-20.0 * np.log10(M)),
    }


# =====================================================================
# 4. Agregacion por bandas (tercios de octava)
# =====================================================================
def third_octave_edges(f_min=50.0, f_max=8000.0):
    """Bordes de bandas de tercio de octava centradas en la serie 1000*2^(n/3)."""
    n = np.arange(-30, 20)
    fc = 1000.0 * 2.0 ** (n / 3.0)
    fc = fc[(fc >= f_min) & (fc <= f_max)]
    lo = fc / 2.0 ** (1.0 / 6.0)
    hi = fc * 2.0 ** (1.0 / 6.0)
    return fc, lo, hi


def aggregate_bands(freqs, report, f_min=50.0, f_max=8000.0):
    """
    Agrega el reporte narrowband a tercios de octava SUMANDO POTENCIAS y recien
    despues dividiendo (promediar dB o promediar ratios sesgaria el resultado).
    Devuelve un dict con arrays por banda + los centros fc.
    """
    fc, lo, hi = third_octave_edges(f_min, f_max)
    f = np.asarray(freqs)
    keep, rows = [], []

    for i in range(len(fc)):
        idx = np.where((f >= lo[i]) & (f < hi[i]))[0]
        if idx.size == 0:
            continue
        keep.append(i)
        Ps_i = report["P_S_in"][idx].sum()
        Pn_i = report["P_N_in"][idx].sum()
        Ps_o = report["P_S_out"][idx].sum()
        Pn_o = report["P_N_out"][idx].sum()
        rows.append({
            "TR": Ps_o / max(Ps_i, 1e-30),
            "NR": Pn_i / max(Pn_o, 1e-30),
            "AG": (Ps_o / max(Pn_o, 1e-30)) / max(Ps_i / max(Pn_i, 1e-30), 1e-30),
            "SNR_in": Ps_i / max(Pn_i, 1e-30),
            "SNR_out": Ps_o / max(Pn_o, 1e-30),
            # las cantidades que no son potencias se promedian ponderadas por la
            # energia de target del bin (los bins vacios no deben pesar).
            "SD_coh": float(np.average(report["SD_coh"][idx],
                                       weights=np.maximum(report["P_S_in"][idx], 1e-30))),
            "WNG": float(np.average(report["WNG"][idx],
                                    weights=np.maximum(report["P_S_in"][idx], 1e-30))),
        })

    out = {"fc": fc[keep]}
    for k in rows[0].keys():
        out[k] = np.array([r[k] for r in rows])
    return out


def band_summary_rows(freqs, report, label, extra=None,
                      bands=((0, 300), (300, 800), (800, 2000), (2000, 8000))):
    """
    Resumen compacto por bandas anchas, en dB. La primera banda (0-300 Hz) es
    deliberadamente la que PESQ no evalua: es el punto ciego del benchmark.
    """
    f = np.asarray(freqs)
    rows = []
    for lo, hi in bands:
        idx = np.where((f >= lo) & (f < hi))[0]
        if idx.size == 0:
            continue
        Ps_i = report["P_S_in"][idx].sum()
        Pn_i = report["P_N_in"][idx].sum()
        Ps_o = report["P_S_out"][idx].sum()
        Pn_o = report["P_N_out"][idx].sum()
        wts = np.maximum(report["P_S_in"][idx], 1e-30)
        row = {
            "label": label,
            "band": f"{lo}-{hi} Hz",
            "f_lo": lo, "f_hi": hi,
            "AG_dB": float(_to_db((Ps_o / max(Pn_o, 1e-30))
                                  / max(Ps_i / max(Pn_i, 1e-30), 1e-30))),
            "TR_dB": float(_to_db(Ps_o / max(Ps_i, 1e-30))),
            "NR_dB": float(_to_db(Pn_i / max(Pn_o, 1e-30))),
            "SNR_in_dB": float(_to_db(Ps_i / max(Pn_i, 1e-30))),
            "SNR_out_dB": float(_to_db(Ps_o / max(Pn_o, 1e-30))),
            "SD_coh": float(np.average(report["SD_coh"][idx], weights=wts)),
            "WNG_dB": float(_to_db(np.average(report["WNG"][idx], weights=wts))),
            "pesq_blind": lo < 300,
        }
        if extra is not None:
            for k, v in extra.items():
                v = np.asarray(v)
                if v.ndim == 1 and v.shape[0] == f.shape[0]:
                    row[k] = float(np.average(v[idx], weights=wts))
        rows.append(row)
    return rows


# =====================================================================
# 5. Figura
# =====================================================================
def plot_diagnostic(freqs, reports, aperture, M, out_path,
                    leakage=None, conditioning=None, lambda_rep=None,
                    f_min=60.0, f_max=8000.0, title=""):
    """
    Panel de 6 graficos. `reports` = {label: report_dict} para superponer
    variantes (tipicamente mask-based vs oracle vs delay-and-sum).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.asarray(freqs)
    sel = (f >= f_min) & (f <= f_max)
    fc = critical_frequency(aperture)

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0))
    fig.suptitle(title or "Diagnostico narrowband del beamformer", fontsize=13)

    def _mark(ax):
        ax.axvline(fc, color="0.35", ls="--", lw=1.0)
        ax.axvspan(f_min, 300, color="0.85", alpha=0.5, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(f_min, f_max)
        ax.grid(True, which="both", alpha=0.25)

    # (a) ganancia de arreglo
    ax = axes[0, 0]
    for lab, r in reports.items():
        ax.plot(f[sel], _to_db(r["AG"])[sel], lw=1.4, label=lab)
    _mark(ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(_to_db(M ** 2), color="g", ls=":", lw=1.0)
    ax.set_ylabel("Array gain [dB]")
    ax.set_title("(a) Ganancia de arreglo  (verde: cota superdirectiva $M^2$)")
    ax.legend(fontsize=7, loc="best")

    # (b) respuesta al target: el test de auto-cancelacion
    ax = axes[0, 1]
    for lab, r in reports.items():
        ax.plot(f[sel], _to_db(r["TR"])[sel], lw=1.4, label=lab)
    _mark(ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$P_{S,out}/P_{S,in}$ [dB]")
    ax.set_title("(b) Respuesta al TARGET  (<0 dB = se come la voz)")
    ax.legend(fontsize=7, loc="best")

    # (c) reduccion de ruido
    ax = axes[1, 0]
    for lab, r in reports.items():
        ax.plot(f[sel], _to_db(r["NR"])[sel], lw=1.4, label=lab)
    _mark(ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$P_{N,in}/P_{N,out}$ [dB]")
    ax.set_title("(c) Reduccion de ruido   [AG = (b) + (c)]")

    # (d) WNG
    ax = axes[1, 1]
    for lab, r in reports.items():
        ax.plot(f[sel], _to_db(r["WNG"])[sel], lw=1.4, label=lab)
    _mark(ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("WNG $=1/\\|w\\|^2$ [dB]")
    ax.set_title("(d) White noise gain  (muy negativo = superdirectivo/fragil)")

    # (e) DOF teorico + rango efectivo medido + fuga de target
    ax = axes[2, 0]
    ax.plot(f[sel], theoretical_dof(f, aperture, M=M)[sel], "k-", lw=1.6,
            label=f"DOF teorico $1+2L/\\lambda$  (L={aperture*100:.0f} cm)")
    if conditioning is not None:
        if "erank_true" in conditioning:
            ax.plot(f[sel], conditioning["erank_true"][sel], lw=1.2,
                    label="rango efectivo $\\Phi_{NN}$ (verdadera)")
        if "erank_est" in conditioning:
            ax.plot(f[sel], conditioning["erank_est"][sel], lw=1.2,
                    label="rango efectivo $\\Phi_{NN}$ (estimada, mascara)")
    _mark(ax)
    ax.axhline(2.0, color="r", ls=":", lw=1.0)
    ax.set_ylabel("grados de libertad")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_title(f"(e) DOF espaciales  ($f_c$={fc:.0f} Hz; rojo: DOF=2)")
    ax.legend(fontsize=7, loc="best")

    # (f) fuga de target en Phi_NN
    ax = axes[2, 1]
    if leakage is not None:
        ax.plot(f[sel], 100 * leakage["leak_NN"][sel], lw=1.4, color="tab:blue",
                label="target dentro de $\\Phi_{NN}$ (self-nulling)")
        ax.plot(f[sel], 100 * leakage["cont_XX"][sel], lw=1.4, color="tab:orange",
                label="ruido dentro de $\\Phi_{XX}$ (colapso de escala)")
        ax.set_ylabel("% de energia")
        ax.set_title("(f) Que metio la mascara en cada covarianza")
        if lambda_rep is not None:
            ax2 = ax.twinx()
            ax2.semilogy(f[sel], lambda_rep["lambda_over_M"][sel], lw=1.6,
                         color="k", ls="--", label="$\\lambda/M$ de Souden")
            ax2.axhline(1.0, color="r", ls=":", lw=1.2)
            ax2.set_ylabel("$\\lambda/M$  (1 = degenerado, $w=u/M$)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        else:
            ax.legend(fontsize=7, loc="best")
    else:
        for lab, r in reports.items():
            ax.plot(f[sel], r["SD_coh"][sel], lw=1.4, label=lab)
        ax.set_ylabel("$1-|coh|^2$")
        ax.set_title("(f) Distorsion (invariante a escala)")
        ax.legend(fontsize=7, loc="best")
    _mark(ax)
    ax.set_xlabel("Frecuencia [Hz]")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
