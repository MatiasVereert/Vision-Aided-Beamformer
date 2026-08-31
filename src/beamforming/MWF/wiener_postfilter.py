"""
MWF POR DESCOMPOSICION: NM-MVDR (+ post-filtro de sustraccion espectral) seguido de
un WIENER MONOCANAL RELAJADO sobre la salida.

FUNDAMENTO
----------
El MWF multicanal admite la factorizacion clasica

    W_MWF = W_MVDR * G_wiener(k,t)

es decir, el MVDR (parte espacial, sin distorsion) seguido de una ganancia ESCALAR
por bin (parte de reduccion de ruido monocanal). El post-filtro (PF) que ya usa
NM_MVDR_PF ocupa el lugar de esa ganancia, pero es CIEGO a la SNR instantanea del
bin: G_pf = smooth + (1-smooth)*mask_soft aplica el mismo gate haya o no ruido
residual DESPUES del beamformer. Este modulo agrega la etapa que falta para cerrar
el MWF: una ganancia de Wiener estimada sobre la SNR RESIDUAL de la salida ya
conformada, o sea la que recorta solo donde el beamformer+PF efectivamente dejaron
ruido.

    X -> NM-MVDR (core base)                            -> Y_bf
      -> PF specsub: G_pf = smooth + (1-smooth)*mask    -> Y_pf
      -> Wiener DD: G_w = max(xi/(1+xi), g_min)         -> Y_out

ESTIMACION DEL RUIDO
--------------------
La PSD de ruido se estima sobre la SALIDA del PF (auto-consistente: el Wiener no
vuelve a restar lo que el PF ya resto) y se guia con la MASCARA DE RUIDO DEL DTLN,
que ya viene calculada gratis para el beamformer:

    a_eff = alpha_n + (1 - alpha_n) * (1 - mask_n)      # mask_n->1 actualiza, ->0 congela
    Phi_n <- a_eff * Phi_n + (1 - a_eff) * |Y_pf|^2

Es un MCRA guiado por mascara: el Phi_n no se contamina con el target porque el
update se congela en los frames de habla.

SNR a-priori por decision-directed (Ephraim-Malah), que es lo que evita el ruido
musical:

    xi = beta * |S_hat(t-1)|^2 / Phi_n + (1-beta) * max(gamma - 1, 0)

"RELAJADO" (parametros de diseno)
---------------------------------
  w_gmin_db : PISO de la ganancia. Es la perilla principal. 0 dB -> IDENTIDAD, o sea
              el core reproduce EXACTAMENTE MVDR_Souden_recursive_mask_specsub_base
              (verificado bit a bit). -6 dB es el punto util; -12 dB ya destruye PESQ.
  w_osf     : factor de (sub)estimacion del ruido. <1 SUBESTIMA el Phi_n -> xi mas
              alto -> ganancia mas cerca de 1, el gate solo muerde donde la SNR
              residual es realmente mala. Pesa tanto como el piso: osf=0.3 fue lo que
              hizo que la etapa DOMINE al barrido de `smooth` del PF (mejor PESQ *y*
              mejor STOI/SI-SDR, no un punto distinto de la misma curva).
  w_beta    : olvido del decision-directed. 0.98 = suave, sin ruido musical.

COSTO EN HW
-----------
Un estado real Phi_n por bin y ~3 multiplicaciones reales por bin y por frame. Sin
matrices, sin inversiones: es la etapa mas barata de toda la cadena.
"""

import numpy as np

from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask


def wiener_dd_gain(Y, mask_n=None, beta=0.98, g_min_db=-6.0, alpha_n=0.9,
                   osf=1.0, xi_min_db=-20.0, init_frames=15,
                   mask_s=None, gmin_mask=False, smooth_f=0, smooth_t=0.0):
    """
    Wiener decision-directed monocanal sobre una STFT (K,T) ya conformada.

    Y         : (K,T) complejo, salida del beamformer (+PF).
    mask_n    : (K,T) en [0,1], mascara de RUIDO del DTLN. Guia el update del Phi_n
                (None -> update ciego con alpha_n constante).
    beta      : olvido del decision-directed.
    g_min_db  : PISO de ganancia en dB (0 -> identidad).
    alpha_n   : olvido del Phi_n en frames 100% ruido.
    osf       : factor de (sub/sobre)-estimacion del ruido (<1 = mas relajado).
    xi_min_db : piso del SNR a-priori (secundario: g_min domina).
    init_frames : frames iniciales para el Phi_n de arranque (pesados por mask_n).

    --- PROTECCIONES DE STOI (off por defecto: no alteran el comportamiento base) ---
    El gate espectral cuesta inteligibilidad porque atenua TAMBIEN en bins con habla,
    y STOI mide exactamente esa distorsion de envolvente por banda. Las tres perillas
    de abajo atacan cada una un mecanismo distinto de esa perdida:

    mask_s    : (K,T) mascara de HABLA (la suave, sin realce). Necesaria si gmin_mask.
    gmin_mask : piso ADAPTATIVO g_min_eff = g_min ** (1 - mask_s). Donde el DTLN dice
                habla (mask_s->1) el piso sube a 1 y el Wiener NO PUEDE atenuar; donde
                dice ruido (mask_s->0) queda el piso pleno. O sea, el recorte se
                concentra en los bins que STOI no cuenta.
    smooth_f  : ancho (en bins, impar) del promedio movil de la ganancia en FRECUENCIA.
                Evita el desgarro espectral bin a bin, que es distorsion pura.
    smooth_t  : suavizado recursivo de la ganancia en el TIEMPO, en [0,1).
                G <- smooth_t*G(t-1) + (1-smooth_t)*G(t). Limita la fluctuacion de la
                envolvente, que es literalmente la variable que correlaciona STOI.

    Devuelve (Y_out, G) con G real (K,T).
    """
    K, T = Y.shape
    P = np.abs(Y) ** 2
    g_min = 10.0 ** (g_min_db / 20.0)      # piso en AMPLITUD
    xi_min = 10.0 ** (xi_min_db / 10.0)    # piso en POTENCIA
    eps = 1e-12

    # --- Phi_n inicial: promedio de los primeros frames pesado por mask_n ---
    n0 = min(init_frames, T)
    if mask_n is not None:
        w0 = np.clip(mask_n[:, :n0], 0.0, 1.0)
        den = w0.sum(axis=1)
        Phi_n = np.where(den > 1e-3,
                         (w0 * P[:, :n0]).sum(axis=1) / np.maximum(den, eps),
                         P[:, :n0].mean(axis=1))
    else:
        Phi_n = P[:, :n0].mean(axis=1)
    Phi_n = np.maximum(Phi_n, eps)

    # --- piso adaptativo por mascara de habla (proteccion de STOI) ---
    if gmin_mask and mask_s is not None:
        ms = np.clip(mask_s[:, :T], 0.0, 1.0)
        if ms.shape[1] < T:
            ms = np.pad(ms, ((0, 0), (0, T - ms.shape[1])), mode='edge')
        G_MIN = g_min ** (1.0 - ms)     # mask_s=1 -> 1.0 (no atenua) ; mask_s=0 -> g_min
    else:
        G_MIN = np.full((K, T), g_min)

    # kernel del promedio movil en frecuencia
    kf = None
    if smooth_f and smooth_f > 1:
        w = int(smooth_f) | 1          # forzar impar
        kf = np.ones(w) / float(w)

    G = np.ones((K, T), dtype=float)
    S_prev = P[:, 0].copy()   # |S_hat(t-1)|^2
    g_prev = None

    for t in range(T):
        if mask_n is not None:
            p = np.clip(mask_n[:, t], 0.0, 1.0)
            a_eff = alpha_n + (1.0 - alpha_n) * (1.0 - p)   # p=1 -> alpha_n ; p=0 -> 1 (congela)
        else:
            a_eff = alpha_n
        Phi_n = a_eff * Phi_n + (1.0 - a_eff) * P[:, t]
        Phi = np.maximum(osf * Phi_n, eps)

        gamma = P[:, t] / Phi                                   # SNR a-posteriori
        xi = beta * (S_prev / Phi) + (1.0 - beta) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, xi_min)

        g = xi / (1.0 + xi)

        # suavizado en frecuencia ANTES del piso, para que el promedio no arrastre
        # hacia abajo los bins ya protegidos por la mascara
        if kf is not None:
            g = np.convolve(g, kf, mode='same')

        g = np.maximum(g, G_MIN[:, t])

        # suavizado temporal (la envolvente es lo que mide STOI)
        if smooth_t > 0.0 and g_prev is not None:
            g = smooth_t * g_prev + (1.0 - smooth_t) * g

        G[:, t] = g
        g_prev = g
        S_prev = (g ** 2) * P[:, t]

    return Y * G, G


def MVDR_Souden_mask_specsub_MWF(X_stft, mask_s, mask_n, mask_s_soft,
                                 min_loading=1e-6, alpha=0.99, smooth=0.33,
                                 w_beta=0.98, w_gmin_db=-6.0, w_alpha_n=0.9,
                                 w_osf=0.3, w_xi_min_db=-20.0,
                                 w_gmin_mask=False, w_smooth_f=0, w_smooth_t=0.0,
                                 save_weights=False, ref_mic_idx=None):
    """
    CORE BASE (NM-MVDR, el del ganador) + specsub (PF) + Wiener DD relajado = MWF.

    Es una EXTENSION ESTRICTA de MVDR_Souden_recursive_mask_specsub_base: con
    w_gmin_db=0 la etapa Wiener es identidad y la salida coincide bit a bit con
    aquel core (misma carga diagonal relativa con piso absoluto, mismo alpha).

    Ver el docstring del modulo para el significado de w_gmin_db / w_osf / w_beta.
    """
    res = MVDR_Souden_recursive_mask(X_stft, mask_s, mask_n,
                                     min_loading=min_loading, alpha=alpha,
                                     save_weights=save_weights, ref_mic_idx=ref_mic_idx)
    Y, W = (res if save_weights else (res, None))
    Y = Y.copy()

    # --- Etapa 1: PF de sustraccion espectral (identica a la del core base) ---
    Tm = min(Y.shape[1], mask_s_soft.shape[1])
    G_pf = smooth + (1.0 - smooth) * np.clip(mask_s_soft[:, :Tm], 0.0, 1.0)
    Y[:, :Tm] *= G_pf

    # --- Etapa 2: Wiener DD sobre la salida del PF (cierra el MWF) ---
    mn = None
    if mask_n is not None:
        mn = mask_n[:, :Y.shape[1]]
        if mn.shape[1] < Y.shape[1]:
            mn = np.pad(mn, ((0, 0), (0, Y.shape[1] - mn.shape[1])), mode='edge')
    Y, _ = wiener_dd_gain(Y, mask_n=mn, beta=w_beta, g_min_db=w_gmin_db,
                          alpha_n=w_alpha_n, osf=w_osf, xi_min_db=w_xi_min_db,
                          mask_s=mask_s_soft, gmin_mask=w_gmin_mask,
                          smooth_f=w_smooth_f, smooth_t=w_smooth_t)

    if save_weights:
        return Y, W
    return Y


# =============================================================================
# POST-FILTRO ADAPTATIVO: agresividad programada por el iSIR ESTIMADO
# =============================================================================
# Motivacion (medido en tests/pf_mwf_stoi_guard_sweep.py): la tasa de cambio
# PESQ-por-STOI del post-filtro depende fuerte del punto de operacion. A iSIR alto
# (>=10 dB) el gate sale casi gratis -- 22-31 puntos de PESQ por punto de STOI
# cedido -- porque la mascara del DTLN acierta y solo recorta ruido. A iSIR bajo
# (<=5 dB) cuesta 3-4x mas: la mascara se equivoca y el gate pega sobre habla.
#
# Corolario de diseno, y es CONTRA-INTUITIVO: hay que ser AGRESIVO ARRIBA y
# CONSERVADOR ABAJO, no al reves. Proteger bins dentro del gate no sirve (la
# mascara ya define la ganancia del PF, asi que protegerse con ella es circular);
# lo que si aporta informacion nueva es el punto de operacion.
#
# OJO: esto es sobre el eje iSIR (interferente direccional). Sobre el eje de SNR
# TERMICA la tendencia medida fue la opuesta (la ventaja del Wiener crecia al bajar
# la SNR), asi que este schedule NO se debe extrapolar a ruido de sensor.


def estimate_isir_db(X_ref, mask_s_soft, lo_db=-15.0, hi_db=30.0):
    """
    iSIR CIEGO en el microfono de referencia, pesado por la mascara del DTLN.

        P_s = sum mask_s * |X_ref|^2 ;  P_n = sum (1 - mask_s) * |X_ref|^2
        iSIR_est = 10 log10(P_s / P_n)

    X_ref       : (K,T) STFT del canal de referencia (la mezcla).
    mask_s_soft : (K,T) mascara de habla SIN realce (la del DTLN tal cual).

    Es un estimador SESGADO -- la mascara se aplica sobre la mezcla, asi que P_s
    arrastra ruido y el rango sale comprimido respecto del iSIR real -- pero para
    programar la agresividad solo hace falta que sea MONOTONO. La compresion se
    absorbe calibrando los umbrales del schedule en unidades del ESTIMADOR, no del
    iSIR verdadero (ver tests/pf_mwf_adaptive_calib.py).

    No usa ninguna senal de referencia: solo la mezcla y la mascara que el
    beamformer ya calcula. Es implementable online (una suma corriente por frame).
    """
    P = np.abs(X_ref) ** 2
    ws = np.clip(mask_s_soft[:, :P.shape[1]], 0.0, 1.0)
    if ws.shape[1] < P.shape[1]:
        P = P[:, :ws.shape[1]]
    Ps = float((ws * P).sum())
    Pn = float(((1.0 - ws) * P).sum())
    val = 10.0 * np.log10(max(Ps, 1e-20) / max(Pn, 1e-20))
    return float(np.clip(val, lo_db, hi_db))


def schedule_aggressiveness(isir_est_db, lo_db, hi_db,
                            smooth_lo, smooth_hi, gmin_lo_db, gmin_hi_db):
    """
    Rampa lineal en dB entre el punto CONSERVADOR (isir <= lo_db) y el AGRESIVO
    (isir >= hi_db). Devuelve (smooth, g_min_db) para esta escena.

    smooth_lo / gmin_lo_db : configuracion en el extremo conservador (iSIR bajo).
    smooth_hi / gmin_hi_db : configuracion en el extremo agresivo (iSIR alto).

    Nota: `smooth` mas ALTO = post-filtro mas SUAVE (1.0 = sin filtro), y g_min mas
    cerca de 0 dB = Wiener mas suave. O sea el extremo conservador lleva smooth alto
    y g_min alto.
    """
    if hi_db <= lo_db:
        t = 1.0
    else:
        t = (float(isir_est_db) - lo_db) / (hi_db - lo_db)
    t = min(max(t, 0.0), 1.0)
    smooth = smooth_lo + t * (smooth_hi - smooth_lo)
    g_min_db = gmin_lo_db + t * (gmin_hi_db - gmin_lo_db)
    return float(smooth), float(g_min_db)
