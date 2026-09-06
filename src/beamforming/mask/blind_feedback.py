"""
blind_feedback.py
=================
LAZO CIEGO REALIMENTADO FRAME A FRAME, CON UN SOLO DTLN.

QUE CAMBIA RESPECTO DE `NM_MVDR_DSM_BLIND`
------------------------------------------
El wrapper de dos pasadas corre el DTLN DOS veces sobre toda la senal:

    mascara(1) = DTLN(x_ref)          <- solo para apuntar
    d          = RTF(Phi_SS(mascara(1)))
    y_fix      = w(d)^H x
    mascara(2) = DTLN(y_fix)          <- la que usa el beamformer

La mascara(1) es un BOOTSTRAP: no llega al beamformer. Y el lazo se bootstrapea
SOLO, porque el estado inicial del estimador ya ES el canal de referencia. Este
lazo arranca con las DOS mascaras en cero, asi que en t = 0 ninguna rama tiene
masa y

    Phi_XX = Phi_NN = 0  ->  Phi_SS = 0  ->  R_est = 0  ->  manda la carga
                         ->  d = e_ref  ->  w = e_ref  ->  y_fix = x_ref

El front-end arranca siendo el canal crudo por construccion, o sea mirando lo
mismo que miraria la pasada (1). Verificado en
tests/window_mismatch/test_blind_feedback_equiv.py: max|d(0) - e_ref| = 1e-10.

OJO CON GENERALIZAR ESTO. Tambien vale si las dos ramas tienen masa comparable
(ahi Phi_XX = Phi_NN = R, la MISMA matriz instantanea, y la resta se cancela),
pero NO vale si la rama de RUIDO se queda sin masa: con Phi_NN ~ 0 la
sustraccion no se cancela, el nivel de ruido que escala la carga es ~0 y d se va
de e_ref. Con mascaras realzadas a p = 8 eso pasa en cualquier bin donde el DTLN
esta seguro de que hay voz. Por eso el arranque en cero no es un detalle de
implementacion: es lo que hace que el bootstrap valga en TODOS los bins.

    mascara(t) = DTLN( y_fix(t) )
    y_fix(t)   = w(d(t))^H x(t) ,   d(t) = RTF(Phi_SS(mascara(t-1)))

Un solo DTLN, la mitad del computo de red. El precio es que CIERRA EL LAZO: hoy
la mascara(1) es un ancla que la realimentacion no puede corromper, y aca no hay
ancla. Por eso `conf_gate` deja de ser un extra y pasa a ser el seguro principal
(ver `estimate_rtf_recursive`).

Y NO HACE FALTA VOLVER AL TIEMPO
--------------------------------
El DTLN enmarca con un buffer deslizante y la STFT del beamformer usa ventana
RECTANGULAR, asi que las dos ventanas son la misma (ver la nota de
`dtln_masks.py`). Verificado con error EXACTAMENTE 0:

    rFFT( bloque i del DTLN )  ==  L * X(:, i-1)          [con L = nperseg]

Como el bloque i corresponde al frame i-1, alimentar al DTLN con el espectro del
frame t devuelve la mascara del frame t: la correccion de +1 frame de
`align_mask_frames` queda IMPLICITA, no hay que aplicarla. Y como no hay iSTFT ni
re-analisis, el lazo no necesita ni un buffer de overlap-add ni una transformada
extra: es la forma "una sola FFT en todo el sistema" que persigue el resto del
repositorio.

OJO: eso SI es un cambio respecto del camino de dos pasadas, donde la mascara(2)
se calcula sobre y_fix RESINTETIZADA (iSTFT con solapamiento) y vuelta a
enmarcar. Como w varia con (k,t), Y_fix no es la STFT de ninguna senal temporal:
resintetizar y re-analizar es una PROYECCION, no la identidad, y mezcla 4 frames
filtrados distintos. Para poder separar los dos efectos, el wrapper
`NM_MVDR_DSM_FB` tiene un modo `mode="spec"` que corre la alimentacion espectral
SIN realimentacion (ver tests/window_mismatch/run_dsm_blind_feedback.py).

VERIFICACION
------------
Con `mask_fixed=(m_s, m_n)` se saltea el DTLN y se usan esas mascaras en las dos
ramas, sin retardo de realimentacion: en ese modo esta funcion reproduce
BIT A BIT `estimate_rtf_recursive` + `MVDR_Souden_recursive_mask_subtract`. Es la
prueba de que la re-implementacion por frames no cambio la matematica.
"""

import numpy as np

from .ds_mask import _rtf_from_loaded
from .scm_calibration import diffuse_coherence


# =====================================================================
# DTLN en modo streaming: un bloque por llamada
# =====================================================================
class DTLNStream:
    """
    Envuelve el interprete tflite para inferencia BLOQUE A BLOQUE, manteniendo el
    estado LSTM entre llamadas. Es el mismo modelo y el mismo lazo interno que
    `get_dtln_masks_soft`, pero expuesto como `step()` para poder intercalarlo
    con la recursion del beamformer.

    A diferencia de las funciones de `dtln_masks.py`, NO enmarca: recibe
    directamente el espectro de magnitud del bloque. Ver la nota del modulo sobre
    por que eso es equivalente a enmarcar con ventana rectangular.
    """

    def __init__(self, model_path):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()
        self.states = np.zeros(self.inp[1]['shape'], dtype=np.float32)

    def step(self, mag):
        """mag: (K,) magnitud del bloque. Devuelve la mascara (K,) en [0,1]."""
        x = np.reshape(np.asarray(mag, dtype=np.float32), (1, 1, -1))
        self.interpreter.set_tensor(self.inp[1]['index'], self.states)
        self.interpreter.set_tensor(self.inp[0]['index'], x)
        self.interpreter.invoke()
        m = self.interpreter.get_tensor(self.out[0]['index'])
        self.states = self.interpreter.get_tensor(self.out[1]['index'])
        return np.squeeze(m)


# =====================================================================
# Estimador de RTF, por frame
# =====================================================================
class RTFEstimator:
    """
    Forma por frame de `estimate_rtf_recursive`. Mismo estado, mismas constantes,
    mismo orden de operaciones; lo unico que cambia es que se llama de a un frame
    para poder intercalar el DTLN en el medio.
    """

    def __init__(self, K, M, ref_mic, rtf_alpha=0.999, rtf_loading=1e-2,
                 rtf_mode="cs", w_mode="ds", bf_loading=1e-6, Gamma=None,
                 sd_eps=1e-2, conf_gate=None, conf_bins=None, conf_smooth=0.9,
                 conf_alpha=None):
        if rtf_mode != "cs":
            raise ValueError("blind_feedback solo implementa rtf_mode='cs'.")
        if w_mode not in ("ds", "sd", "mvdr"):
            raise ValueError(f"w_mode desconocido: {w_mode!r}")
        self.K, self.M, self.ref = K, M, ref_mic
        self.a = float(rtf_alpha)
        self.load = float(rtf_loading)
        self.w_mode = w_mode
        self.bf_loading = float(bf_loading)
        self.eye = np.eye(M)[None, :, :]

        self.G_inv = None
        if w_mode == "sd":
            if Gamma is None:
                raise ValueError("w_mode='sd' necesita Gamma (K, M, M).")
            eps = float(np.clip(sd_eps, 0.0, 1.0))
            G = ((1.0 - eps) * np.asarray(Gamma).astype(np.complex128)
                 + (eps + 1e-12) * np.eye(M)[None, :, :])
            self.G_inv = np.linalg.inv(G)

        self.Num_XX = np.zeros((K, M, M), dtype=np.complex128)
        self.Num_NN = np.zeros((K, M, M), dtype=np.complex128)
        self.Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
        self.Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

        # gate de arranque en frio + recursion SOMBRA que mide la confianza
        self.conf_gate = conf_gate
        self.cb = np.ones(K, dtype=bool) if conf_bins is None else np.asarray(conf_bins, bool)
        self.conf_smooth = float(conf_smooth)
        self.a_c = self.a if conf_alpha is None else float(conf_alpha)
        self.use_conf = conf_gate is not None
        self.Num_XXs = np.zeros((K, M, M), dtype=np.complex128) if self.use_conf else None
        self.Den_XXs = np.zeros((K, 1, 1), dtype=np.float64) if self.use_conf else None
        self.conf = 0.0          # arranca cerrado: seguro por default
        self.last_gate = 1.0

    def step(self, X_frame, m_s, m_n):
        """X_frame: (K, M). m_s, m_n: (K,). Devuelve w (K, M) y d (K, M)."""
        M = self.M
        R = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        ms = m_s[:, None, None]
        mn = m_n[:, None, None]

        g = 1.0 if not self.use_conf else float(self.conf >= self.conf_gate)
        self.last_gate = g

        self.Num_XX = self.a * self.Num_XX + g * ms * R
        self.Den_XX = self.a * self.Den_XX + g * ms
        self.Num_NN = self.a * self.Num_NN + mn * R
        self.Den_NN = self.a * self.Den_NN + mn

        Phi_XX = self.Num_XX / (self.Den_XX + 1e-15)
        Phi_NN = self.Num_NN / (self.Den_NN + 1e-15)
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        Phi_SS = Phi_XX - Phi_NN
        Phi_SS = 0.5 * (Phi_SS + np.conj(np.transpose(Phi_SS, (0, 2, 1))))

        nlev = np.real(np.trace(Phi_NN, axis1=1, axis2=2)) / M

        if self.use_conf:
            self.Num_XXs = self.a_c * self.Num_XXs + ms * R
            self.Den_XXs = self.a_c * self.Den_XXs + ms
            Phi_XXs = self.Num_XXs / (self.Den_XXs + 1e-15)
            tr_ss = (np.real(np.trace(Phi_XXs, axis1=1, axis2=2)) / M) - nlev
            conf_inst = float(np.mean(tr_ss[self.cb] > 0.0))
            self.conf = self.conf_smooth * self.conf + (1.0 - self.conf_smooth) * conf_inst

        load_rtf = self.load * nlev + 1e-20

        evals, evecs = np.linalg.eigh(Phi_SS)
        evals = np.maximum(evals, 0.0)
        R_est = np.einsum("fmp,fnp->fmn", evecs * evals[:, None, :], evecs.conj())

        d = _rtf_from_loaded(R_est, load_rtf, self.ref)

        if self.w_mode == "ds":
            den = np.real(np.einsum("fm,fm->f", d.conj(), d))
            w = d / (den[:, None] + 1e-30)
        elif self.w_mode == "sd":
            Gi_d = np.einsum("fmn,fn->fm", self.G_inv, d)
            den = np.real(np.einsum("fm,fm->f", d.conj(), Gi_d))
            w = Gi_d / (den[:, None] + 1e-30)
        else:
            load_nn = self.bf_loading * nlev + 1e-20
            Phi_NN_l = Phi_NN + self.eye * load_nn[:, None, None]
            Pi_d = np.linalg.solve(Phi_NN_l, d[:, :, None])[:, :, 0]
            den = np.real(np.einsum("fm,fm->f", d.conj(), Pi_d))
            w = Pi_d / (den[:, None] + 1e-30)

        return w, d


# =====================================================================
# Nucleo de Souden con sustraccion, por frame
# =====================================================================
class SoudenSubtractCore:
    """Forma por frame de `MVDR_Souden_recursive_mask_subtract` (mu, BAN, PSD)."""

    def __init__(self, K, M, ref_mic, alpha=0.99, min_loading=1e-9, mu=0.0,
                 lambda_floor=1e-3, psd_project=True, ban=False):
        self.K, self.M, self.ref = K, M, ref_mic
        self.alpha = float(alpha)
        self.min_loading = float(min_loading)
        self.mu = float(mu)
        self.lambda_floor = float(lambda_floor)
        self.psd_project = bool(psd_project)
        self.ban = bool(ban)
        self.eye = np.eye(M)[None, :, :]
        self.Num_XX = np.zeros((K, M, M), dtype=np.complex128)
        self.Num_NN = np.zeros((K, M, M), dtype=np.complex128)
        self.Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
        self.Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    def step(self, X_frame, m_s, m_n):
        """Devuelve (Y (K,), weights (K, M))."""
        M = self.M
        R = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        ms = m_s[:, None, None]
        mn = m_n[:, None, None]

        self.Num_XX = self.alpha * self.Num_XX + ms * R
        self.Den_XX = self.alpha * self.Den_XX + ms
        self.Num_NN = self.alpha * self.Num_NN + mn * R
        self.Den_NN = self.alpha * self.Den_NN + mn

        Phi_XX = self.Num_XX / (self.Den_XX + 1e-15)
        Phi_NN = self.Num_NN / (self.Den_NN + 1e-15)
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        Phi_SS = Phi_XX - Phi_NN
        Phi_SS = 0.5 * (Phi_SS + np.conj(np.transpose(Phi_SS, (0, 2, 1))))

        if self.psd_project:
            evals, evecs = np.linalg.eigh(Phi_SS)
            evals = np.maximum(evals, 0.0)
            Phi_SS = np.einsum("fmp,fnp->fmn", evecs * evals[:, None, :], evecs.conj())

        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = self.min_loading * (tr_Phi / M)
        Phi_NN_stable = Phi_NN + self.eye * (adaptive_load[:, None, None] + 1e-12)

        B = np.linalg.solve(Phi_NN_stable, Phi_SS)
        lambda_S = np.real(np.trace(B, axis1=1, axis2=2))
        lambda_S = np.maximum(lambda_S, self.lambda_floor)
        weights = B[:, :, self.ref] / (lambda_S[:, None] + self.mu + 1e-15)

        if self.ban:
            Phi_NN_w = np.einsum("fmn,fn->fm", Phi_NN_stable, weights)
            den = np.real(np.einsum("fm,fm->f", weights.conj(), Phi_NN_w))
            num = np.sqrt(np.real(np.einsum("fm,fm->f", Phi_NN_w.conj(), Phi_NN_w)) / M)
            weights = weights * (num / (den + 1e-15))[:, None]

        Y = np.einsum("fm,fm->f", weights.conj(), X_frame)
        return Y, weights


# =====================================================================
# El lazo
# =====================================================================
def blind_feedback_stft(X_stft, model_path, nperseg, ref_mic_idx=None,
                        sharpen_exp=8.0, rtf_alpha=0.999, rtf_loading=1e-2,
                        rtf_mode="cs", w_mode="ds", bf_loading=1e-6,
                        alpha=0.99, min_loading=1e-9, mu=0.0, lambda_floor=1e-3,
                        psd_project=True, ban=False, smooth=None,
                        conf_gate=None, conf_bins=None, conf_smooth=0.9,
                        conf_alpha=None, mic_coords=None, freqs=None,
                        sd_eps=1e-2, sd_field="spherical", mask_warp=None,
                        feedback=True, mask_fixed=None, return_diag=False,
                        progress=True):
    """
    Corre la cadena completa frame a frame sobre una STFT de ANALISIS RECTANGULAR.

    Args:
        X_stft: (K, T, M) STFT multicanal, ventana rectangular (`boxcar`).
        model_path: .tflite del DTLN (primera etapa).
        nperseg: L. Hace falta para deshacer el escalado 1/L de scipy antes de
            darle la magnitud al DTLN.
        feedback: True -> la mascara del frame t-1 alimenta el estimador de RTF
            (UN solo DTLN, lazo cerrado). False -> el estimador de RTF nunca ve
            la mascara: se queda con d = e_ref y el DTLN mira el canal crudo, o
            sea el sistema base. Sirve de control.
        mask_fixed: (m_s, m_n) de (K, T). Saltea el DTLN y usa esas mascaras en
            las dos ramas SIN retardo -- modo de verificacion, reproduce
            bit a bit el camino de dos pasadas (ver el docstring del modulo).
        smooth: post-filtro G = smooth + (1-smooth) m_raw. None = desactivado.

    Returns:
        (Y_stft (K,T), weights (K,T,M))  -- o (..., diag) con return_diag=True.
    """
    X_stft = np.asarray(X_stft)
    K, T, M = X_stft.shape
    ref = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
    p = float(sharpen_exp)

    Gamma = None
    if w_mode == "sd":
        if mic_coords is None or freqs is None:
            raise ValueError("w_mode='sd' necesita mic_coords y freqs.")
        Gamma = diffuse_coherence(np.asarray(mic_coords, dtype=np.float64),
                                  np.asarray(freqs, dtype=np.float64), field=sd_field)

    rtf = RTFEstimator(K, M, ref, rtf_alpha=rtf_alpha, rtf_loading=rtf_loading,
                       rtf_mode=rtf_mode, w_mode=w_mode, bf_loading=bf_loading,
                       Gamma=Gamma, sd_eps=sd_eps, conf_gate=conf_gate,
                       conf_bins=conf_bins, conf_smooth=conf_smooth,
                       conf_alpha=conf_alpha)
    core = SoudenSubtractCore(K, M, ref, alpha=alpha, min_loading=min_loading,
                              mu=mu, lambda_floor=lambda_floor,
                              psd_project=psd_project, ban=ban)

    dtln = None if mask_fixed is not None else DTLNStream(model_path)

    Y_stft = np.zeros((K, T), dtype=np.complex128)
    W_out = np.zeros((K, T, M), dtype=np.complex128)
    diag = ({k: np.zeros((K, T)) for k in ("m_raw", "gate", "conf")}
            if return_diag else None)

    # Mascara que ve el estimador de RTF en el frame t. Arranca en CERO: con las
    # dos ramas sin masa, Phi_XX = Phi_NN = 0 -> Phi_SS = 0 -> d = e_ref, o sea
    # y_fix(0) = x_ref. Es el auto-bootstrap descrito arriba.
    m_s_fb = np.zeros(K)
    m_n_fb = np.zeros(K)

    for t in range(T):
        if progress and (t % 32 == 0 or t == T - 1):
            print(f"\r  [feedback] frame {t+1}/{T}", end="")
        X_frame = X_stft[:, t, :]

        if mask_fixed is not None:
            # Modo verificacion: las mascaras dadas alimentan las dos ramas, sin
            # retardo, exactamente como en el camino de dos pasadas.
            m_s = np.asarray(mask_fixed[0])[:, t]
            m_n = np.asarray(mask_fixed[1])[:, t]
            w_fix, _ = rtf.step(X_frame, m_s, m_n)
        else:
            # 1. apuntar con la mascara del frame ANTERIOR (estrictamente causal)
            w_fix, _ = rtf.step(X_frame, m_s_fb, m_n_fb)

            # 2. front-end: una combinacion lineal, en el mismo dominio espectral
            Y_fix = np.einsum("fm,fm->f", w_fix.conj(), X_frame)

            # 3. el DTLN consume el espectro directamente. El factor L deshace el
            #    escalado de scipy.signal.stft con ventana rectangular, de modo
            #    que la magnitud es la del bloque que enmarcaria el DTLN. Al
            #    corresponder el bloque t+1 al frame t, la mascara sale YA
            #    alineada: no hay que aplicar align_mask_frames.
            m_raw = np.clip(np.asarray(dtln.step(np.abs(nperseg * Y_fix)),
                                       dtype=np.float64), 0.0, 1.0)

            if mask_warp is None:
                m_s = m_raw ** p
                m_n = (1.0 - m_raw) ** p
            else:
                a_s, b_s, a_n, b_n = mask_warp
                m_s = np.clip(a_s * m_raw + b_s, 1e-4, 1.0)
                m_n = np.clip(a_n * (1.0 - m_raw) + b_n, 1e-4, 1.0)

            if feedback:
                m_s_fb, m_n_fb = m_s, m_n
            if diag is not None:
                diag["m_raw"][:, t] = m_raw

        # 4. nucleo, con la mascara de ESTE frame
        Y, w = core.step(X_frame, m_s, m_n)

        # 5. post-filtro de sustraccion espectral
        if smooth is not None and mask_fixed is None:
            Y = Y * (smooth + (1.0 - smooth) * m_raw)

        Y_stft[:, t] = Y
        W_out[:, t, :] = w
        if diag is not None:
            diag["gate"][:, t] = rtf.last_gate
            diag["conf"][:, t] = rtf.conf

    if progress:
        print()
    return (Y_stft, W_out, diag) if return_diag else (Y_stft, W_out)
