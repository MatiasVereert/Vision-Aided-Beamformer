"""
oracle_scm.py
=============
MVDR clasico (formulacion de vector de apuntamiento) con la matriz de covarianza
de ruido estimada de forma ORACLE: directamente de la señal de ruido+interferencia
LIMPIA multicanal, en vez de estimarla de la mezcla con un VAD.

    w(k) = Phi_NN^-1(k) d(k) / ( d^H(k) Phi_NN^-1(k) d(k) )

Es la pieza que faltaba para separar las dos fuentes de error del MVDR geometrico:

  - `MVDR_recursive` (base.py)      : d geometrico + Phi_NN estimada de la mezcla
                                      con VAD  -> error de apuntamiento Y error de
                                      estimacion de la covarianza, mezclados.
  - `MVDR_geo_oracle_scm` (aca)     : d geometrico + Phi_NN ORACLE -> aisla el error
                                      de APUNTAMIENTO (el modelo geometrico del
                                      steering vector) con la covarianza perfecta.
  - `MVDR_Souden_recursive_oracle`  : sin d; Phi_SS y Phi_NN oracle -> el techo del
    (mask/souden_mvdr.py)             MVDR sin ningun modelo geometrico.

Comparando los tres, la brecha entre este y el Souden-oracle es exactamente lo que
cuesta describir la fuente con un modelo geometrico (RTF de campo cercano en sala
reverberante) teniendo la misma informacion de segundo orden del ruido.

La covarianza se acumula recursivamente con el mismo factor de olvido `alpha` que
la familia Souden, y la carga diagonal usa la MISMA forma que el resto del repo:
    load = max( rel_loading * tr(Phi_NN)/M , min_loading )
con `rel_loading` como escala relativa (adimensional) y `min_loading` como piso
absoluto. Es la misma expresion que MVDR/base.py y que mask/souden_mvdr.py (donde
el rol de `rel_loading` lo cumple el parametro llamado alli `min_loading`).
"""

import numpy as np

from beamforming.signal_model import compute_rtf_steering_vector


def MVDR_geo_oracle_scm(X_stft, N_stft, fs, array_geometry, source_pos,
                        rel_loading=1e-2, min_loading=1e-9, alpha=0.99,
                        ref_mic_idx=None, sv_mode="near_field",
                        save_weights=False, verbose=False):
    """
    MVDR clasico con steering vector GEOMETRICO y Phi_NN ORACLE, recursivo por frame.

    Parameters
    ----------
    X_stft : (K, T, M) complex
        Mezcla OBSERVADA (lo que se filtra), en el mismo dominio en el que se
        entrega N_stft (p.ej. post hardware mismatch).
    N_stft : (K, T, M) complex
        Ruido + interferencia LIMPIO multicanal (referencia oracle). De aca sale
        Phi_NN; nunca se filtra ni se suma a la salida.
    fs : float
        Frecuencia de muestreo (define la grilla de frecuencias de los K bins).
    array_geometry : (M, 3)
        Coordenadas de los microfonos.
    source_pos : (1, 3) o (3,)
        Posicion ASUMIDA de la fuente (el benchmark ya le inyecta el error de DOA
        si lo hay). Con ella se arma el steering vector: es la unica informacion
        no-oracle del beamformer.
    rel_loading : float
        Escala RELATIVA de la carga diagonal: load = max(rel_loading*tr(Phi_NN)/M,
        min_loading). Default 1e-2, que es donde este beamformer rinde mejor sobre
        MIRD (con 1e-6 se autocancela por el desajuste del steering vector: pierde
        ~1.8 dB de SDR y ~0.06 de PESQ a cambio de ~2 dB de SIR). Para igualarlo al
        valor que usa la familia mask-based, pasar rel_loading=1e-6.
    min_loading : float
        Piso ABSOLUTO de la carga (default 1e-9); solo protege Phi_NN ~ 0.
    alpha : float
        Factor de olvido de la covarianza recursiva (alpha=1 -> oraculo global).
    ref_mic_idx : int or None
        Microfono de referencia del steering vector (define respecto a que canal
        se cumple la restriccion distortionless). None -> M // 2, la MISMA
        convencion que la familia Souden (souden_mvdr.py), para que la comparacion
        contra ella sea justa. Poner 0 para alinearse con DS / MVDR-geo de base.py.
    sv_mode : {"near_field", "far_field"}
        Modelo de propagacion del steering vector (igual que DS / MVDR-geo).
    save_weights : bool
        Si True devuelve tambien los pesos (K, T, M).
    verbose : bool
        Traza de progreso por frame (como las otras rutinas del repo).

    Returns
    -------
    Y_stft : (K, T) complex  [, weights_rec : (K, T, M) complex]
    """
    X_stft = np.asarray(X_stft)
    N_stft = np.asarray(N_stft)
    K, T, M = X_stft.shape

    if N_stft.shape != X_stft.shape:
        raise ValueError(f"N_stft {N_stft.shape} debe tener la misma forma que X_stft {X_stft.shape}")

    if ref_mic_idx is None:
        ref_mic_idx = M // 2

    # Grilla de frecuencias de la STFT (rfft): K bins de 0 a fs/2, igual que base.py.
    frecs = np.linspace(0, fs / 2, K)

    # Steering vector geometrico (K, M), normalizado al microfono de referencia.
    sv = compute_rtf_steering_vector(
        frecs, np.atleast_2d(source_pos), array_geometry,
        ref_mic_idx=ref_mic_idx, mode=sv_mode, squeeze=True
    )
    sv = np.reshape(sv, (K, M))

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Acumuladores de la covarianza oracle de ruido. El denominador es comun
    # (peso 1 por frame), igual que en MVDR_Souden_recursive_oracle.
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den = np.zeros((K, 1, 1), dtype=np.float64)

    eye = np.eye(M)[np.newaxis, :, :]

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        if verbose:
            print(f"\rProcessing frame {m} of {T}", end="")

        X_frame = X_stft[:, m, :]
        N_frame = N_stft[:, m, :]

        # Covarianza ORACLE: outer product del ruido limpio (sin mascara, sin VAD).
        R_N = np.einsum("fm,fn->fmn", N_frame, N_frame.conj())
        Num_NN = alpha * Num_NN + R_N
        Den = alpha * Den + 1.0
        Phi_NN = Num_NN / (Den + 1e-15)

        # Carga diagonal: escala relativa + piso absoluto (misma forma que base.py).
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = rel_loading * (tr_Phi / M)
        loading = np.maximum(adaptive_load, min_loading)

        Phi_NN_stable = Phi_NN + eye * loading[:, np.newaxis, np.newaxis]
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        # -----------------------------------------------------------------
        # MVDR CLASICO:  w = Phi_NN^-1 d / (d^H Phi_NN^-1 d)
        # -----------------------------------------------------------------
        weights_nom = np.einsum("fmn,fn->fm", Phi_NN_inv, sv)              # (K, M)
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)        # (K,)
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-15)
        # -----------------------------------------------------------------

        if save_weights:
            weights_rec[:, m, :] = weights

        # Los pesos se aplican a la MEZCLA OBSERVADA.
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    return Y_stft


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Test local sintetico (sin sala): campo libre near-field generado a mano,
    # arreglo lineal de 8 micros. Verifica lo que tiene que verificar el core:
    #   1) restriccion distortionless exacta: w^H d = 1 en todo bin/frame.
    #   2) con d perfecto, cancela interferencias muy por encima del DS.
    #   3) el error de apuntamiento (DOA errada) lo degrada -> es un MVDR
    #      geometrico de verdad, no un oraculo disfrazado.
    # ------------------------------------------------------------------
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import scipy.signal as sig

    from beamforming.MVDR.oracle_scm import MVDR_geo_oracle_scm  # noqa: F401 (self-import ok)

    rng = np.random.default_rng(0)
    FS, DUR, C = 16000, 4.0, 343.0
    N = int(FS * DUR)
    NPERSEG, NOVERLAP = 512, 384

    # Arreglo lineal de 8 micros (8 cm de apertura), centrado en el origen.
    M = 8
    mic_coords = np.zeros((M, 3))
    mic_coords[:, 0] = np.linspace(-0.04, 0.04, M)

    def pos(ang_deg, r=1.5):
        a = np.deg2rad(ang_deg)
        return np.array([r * np.cos(a), r * np.sin(a), 0.0])

    src_pos = pos(90.0)                       # broadside
    int_pos = [pos(30.0), pos(140.0)]

    def propagate(sig_mono, p):
        """Campo libre near-field: retardo fraccional (fase) + 1/r, en frecuencia."""
        d = np.linalg.norm(p[None, :] - mic_coords, axis=1)          # (M,)
        S = np.fft.rfft(sig_mono, n=N)
        f = np.fft.rfftfreq(N, 1 / FS)
        H = np.exp(-2j * np.pi * f[None, :] * d[:, None] / C) / d[:, None]
        return np.fft.irfft(S[None, :] * H, n=N, axis=1)

    def band_noise(seed):
        x = rng.standard_normal(N)
        b, a = sig.butter(4, [300 / (FS / 2), 3400 / (FS / 2)], btype="band")
        return sig.lfilter(b, a, x)

    target_m = propagate(band_noise(1), src_pos)
    interf_m = sum(propagate(band_noise(10 + k), p) for k, p in enumerate(int_pos))
    diffuse = 10 ** (-30 / 20) * np.std(interf_m) * rng.standard_normal((M, N))
    noise_m = interf_m + diffuse
    mix_m = target_m + noise_m

    def stft3(x):
        _, _, Z = sig.stft(x, fs=FS, window="hamming", nperseg=NPERSEG,
                           noverlap=NOVERLAP, nfft=NPERSEG)
        return np.transpose(Z, (1, 2, 0))     # (K, T, M)

    X, S, Nz = stft3(mix_m), stft3(target_m), stft3(noise_m)
    K, T, _ = X.shape
    print(f"[*] Escena sintetica: M={M}, K={K}, T={T}")

    def run(src, **kw):
        """Filtra mezcla, target-solo y ruido-solo con LOS MISMOS pesos -> SIR real."""
        _, w = MVDR_geo_oracle_scm(X, Nz, FS, mic_coords, src.reshape(1, 3),
                                   save_weights=True, **kw)
        y = np.einsum("ktm,ktm->kt", w.conj(), X)
        ys = np.einsum("ktm,ktm->kt", w.conj(), S)
        yn = np.einsum("ktm,ktm->kt", w.conj(), Nz)
        return w, y, ys, yn

    w, y, ys, yn = run(src_pos)

    # --- 1) distortionless: w^H d = 1 ---
    frecs = np.linspace(0, FS / 2, K)
    d = compute_rtf_steering_vector(frecs, src_pos.reshape(1, 3), mic_coords,
                                    ref_mic_idx=M // 2, mode="near_field", squeeze=True)
    resp = np.einsum("ktm,km->kt", w.conj(), d)
    err = np.max(np.abs(resp - 1.0))
    print(f"[1] max |w^H d - 1| = {err:.3e}  -> {'OK' if err < 1e-6 else 'FALLA'}")

    # --- 2) SIR de salida vs DS y vs la entrada ---
    def sir(a, b):
        return 10 * np.log10(np.sum(np.abs(a) ** 2) / np.sum(np.abs(b) ** 2))

    ref = M // 2
    sir_in = sir(S[:, :, ref], Nz[:, :, ref])
    w_ds = np.broadcast_to((d / M)[:, None, :], w.shape)
    sir_ds = sir(np.einsum("ktm,ktm->kt", w_ds.conj(), S),
                 np.einsum("ktm,ktm->kt", w_ds.conj(), Nz))
    sir_out = sir(ys, yn)
    print(f"[2] SIR entrada = {sir_in:6.2f} dB | DS = {sir_ds:6.2f} dB | "
          f"MVDR-geo+SCM oracle = {sir_out:6.2f} dB  -> "
          f"{'OK' if sir_out > sir_ds + 5 else 'FALLA'}")

    # --- 3) sensibilidad al error de apuntamiento ---
    # Con Phi_NN ORACLE no hay autocancelacion (el ruido de referencia no contiene
    # target), asi que apuntar mal NO se ve en el SIR: se ve como DISTORSION del
    # target. Se mide el error relativo de la componente de target a la salida
    # contra el target tal como lo recibe el microfono de referencia.
    def distor_db(ys_x):
        e = ys_x - S[:, :, ref]
        return 10 * np.log10(np.sum(np.abs(e) ** 2) / np.sum(np.abs(S[:, :, ref]) ** 2))

    print(f"[3] error DOA  0.0 deg -> SIR = {sir_out:6.2f} dB | "
          f"distorsion del target = {distor_db(ys):6.2f} dB")
    for e in (2.0, 10.0):
        _, _, ys_e, yn_e = run(pos(90.0 + e))
        print(f"[3] error DOA {e:4.1f} deg -> SIR = {sir(ys_e, yn_e):6.2f} dB | "
              f"distorsion del target = {distor_db(ys_e):6.2f} dB")

    # --- 4) la carga relativa hace el trade-off esperado (mas carga -> menos nulo) ---
    for rl in (1e-6, 1e-2, 1e-1):
        _, _, ys_l, yn_l = run(src_pos, rel_loading=rl)
        print(f"[4] rel_loading={rl:.0e} -> SIR = {sir(ys_l, yn_l):6.2f} dB | "
              f"distorsion del target = {distor_db(ys_l):6.2f} dB")
