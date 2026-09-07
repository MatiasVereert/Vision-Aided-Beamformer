"""
output_feedback.py
==================
LAZO CERRADO SOBRE LA SALIDA DEL BEAMFORMER (sin front-end propio, sin RTF).

QUE CAMBIA RESPECTO DE `blind_feedback.py`
------------------------------------------
`NM_MVDR_DSM_FB` son, en el fondo, DOS beamformers semi-desacoplados:

    d(t)   = RTF(Phi_SS(mascara(t-1)))          <- eigh de Phi_SS, por bin
    y_fe   = w(d)^H x                            <- front-end SOLO para la mascara
    m(t)   = DTLN(y_fe)
    Y      = core_Souden(X, m(t))                <- el que realmente se escucha

El front-end existe porque el nucleo no puede alimentar al DTLN: sus pesos
dependen de la mascara que el DTLN todavia no calculo. Pero con
`block_update`, el frame t YA se filtra con pesos calculados en t-1: esos pesos
estan disponibles ANTES de correr la red. O sea que la excusa desaparece.

    Y(t)   = w(t-1)^H x(t)                       <- el unico beamformer
    m(t)   = DTLN( Y(t) )                        <- la mascara sale de la SALIDA
    w(t)   = core_Souden.solve( SCM(m(t)) )

Dos ventajas:
  * se cae el eigh de Phi_SS del front-end (la mitad cara de la etapa barata) y
    con el toda la estimacion de RTF: queda UN solo camino de pesos;
  * el DTLN come una senal con el iSIR del MVDR completo, no el de un DS
    apuntado. Es la mejor entrada que el sistema puede fabricar.

Y UN PELIGRO NUEVO, DE OTRA CATEGORIA
-------------------------------------
En `blind_feedback` la realimentacion entra por el APUNTAMIENTO: si la RTF se
degrada, el peor caso es d -> e_ref, o sea el canal crudo. La mascara sigue
saliendo de una senal que contiene la voz.

Aca la realimentacion entra por la SENAL que ve la red, y el lazo es
autoexcitado en el sentido malo:

    voz cancelada en Y  ->  m_s ~ 0, m_n ~ 1  ->  la voz entra a Phi_NN
                        ->  el MVDR la anula mas  ->  voz mas cancelada

Es un estado ABSORBENTE: una vez adentro, no hay nada en el lazo que informe que
la voz existe. Cualquier transitorio malo (arranque en frio, un corte de voz
largo, un interferente que domina) puede meterlo ahi y no vuelve.

LAS TRES DEFENSAS (independientes, se pueden combinar)
------------------------------------------------------
1. `leak` -- FUGA DEL CANAL DE REFERENCIA (la principal, y casi gratis).
   La red no come Y sino una mezcla convexa con el canal crudo:

       y_mask(t) = (1 - b) * Y(t) + b * x_ref(t)

   Como el nucleo de Souden es distorsionless respecto del mic de referencia
   (w^H a = 1), la componente de VOZ de los dos terminos es la misma senal y se
   suma coherente: la mezcla no distorsiona la voz. El ruido de x_ref, en
   cambio, entra atenuado b.

   Lo importante es el PEOR CASO: si el beamformer cancela la voz por completo
   (Y = 0), entonces y_mask = b * x_ref, o sea EXACTAMENTE el canal de
   referencia escalado. El lazo degrada de forma CONTINUA al sistema base en
   vez de colapsar; la mascara se puede recuperar y sacar al lazo del pozo. Es
   la misma idea que `rtf_loading` en el lazo ciego (d -> e_ref), aplicada a la
   senal en vez de al apuntamiento.

   El precio es que la ganancia de iSIR que ve la red queda ACOTADA a ~
   -20 log10(b) dB (b = 0.05 -> 26 dB, de sobra). No es un limite del sistema:
   solo de lo que ve el DTLN.

2. `guard='snr'` -- PERRO GUARDIAN MASK-INDEPENDIENTE (sin segunda red).
   El colapso no se puede detectar mirando la mascara ni Phi_NN: las dos estan
   contaminadas por el propio lazo (con la voz adentro de la rama de ruido,
   Phi_NN se infla y el "nivel de ruido" deja de ser el nivel de ruido). Hace
   falta un estadistico que NO dependa de la mascara. El mas barato:

       P_x(t)  = potencia del canal de referencia en la banda de voz
       nmin(t) = min-tracking de P_x (baja instantaneo, sube `guard_rise`/frame)
       snr(t)  = P_x / nmin                        <- proxy de actividad
       mass(t) = masa de la mascara en la misma banda

   Si el canal crudo dice "hay algo bastante arriba del piso" y la mascara dice
   "no hay voz" de forma sostenida, se sospecha colapso y se ABRE el lazo
   (b = 1: la red vuelve a mirar x_ref) durante `guard_hold` frames. Con la
   mascara sana, Phi_NN se limpia sola en ~1/(1-alpha) frames.

   Los FALSOS POSITIVOS son benignos por construccion: un tramo de interferente
   puro tambien dispara (energia alta, mascara baja) y lo unico que pasa es que
   durante ese tramo la mascara sale del canal crudo, o sea del sistema base.

3. `fuse` -- LAS DOS MASCARAS, FUNDIDAS (y no cuesta lo que parece).
   Un segundo DTLN sobre el canal de referencia, y las dos mascaras crudas se
   combinan con una regla SIN PARAMETROS. La medicion de costo dice que esto no
   es la opcion cara: la etapa 1 del DTLN son 0.009 ms/frame contra los 1.554 ms
   del eigh de Phi_SS que este esquema elimina -- 170 veces menos. Lo caro era
   el autovector, no la red.

   Las dos fuentes tienen modos de falla COMPLEMENTARIOS, que es lo que hace
   que valga la pena fundirlas y no elegir una:

     mascara del canal de referencia : buena con iSIR ALTO (el target ya se ve
         solo); mala con iSIR bajo, donde el target esta enterrado.
     mascara de la salida del BF     : buena con iSIR BAJO (el beamformer
         limpia el interferente); con iSIR alto la senal es tan limpia que la
         mascara SATURA y, con sharpen_exp=8, la rama de RUIDO se queda sin
         masa (m_n=(1-m)^8 -> 0) y Phi_NN queda mal condicionada.

   Medido: OFB gana por debajo de iSIR ~5 y pierde por arriba de ~10, que es
   exactamente el cruce que predice esa complementariedad.

   Reglas implementadas (a, b = mascaras CRUDAS de referencia y de salida):
     "max"   1 - (1-a)(1-b) al limite duro: protege contra la auto-cancelacion
             (el lazo nunca puede NEGAR voz que la rama de atras ve), pero
             AGRAVA la saturacion de iSIR alto.
     "min"   consenso para declarar voz. Al reves: arregla la saturacion y
             desprotege la auto-cancelacion.
     "mean"  promedio aritmetico. El compromiso ingenuo.
     "gmean" media geometrica sqrt(ab) = promedio en el dominio log.
     "bayes" SUMA DE LOG-ODDS: p = ab / (ab + (1-a)(1-b)). Es la fusion de dos
             detectores independientes de P(voz), y es la unica de la lista que
             sale de un modelo en vez de una heuristica. Propiedad util: una
             mascara INSEGURA (p ~ 0.5, log-odds ~ 0) no aporta nada y manda la
             segura -- que es el comportamiento que se busca. Contra: si una se
             equivoca CON CONFIANZA, la arrastra.
     "nor"   noisy-OR, 1 - (1-a)(1-b): la version blanda de "max", protectora
             por construccion.
     "back"  CONTROL: se descarta la mascara de la salida y queda solo la de
             atras. Con fuse_src="median" es la forma clasica de la literatura
             (una mascara por canal, mediana sobre canales); con "ref" es la
             mascara del mic de referencia sola. Sin este control no se puede
             saber si fundir aporta o solo interpola.

   `guard='dual'` queda como alias historico de `fuse="max"`.

Ademas, `mask_floor` pone un piso a la rama de senal (m_s <- f + (1-f) m_s), lo
que garantiza que Phi_XX nunca deja de acumular: es una defensa continua sobre
la ESTADISTICA en vez de sobre la senal, y sale gratis.

ARRANQUE
--------
Los pesos arrancan en e_ref (no en cero como el nucleo sobre estado nulo), asi
que Y(0) = x_ref y el primer frame que ve la red es el canal crudo: el mismo
bootstrap que el lazo ciego, sin necesidad de ninguna pasada previa. `warmup`
mantiene b = 1 los primeros frames si se quiere anclar mas tiempo.
"""

import numpy as np

from .blind_feedback import DTLNStream, SoudenSubtractCore


def output_feedback_stft(X_stft, model_path, nperseg, ref_mic_idx=None,
                         sharpen_exp=8.0, alpha=0.99, min_loading=1e-9,
                         mu=0.0, lambda_floor=1e-3, psd_project=True,
                         ban=False, smooth=None, mask_warp=None,
                         block_update=1, leak=0.05, leak_smooth=0.5,
                         warmup=0, mask_floor=0.0, guard=None,
                         guard_bins=None, guard_snr_db=6.0, guard_mass=0.08,
                         guard_smooth=0.9, guard_hold=64, guard_rise=1.0005,
                         fuse=None, fuse_src="ref", fuse_model_path=None,
                         guard_model_path=None, poison=None, stage2=None,
                         model2_path=None, hop=None, return_diag=False,
                         progress=True):
    """
    Corre el lazo de salida frame a frame sobre una STFT de analisis RECTANGULAR.

    Args:
        X_stft: (K, T, M), ventana rectangular (el bloque del DTLN y el frame de
            la STFT son las mismas muestras; ver `blind_feedback`).
        block_update: P >= 1. Periodo de recalculo de los pesos. El frame t se
            filtra SIEMPRE con pesos anteriores -- es la condicion que hace
            posible este esquema, no una optimizacion opcional.
        leak: b de la fuga (0 = lazo puro, sin defensa; 1 = sistema base).
        leak_smooth: suavizado de b (evita saltos bruscos en la entrada de la
            red, que tiene estado LSTM).
        warmup: frames iniciales con b = 1.
        mask_floor: piso de la rama de senal.
        fuse: None | 'max' | 'min' | 'mean' | 'gmean' | 'bayes' | 'nor' | 'back'.
            Funde la mascara de la salida con la de un segundo DTLN sobre el
            canal de referencia. Ninguna de las reglas tiene parametros.
        fuse_src: de donde sale la mascara de ATRAS. "ref" (default) = un solo
            DTLN sobre el canal de referencia. "median" = un DTLN POR CANAL y la
            MEDIANA sobre los M canales, que es la forma clasica de la
            literatura de mask-beamforming. La mediana reduce la VARIANZA del
            estimador, no el SNR de entrada (los M canales ven la misma mezcla
            con casi el mismo SIR), asi que deberia mejorar la mitad de atras de
            la fusion -- la que manda con iSIR alto. Cuesta M invokes por frame:
            medido, 361 k MACs int8 y 10.7 us por canal en x86, o sea 86 us para
            M=8 (1.1% del periodo de hop). Los M canales comparten los MISMOS
            pesos, asi que el tráfico de memoria no se multiplica por M.
        fuse_model_path: .tflite de la segunda red (default: el mismo).
        guard: None | 'snr' | 'dual' | 'both'.
        guard_bins: (K,) bool, banda donde se miden el proxy y la masa.
        guard_model_path: .tflite de la segunda red (default: el mismo).
        stage2: None (default) o "pf". Con "pf" el post-filtro deja de ser la
            ganancia espectral G y pasa a ser el DTLN COMPLETO aplicado a la
            salida del beamformer: el bloque enmascarado por la etapa 1 se
            vuelve al tiempo, lo procesa la etapa 2 y se reconstruye por
            overlap-add, igual que `apply_dtln_post_tflite_realtime`. La etapa 1
            ya se estaba corriendo para la mascara, asi que lo unico que se
            agrega es la etapa 2 -- y se aplica EXACTAMENTE sobre la senal en la
            que se estimo, que es la ventaja estructural de este esquema.
            `smooth` sigue valiendo: la ganancia que entra a la etapa 2 es
            G = smooth + (1-smooth) m_raw (con smooth=None, el DTLN tal cual).
            En este modo la salida del sistema es la senal de TIEMPO devuelta
            como tercer elemento; Y_stft queda como el espectro SIN post-filtro.
        model2_path: .tflite de la segunda etapa (obligatorio con stage2).
        hop: salto de la STFT en muestras (default nperseg // 4). Solo lo usa el
            overlap-add de la etapa 2.
        poison: (t0, t1) -- BANCO DE ESTRES, no una opcion de produccion. En
            esos frames se fuerza la mascara a "no hay voz" (m_s -> 0, m_n -> 1)
            EN LA ESTADISTICA, o sea se mete al lazo en el estado que se quiere
            evitar. Lo que se mide es lo de despues de t1: si el estado es
            absorbente la masa de la mascara no vuelve, y si hay mecanismo de
            recuperacion vuelve. Es la unica forma de responder la pregunta sin
            esperar a que la escena tenga la suerte de disparar el modo de falla.

    Returns:
        (Y_stft (K,T), weights (K,T,M))  -- o (..., diag) con return_diag=True.
        Con stage2 se agrega la senal de tiempo ANTES del diag:
        (Y_stft, weights, y_time[, diag]).
    """
    X_stft = np.asarray(X_stft)
    K, T, M = X_stft.shape
    ref = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
    if guard not in (None, "snr", "dual", "both"):
        raise ValueError(f"guard desconocido: {guard!r} (None|'snr'|'dual'|'both')")
    _FUSE = (None, "max", "min", "mean", "gmean", "bayes", "nor", "back")
    if fuse not in _FUSE:
        raise ValueError(f"fuse desconocido: {fuse!r} {_FUSE}")
    if fuse is None and guard in ("dual", "both"):
        fuse = "max"                      # alias historico
    if stage2 not in (None, "pf"):
        raise ValueError(f"stage2 desconocido: {stage2!r} (None|'pf')")
    if stage2 is not None and model2_path is None:
        raise ValueError("stage2='pf' necesita model2_path.")
    p = float(sharpen_exp)
    P = max(1, int(block_update))
    b_nom = float(np.clip(leak, 0.0, 1.0))
    gb = np.ones(K, dtype=bool) if guard_bins is None else np.asarray(guard_bins, bool)

    core = SoudenSubtractCore(K, M, ref, alpha=alpha, min_loading=min_loading,
                              mu=mu, lambda_floor=lambda_floor,
                              psd_project=psd_project, ban=ban)
    dtln = DTLNStream(model_path, model2_path=model2_path if stage2 else None)
    _mpath = fuse_model_path or guard_model_path or model_path
    if fuse is None:
        dtln_ref = None
    elif fuse_src == "median":
        # Un interprete POR CANAL: cada uno lleva su propio estado LSTM.
        dtln_ref = [DTLNStream(_mpath) for _ in range(M)]
    elif fuse_src == "ref":
        dtln_ref = DTLNStream(_mpath)
    else:
        raise ValueError(f"fuse_src desconocido: {fuse_src!r} ('ref'|'median')")
    use_snr = guard in ("snr", "both")

    # Arranque: w = e_ref  ->  Y(0) = x_ref (el canal crudo, como el bootstrap
    # del lazo ciego). El nucleo sobre estado nulo daria w = 0, que dejaria a la
    # red sin nada que mirar en el primer frame.
    w_hold = np.zeros((K, M), dtype=np.complex128)
    w_hold[:, ref] = 1.0

    Y_stft = np.zeros((K, T), dtype=np.complex128)
    W_out = np.zeros((K, T, M), dtype=np.complex128)
    diag = ({k: np.zeros(T) for k in ("beta", "snr_db", "mass", "open")}
            if return_diag else None)
    if return_diag:
        diag["m_raw"] = np.zeros((K, T))

    t_poison = (-1, -1) if poison is None else (int(poison[0]), int(poison[1]))

    # Overlap-add de la etapa 2, identico al del DTLN original (suma simple, sin
    # ventana de sintesis). El bloque i del DTLN es el frame i-1, asi que el
    # bloque del frame t se escribe en la posicion (t+1)*hop.
    S_hop = int(nperseg // 4 if hop is None else hop)
    y_ola = ola_buf = None
    if stage2 is not None:
        y_ola = np.zeros((T + 2) * S_hop + nperseg, dtype=np.float64)
        ola_buf = np.zeros(nperseg, dtype=np.float64)

    b = 1.0
    nmin = None
    snr_sm = 1.0
    mass_sm = 1.0
    hold_left = 0

    for t in range(T):
        if progress and (t % 32 == 0 or t == T - 1):
            print(f"\r  [outfb P={P}] frame {t+1}/{T}", end="")
        X_frame = X_stft[:, t, :]

        # --- CAMINO CRITICO: la unica combinacion lineal del sistema ---------
        Y = np.einsum("fm,fm->f", w_hold.conj(), X_frame)

        # --- entrada de la red: salida del BF + fuga del canal de referencia -
        b_tgt = 1.0 if (t < warmup or hold_left > 0) else b_nom
        b = leak_smooth * b + (1.0 - leak_smooth) * b_tgt
        assert np.isscalar(b) or np.ndim(b) == 0, (
            "el coeficiente de fuga tiene que seguir siendo ESCALAR: si alguna "
            "rama lo pisa con un vector, la fuga pasa a ser por bin sin que se "
            "note (y rompe return_diag).")
        Y_mask = (1.0 - b) * Y + b * X_frame[:, ref]

        m_raw = np.clip(np.asarray(dtln.step(np.abs(nperseg * Y_mask)),
                                   dtype=np.float64), 0.0, 1.0)
        if dtln_ref is not None:
            if isinstance(dtln_ref, list):
                # Mediana sobre los canales (la forma clasica). La mediana --y no
                # la media-- porque un canal con un nulo espacial en la fuente da
                # una mascara arbitrariamente mala, y la mediana la ignora.
                a = np.median(
                    [np.asarray(d.step(np.abs(nperseg * X_frame[:, m])),
                                dtype=np.float64)
                     for m, d in enumerate(dtln_ref)], axis=0)
                a = np.clip(a, 0.0, 1.0)
            else:
                a = np.clip(np.asarray(
                    dtln_ref.step(np.abs(nperseg * X_frame[:, ref])),
                    dtype=np.float64), 0.0, 1.0)
            # OJO: aca NO se puede usar `b`, que es el coeficiente ESCALAR de
            # fuga y sobrevive de un frame al siguiente (b = leak_smooth * b +
            # ...). Pisarlo con la mascara convertia la fuga en un vector por
            # bin desde el frame 2. m_out = mascara de la SALIDA, a = la de atras.
            m_out = m_raw
            if fuse == "back":
                m_raw = a
            elif fuse == "max":
                m_raw = np.maximum(a, m_out)
            elif fuse == "min":
                m_raw = np.minimum(a, m_out)
            elif fuse == "mean":
                m_raw = 0.5 * (a + m_out)
            elif fuse == "gmean":
                m_raw = np.sqrt(a * m_out)
            elif fuse == "nor":
                m_raw = 1.0 - (1.0 - a) * (1.0 - m_out)
            else:                                    # "bayes": suma de log-odds
                e = 1e-6
                aa, oo = np.clip(a, e, 1 - e), np.clip(m_out, e, 1 - e)
                num = aa * oo
                m_raw = num / (num + (1.0 - aa) * (1.0 - oo))

        if mask_warp is None:
            m_s, m_n = m_raw ** p, (1.0 - m_raw) ** p
        else:
            a_s, b_s, a_n, b_n = mask_warp
            m_s = np.clip(a_s * m_raw + b_s, 1e-4, 1.0)
            m_n = np.clip(a_n * (1.0 - m_raw) + b_n, 1e-4, 1.0)
        if mask_floor > 0.0:
            m_s = mask_floor + (1.0 - mask_floor) * m_s
        if t_poison[0] <= t < t_poison[1]:
            # Envenenamiento deliberado: la voz entra entera a la rama de ruido.
            m_s, m_n = np.full(K, 1e-6), np.ones(K)

        if stage2 is not None:
            # El DTLN entero sobre la salida: ganancia de la etapa 1 (relajada
            # por `smooth` si esta), vuelta al tiempo, etapa 2 y overlap-add.
            G = m_raw if smooth is None else (smooth + (1.0 - smooth) * m_raw)
            blk = np.fft.irfft(nperseg * Y * G, n=nperseg)
            ob = dtln.step2(blk)
            ola_buf[:-S_hop] = ola_buf[S_hop:]
            ola_buf[-S_hop:] = 0.0
            ola_buf += ob
            i0 = (t + 1) * S_hop
            y_ola[i0:i0 + S_hop] = ola_buf[:S_hop]
        elif smooth is not None:
            Y = Y * (smooth + (1.0 - smooth) * m_raw)
        Y_stft[:, t] = Y
        W_out[:, t, :] = w_hold

        # --- PERRO GUARDIAN: estadistico independiente de la mascara ---------
        if use_snr:
            P_x = float(np.mean(np.abs(X_frame[gb, ref]) ** 2)) + 1e-20
            nmin = P_x if nmin is None else min(P_x, nmin * guard_rise)
            snr = P_x / (nmin + 1e-20)
            mass = float(np.mean(m_raw[gb]))
            snr_sm = guard_smooth * snr_sm + (1.0 - guard_smooth) * snr
            mass_sm = guard_smooth * mass_sm + (1.0 - guard_smooth) * mass
            if hold_left > 0:
                hold_left -= 1
            elif (10.0 * np.log10(snr_sm) > guard_snr_db) and (mass_sm < guard_mass):
                hold_left = int(guard_hold)
            if diag is not None:
                diag["snr_db"][t] = 10.0 * np.log10(snr_sm)
                diag["mass"][t] = mass_sm
        if diag is not None:
            diag["beta"][t] = b
            diag["open"][t] = float(hold_left > 0)
            diag["m_raw"][:, t] = m_raw

        # --- FUERA DEL CAMINO CRITICO ---------------------------------------
        core.update(X_frame, m_s, m_n)
        if t % P == 0:
            w_hold = core.solve()

    if progress:
        print()
    if stage2 is not None:
        return (Y_stft, W_out, y_ola, diag) if return_diag else (Y_stft, W_out, y_ola)
    return (Y_stft, W_out, diag) if return_diag else (Y_stft, W_out)
