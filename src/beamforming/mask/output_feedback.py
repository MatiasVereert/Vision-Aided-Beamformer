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
     "out"   la mascara de la SALIDA sola. NO es igual a fuse=None: la segunda
             red igual corre, asi que `pf_mask` sigue teniendo las dos mascaras
             para elegir. Es el control que hace completo el barrido.
     "isir"  la interpolacion agendada, igual que en `pf_mask` pero con su
             propio cruce (`fuse_isir`): el SCM y el post-filtro sufren la
             misma saturacion pero no necesariamente en el mismo iSIR.
     "back"  CONTROL: se descarta la mascara de la salida y queda solo la de
             atras. Con fuse_src="median" es la forma clasica de la literatura
             (una mascara por canal, mediana sobre canales); con "ref" es la
             mascara del mic de referencia sola. Sin este control no se puede
             saber si fundir aporta o solo interpola.

   `guard='dual'` queda como alias historico de `fuse="max"`.

   OJO CON EL BARRIDO VIEJO DE `fuse` (el que eligio "mean"): esta CONFUNDIDO.
   Se corrio cuando la mascara fundida alimentaba el SCM *y* el post-filtro, asi
   que "gana mean" puede significar "mean es la mejor para el POST-FILTRO" --
   que es justo lo que el punto 4 muestra que era falso. Con `pf_mask` el
   barrido hay que rehacerlo con el post-filtro FIJO; recien ahi mide el SCM.

4. `pf_mask` -- DESACOPLA LA MASCARA DEL POST-FILTRO DE LA DEL SCM.
   Hasta aca la mascara fundida hacia las dos cosas: alimentar el SCM y, si hay
   `smooth`, multiplicar la salida. Son dos trabajos DISTINTOS y el optimo no
   tiene por que ser el mismo:

     el SCM   estima estadisticas de LARGO PLAZO y necesita masa en las DOS
              ramas -- por eso le duele que `m_out` sature con iSIR alto
              ((1-m)^p -> 0 deja Phi_NN mal condicionada) y por eso fundir con
              la mascara de atras lo ayuda.
     el PF    multiplica `Y`, la salida del beamformer, frame a frame. La unica
              mascara que DESCRIBE esa senal es `m_out`: se estimo sobre ella.
              La de atras describe `x_ref`, que tiene iSIR peor, asi que meterla
              en la ganancia le vuelve a cobrar al target el ruido que el
              beamformer ya saco -- sobre-supresion pura. Y la saturacion, que
              es el argumento a favor de fundir, aca no molesta: si `m_out`
              satura, G -> 1 y el PF no hace nada, que es lo correcto cuando no
              queda nada que sacar.

   `pf_mask` toma las mismas reglas que `fuse` mas "out" (la mascara de la
   salida cruda). None = comportamiento historico: el PF usa la fundida.

   MEDIDO (MIRD, rt60=0.61, interf 45deg, Delta PESQ; mean/out/back):
     iSIR -5   0.491 / 0.507 / 0.463     gana "out"
     iSIR  0   0.753 / 0.736 / 0.750     empate
     iSIR  5   0.884 / 0.839 / 0.911     gana "back"
     iSIR 10   0.911 / 0.845 / 0.968     gana "back"
     iSIR 15   0.792 / 0.707 / 0.873     gana "back"

   O sea: la hipotesis de arriba ("el PF quiere la mascara que describe la
   senal que multiplica") se cumple SOLO con iSIR bajo, y se da vuelta con iSIR
   alto. La explicacion no es la sobre-supresion sino el DOMINIO DE
   ENTRENAMIENTO de la red: el DTLN se entreno sobre senales de MICROFONO. Con
   iSIR bajo la salida del beamformer todavia esta sucia y cae dentro de ese
   dominio (y encima con mejor SNR que el mic, asi que la mascara es mejor).
   Con iSIR alto la salida es tan limpia que se va del dominio, la mascara
   satura, G -> 1 y el post-filtro deja de existir: "out" con iSIR 15 no es un
   PF peor, es casi NO tener PF. La mascara del mic crudo, en cambio, sigue en
   dominio siempre -- por eso "back" gana 4 de 5 celdas y es el default
   razonable, no "mean" (la media es la peor de las tres en 3 de 5 celdas: no
   interpola entre dos aciertos, promedia el acierto con el error).

5. `pf_mask="isir"` -- LA INTERPOLACION, agendada por el iSIR estimado.
   m_pf = (1-c) m_out + c a,  c = sigmoide((iSIR_est - centro) / ancho),
   con c -> 1 (mascara de atras pura) cuando el iSIR sube. Recupera la unica
   celda donde "back" pierde (iSIR -5) sin resignar las otras cuatro.

   El iSIR se estima CIEGO y con constante de tiempo larga (`pf_isir_alpha`),
   porque el cruce es una propiedad de la ESCENA y no del frame: sale de la
   mascara de atras sobre el canal de referencia, que ya se calcula para la
   fusion. El estimador esta SESGADO en valor absoluto -- la mascara del DTLN
   no separa target de interferente -- pero es monotono en el iSIR real, que es
   todo lo que el sigmoide necesita. Ver `ISIRTracker`: `isir_est` elige entre
   el estimador historico ('sum') y el que menos se mueve con el locutor
   ('band'), y `isir_calib` invierte la recta medida para que `pf_isir` y
   `pf_isir_db` esten en dB de iSIR REAL en vez del dominio crudo del
   estimador. Sin calibrar, el oraculo y el ciego NO corren la misma agenda y
   la comparacion entre los dos mide dos cosas mezcladas.

   Alternativa medida y descartada: manejar `c` con un detector de SATURACION
   de la mascara (la fraccion de bins con m > 0.9), que ataca la causa en vez
   de un correlato. Es monotono (Spearman 0.97) pero su dispersion entre
   escenas, llevada a dB reales, es PEOR que la de 'band' (3.1 contra 1.9 dB),
   asi que no compra nada y encima cambia las unidades de la agenda.

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

_FUSE_RULES = ("max", "min", "mean", "gmean", "bayes", "nor", "back", "out")

# Banda donde la mascara del DTLN es informativa. Abajo de 300 Hz manda el
# rumble y arriba de 3400 Hz casi no queda energia: los dos extremos le meten
# varianza al estimador de iSIR sin aportar informacion.
ISIR_BAND_HZ = (300.0, 3400.0)


class ISIRTracker:
    """
    ESTIMADOR CIEGO DEL iSIR DE LA ESCENA, a partir de la mascara del DTLN sobre
    el canal de referencia. Es lo que hace autonoma la agenda de `pf_mask='isir'`
    (y la de `fuse='isir'`): sin el, el cruce del sigmoide habria que pasarselo
    de afuera, o sea saber el iSIR de antemano.

    La idea es la unica disponible sin informacion nueva: la mascara `a` ya
    reparte la energia del canal de referencia en "voz" y "resto", asi que el
    cociente de esas dos energias es un correlato del iSIR. No hace falta que
    sea insesgado -- hace falta que sea MONOTONO, porque el sesgo se absorbe en
    la calibracion afin (`calib`).

    MODOS
    -----
    "sum"  (LEGACY) s = <sum_k a Px>, n = <sum_k (1-a) Px>, hat = 10log10(s/n).
           Es el estimador con el que se calibro `pf_isir=(0.0, 2.0)`. Su
           problema es el CICLO DE TRABAJO del locutor: en los silencios `a`->0,
           asi que `s` no acumula y `n` si, y el valor converge a algo del orden
           de (duty * P_target)/P_ruido. Medido: entre dos locutores el mismo
           iSIR de +20 dB da 5.9 dB o 14.6 dB de estimador.
    "band" (DEFAULT NUEVO) el mismo cociente pero POR BIN OCUPADO -- las
           potencias se dividen por la masa de mascara acumulada -- y
           restringido a `ISIR_BAND_HZ`. La normalizacion por masa cancela el
           ciclo de trabajo a primer orden; la banda saca la varianza de los
           extremos del espectro.

    MEDIDO (tests/ofb_isir_estimator_check.py, 8 escenas MIRD no-voz: 2 salas x
    2 angulos x 2 locutores, iSIR de -10 a +20 dB), dispersion entre escenas a
    iSIR fijo, expresada en dB REALES (o sea sigma/g, que es cuanto se corre el
    cruce de la agenda al cambiar de sala o de locutor):

        sum   2.6 dB      band  1.9 dB      (ancho del sigmoide: 3.5 dB)

    y la monotonia por escena es perfecta (Spearman 1.00) en los dos, con
    rho global 0.96 (sum) contra 0.98 (band). O sea que los dos sirven y `band`
    es el que menos se mueve con el locutor -- que es la nuisance que un sistema
    desplegado ve todo el tiempo.

    LIMITE FUNDAMENTAL (no es de calibracion): la mascara del DTLN detecta VOZ,
    no el TARGET. Con un interferente HABLADO la energia del interferente cae en
    la rama de senal y el iSIR deja de ser observable por esta via: medido,
    Spearman ~0 y hasta negativo por escena. El dano esta acotado (la agenda
    elige un punto arbitrario de una interpolacion cuyos dos extremos son
    mascaras razonables) pero la ganancia de la agenda se pierde. Sacar el iSIR
    en ese caso pide informacion ESPACIAL, que este estimador no mira.

    COSTO
    -----
    Nada. La mascara `a` ya esta calculada (la usa la fusion), asi que lo unico
    que agrega el estimador por frame son, sobre los 99 bins de la banda: UN
    producto interno (99 MAC), DOS reducciones (la rama de ruido sale por
    identidad, ver `update`), una recursion de 4 taps, una division y un log10.
    Unos 300 MAC y un logaritmo.

    Medido en x86 (K=257, M=8, hop=128 a 16 kHz): 5.3 us por frame, contra
    2135 us del core de Souden y 8.7 us de cada invoke del DTLN. Es el 0.24%
    del frame y el 0.07% del periodo de hop. La agenda del post-filtro sale,
    para todo efecto practico, gratis.

    CALIBRACION
    -----------
    `calib=(g, b)` invierte el ajuste medido hat = g*iSIR + b, asi que `value()`
    devuelve dB de iSIR REAL y el centro/ancho del sigmoide se expresan en dB
    reales. Eso es lo que hace comparable el modo oraculo con el ciego: los dos
    hablan el mismo idioma. `calib=None` devuelve el estimador CRUDO (el
    comportamiento historico, donde `pf_isir` estaba en el dominio del
    estimador).

    OJO: (g, b) dependen de `alpha` -- mas suavizado comprime menos y sube la
    pendiente. Los defaults de `NM_MVDR_OFB_AUTO` van juntos; cambiar uno sin
    el otro descalibra la agenda.
    """

    def __init__(self, mode="band", alpha=0.998, calib=None, bins=None):
        if mode not in ("sum", "band"):
            raise ValueError(f"modo de estimador desconocido: {mode!r} ('sum'|'band')")
        self.mode = mode
        self.alpha = float(np.clip(alpha, 0.0, 0.9999))
        self.calib = None if calib is None else (float(calib[0]), float(calib[1]))
        if self.calib is not None and abs(self.calib[0]) < 1e-6:
            raise ValueError("la pendiente de la calibracion no puede ser 0.")
        # `bins` acota la banda del modo 'band'. None = todo el espectro (el que
        # llama tiene que pasarlo; el core lo arma con ISIR_BAND_HZ). Una banda
        # CONTIGUA (que es lo que sale de un par de frecuencias de corte) se
        # guarda como `slice`: indexar con booleanos copia el arreglo, y aca la
        # copia es la mitad del costo.
        self.bins = None
        if bins is not None:
            if isinstance(bins, slice):
                self.bins = bins
            else:
                idx = np.flatnonzero(np.asarray(bins, dtype=bool))
                if not len(idx):
                    raise ValueError("la banda del estimador quedo vacia.")
                self.bins = (slice(int(idx[0]), int(idx[-1]) + 1)
                             if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
                             else np.asarray(bins, dtype=bool))
        self.state = None                  # (P_senal, masa_senal, P_ruido, masa_ruido)

    def update(self, a, Px):
        """
        Un frame: `a` (K,) mascara del canal de referencia, `Px` (K,) su
        densidad de potencia |X_ref|^2. Devuelve el iSIR estimado en dB (real si
        hay `calib`, crudo si no).

        La rama de RUIDO no se calcula: sale por identidad de la de senal,

            sum (1-a) Px = sum Px - sum a Px       sum (1-a) = Kb - sum a

        asi que el frame son UN producto interno y DOS reducciones, sin
        materializar (1-a) ni los dos productos elemento a elemento. Es 4 veces
        mas rapido que la forma directa y, en un puerto a hardware, es tambien
        la forma que menos memoria intermedia pide.
        """
        if self.mode == "sum" or self.bins is None:
            ab, pb = a, Px
        else:
            ab, pb = a[self.bins], Px[self.bins]
        s = float(ab @ pb)                     # sum a Px
        ms = float(ab.sum())                   # masa de la rama de senal
        if self.mode == "sum":
            # LEGACY: sin normalizar por masa (las masas quedan en 1).
            v = np.array([s, 1.0, float(pb.sum()) - s, 1.0])
        else:
            v = np.array([s, ms, float(pb.sum()) - s, float(len(ab)) - ms])
        self.state = v if self.state is None else (self.alpha * self.state +
                                                   (1.0 - self.alpha) * v)
        s, ms, n, mn = self.state
        hat = 10.0 * np.log10(((s / max(ms, 1e-20)) + 1e-20) /
                              ((n / max(mn, 1e-20)) + 1e-20))
        if self.calib is None:
            return float(hat)
        g, b = self.calib
        return float((hat - b) / g)


def _fuse_masks(rule, a, m_out):
    """
    Combina las dos mascaras CRUDAS con una regla sin parametros.
    `a` = mascara de ATRAS (canal de referencia, o mediana sobre canales),
    `m_out` = mascara de la SALIDA del beamformer. Ver el docstring del modulo.
    """
    if rule == "out":
        return m_out
    if rule == "back":
        return a
    if rule == "max":
        return np.maximum(a, m_out)
    if rule == "min":
        return np.minimum(a, m_out)
    if rule == "mean":
        return 0.5 * (a + m_out)
    if rule == "gmean":
        return np.sqrt(a * m_out)
    if rule == "nor":
        return 1.0 - (1.0 - a) * (1.0 - m_out)
    if rule == "bayes":                              # suma de log-odds
        e = 1e-6
        aa, oo = np.clip(a, e, 1 - e), np.clip(m_out, e, 1 - e)
        num = aa * oo
        return num / (num + (1.0 - aa) * (1.0 - oo))
    raise ValueError(f"regla de fusion desconocida: {rule!r} {_FUSE_RULES}")


def output_feedback_stft(X_stft, model_path, nperseg, ref_mic_idx=None,
                         sharpen_exp=8.0, alpha=0.99, min_loading=1e-9,
                         mu=0.0, lambda_floor=1e-3, psd_project=True,
                         ban=False, smooth=None, mask_warp=None,
                         block_update=1, leak=0.05, leak_smooth=0.5,
                         warmup=0, mask_floor=0.0, guard=None,
                         guard_bins=None, guard_snr_db=6.0, guard_mass=0.08,
                         guard_smooth=0.9, guard_hold=64, guard_rise=1.0005,
                         fuse=None, fuse_src="ref", fuse_model_path=None,
                         fuse_isir=(9.0, 3.0),
                         pf_mask=None, pf_isir=(0.0, 2.0), pf_isir_alpha=0.995,
                         pf_isir_db=None, isir_est="sum", isir_calib=None,
                         isir_freqs=None,
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
        fuse_isir: (centro_db, ancho_db) de la agenda del SCM (`fuse='isir'`).
            El default (9.0, 3.0) sale del cruce viejo, medido cuando la
            mascara hacia los dos trabajos: es un PUNTO DE PARTIDA, no una
            calibracion. El cruce real del SCM sale de comparar fuse='out' vs
            fuse='back' celda por celda con el post-filtro ya fijo.
        pf_mask: que mascara alimenta el POST-FILTRO (`smooth` / la ganancia que
            entra a la etapa 2), independientemente de la que alimenta el SCM.
            None (default) = la misma que el SCM. 'out' = la mascara cruda de la
            salida del BF, que es la que describe la senal que el PF multiplica;
            tambien acepta cualquier regla de `fuse` ('back', 'max', ...) para
            barrer combinaciones SCM/PF cruzadas. Solo tiene efecto con
            `fuse` activo (sin segunda red hay una sola mascara). 'isir'
            interpola entre las dos segun el iSIR estimado (ver el modulo).
        pf_isir: (centro_db, ancho_db) del sigmoide de `pf_mask='isir'`. OJO:
            se calibra contra `diag["isir_hat"]` (el estimador, sesgado), no
            contra el iSIR nominal de la escena.
        pf_isir_alpha: suavizado del estimador de iSIR. Largo a proposito
            (0.995 ~ 1.6 s con hop de 8 ms): el cruce es una propiedad de la
            escena, no del frame.
        pf_isir_db: ORACULO -- fija el iSIR en vez de estimarlo. Solo para
            medir el techo de la agenda; no es una configuracion de produccion.
            Con `isir_calib` puesto esta en dB REALES y es directamente
            comparable con el estimador (que devuelve dB reales); sin calibrar
            queda en el dominio crudo del estimador, que NO es el mismo -- por
            eso el oraculo viejo y el ciego viejo no corrian la misma agenda.
        isir_est: 'sum' (legacy) | 'band'. Ver `ISIRTracker`.
        isir_calib: (g, b) del ajuste hat = g*iSIR + b. Puesto, el estimador
            devuelve dB REALES y `pf_isir`/`fuse_isir`/`pf_isir_db` se expresan
            en dB reales. None = dominio crudo del estimador (historico).
        isir_freqs: (K,) frecuencias de los bins, para acotar la banda del modo
            'band' a `ISIR_BAND_HZ`. None = banda completa.
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
            G = smooth + (1-smooth) m_pf (con smooth=None, el DTLN tal cual).
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
    # 'out' NO es lo mismo que fuse=None: apaga la fusion para el SCM pero deja
    # la segunda red corriendo, que es lo que necesita `pf_mask` para tener las
    # dos mascaras. Sin el, "SCM con la mascara de la salida sola + post-filtro
    # agendado" no se puede expresar y el barrido de `fuse` queda incompleto.
    _FUSE = (None,) + _FUSE_RULES + ("isir", "cov")
    if fuse not in _FUSE:
        raise ValueError(f"fuse desconocido: {fuse!r} {_FUSE}")
    if fuse == "cov" and mask_warp is not None:
        raise ValueError("fuse='cov' fusiona las RAMAS ya afiladas, asi que no "
                         "se combina con mask_warp (que reemplaza el afilado).")
    fus_c0, fus_w = (float(fuse_isir[0]), float(fuse_isir[1]))
    if fus_w <= 0.0:
        raise ValueError(f"el ancho del sigmoide tiene que ser > 0: {fus_w}")
    _PF = _FUSE_RULES + ("isir",)
    if pf_mask is not None and pf_mask not in _PF:
        raise ValueError(f"pf_mask desconocido: {pf_mask!r} (None|{_PF})")
    if pf_mask == "isir" and fuse is None:
        raise ValueError("pf_mask='isir' necesita la segunda red (fuse=...): "
                         "interpola ENTRE las dos mascaras.")
    isir_c0, isir_w = (float(pf_isir[0]), float(pf_isir[1]))
    if isir_w <= 0.0:
        raise ValueError(f"el ancho del sigmoide tiene que ser > 0: {isir_w}")
    isir_bins = None
    if isir_freqs is not None:
        f = np.asarray(isir_freqs, dtype=np.float64)
        isir_bins = (f >= ISIR_BAND_HZ[0]) & (f <= ISIR_BAND_HZ[1])
    isir_tracker = ISIRTracker(mode=isir_est, alpha=pf_isir_alpha,
                               calib=isir_calib, bins=isir_bins)
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
        diag["m_pf"] = np.zeros((K, T))
        diag["isir_hat"] = np.zeros(T)     # el que consume la agenda
        diag["isir_est"] = np.zeros(T)     # el del estimador (aunque haya oraculo)
        diag["c_pf"] = np.zeros(T)
        diag["c_scm"] = np.zeros(T)

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
    c_pf = c_scm = 0.0                # peso de la mascara de ATRAS (PF / SCM)
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
        m_branch = None        # ramas ya afiladas, si la fusion es en covarianza
        # Sin segunda red hay una sola mascara y el PF usa esa. Con `fuse`
        # activo, el default sigue siendo la FUNDIDA (comportamiento historico)
        # y `pf_mask` es lo que las desacopla -- se reasigna abajo, despues de
        # fundir, porque aca `m_raw` todavia es la mascara cruda de la salida.
        m_pf = m_raw
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

            # El iSIR se estima UNA vez por frame y lo comparten las dos
            # agendas (SCM y post-filtro), que tienen el mismo signo -- la
            # mascara de atras manda con iSIR alto -- pero NO el mismo cruce:
            # son dos fallas distintas de la misma saturacion.
            if fuse == "isir" or pf_mask == "isir":
                # El estimador corre SIEMPRE (aunque haya oraculo): es barato y
                # asi el diagnostico deja ver el error del estimador contra el
                # valor verdadero en la misma corrida.
                est = isir_tracker.update(a, np.abs(X_frame[:, ref]) ** 2)
                isir_hat = float(pf_isir_db) if pf_isir_db is not None else est
                if diag is not None:
                    diag["isir_hat"][t] = isir_hat
                    diag["isir_est"][t] = est

            if fuse == "isir":
                c_scm = 1.0 / (1.0 + np.exp(-(isir_hat - fus_c0) / fus_w))
                m_raw = (1.0 - c_scm) * m_out + c_scm * a
            elif fuse == "cov":
                # PROMEDIO EN EL DOMINIO DE LAS COVARIANZAS. Promediar los
                # acumuladores de dos SCM, una por mascara, es identico a
                # promediar las RAMAS YA AFILADAS -- Num y Den son lineales en
                # la mascara, asi que el promedio se puede meter adentro de la
                # suma y no hacen falta dos juegos de acumuladores (que serian
                # otros 261 KB de estado con K=257, M=8):
                #     Phi = Sum a^t (a^p + b^p)/2 R / Sum a^t (a^p + b^p)/2
                # Las dos diferencias con `mean` (que hace ((a+b)/2)^p):
                #  1. en el dominio de la mascara esto es la MEDIA DE POTENCIA
                #     de orden p, que cae entre 'mean' y 'max' -- un max blando.
                #  2. DESACOPLA las ramas: m_n deja de ser (1 - m_s^(1/p))^p y
                #     por convexidad recibe MAS masa, que es exactamente la
                #     falla conocida (Phi_NN sin masa cuando la mascara satura).
                m_branch = (0.5 * (a ** p + m_out ** p),
                            0.5 * ((1.0 - a) ** p + (1.0 - m_out) ** p))
                # Equivalente en el dominio de la mascara: para el diag, el
                # perro guardian y el default del post-filtro.
                m_raw = m_branch[0] ** (1.0 / p)
            else:
                m_raw = _fuse_masks(fuse, a, m_out)

            if pf_mask is None:
                m_pf = m_raw
            elif pf_mask == "isir":
                # c -> 1 (mascara de ATRAS pura) con iSIR alto, que es donde la
                # salida del BF se va del dominio de la red y su mascara satura.
                c_pf = 1.0 / (1.0 + np.exp(-(isir_hat - isir_c0) / isir_w))
                m_pf = (1.0 - c_pf) * m_out + c_pf * a
            else:
                m_pf = _fuse_masks(pf_mask, a, m_out)

        if m_branch is not None:
            m_s, m_n = m_branch          # ya vienen afiladas (fuse='cov')
        elif mask_warp is None:
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
            G = m_pf if smooth is None else (smooth + (1.0 - smooth) * m_pf)
            blk = np.fft.irfft(nperseg * Y * G, n=nperseg)
            ob = dtln.step2(blk)
            ola_buf[:-S_hop] = ola_buf[S_hop:]
            ola_buf[-S_hop:] = 0.0
            ola_buf += ob
            i0 = (t + 1) * S_hop
            y_ola[i0:i0 + S_hop] = ola_buf[:S_hop]
        elif smooth is not None:
            Y = Y * (smooth + (1.0 - smooth) * m_pf)
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
            diag["m_pf"][:, t] = m_pf
            diag["c_pf"][t] = c_pf
            diag["c_scm"][t] = c_scm

        # --- FUERA DEL CAMINO CRITICO ---------------------------------------
        core.update(X_frame, m_s, m_n)
        if t % P == 0:
            w_hold = core.solve()

    if progress:
        print()
    if stage2 is not None:
        return (Y_stft, W_out, y_ola, diag) if return_diag else (Y_stft, W_out, y_ola)
    return (Y_stft, W_out, diag) if return_diag else (Y_stft, W_out)
