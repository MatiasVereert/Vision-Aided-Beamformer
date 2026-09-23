"""
ofb.py -- OUTPUT-FEEDBACK MVDR: LA ARQUITECTURA FINAL, SIN OPCIONES
===================================================================
Este modulo es la CRISTALIZACION de la configuracion que gano todos los
barridos. `output_feedback.py` es el banco de pruebas donde se eligio cada
pieza -- 43 parametros, ocho reglas de fusion, tres defensas, dos caminos de
post-filtro -- y sigue ahi como registro de lo medido. Aca queda UNA sola
cadena, sin un condicional de configuracion, que es la que se porta a C++.

EL LAZO
-------
Un solo beamformer. Como el frame t se filtra con los pesos calculados en t-1,
esos pesos estan listos ANTES de correr la red, asi que la red puede comer
directamente la SALIDA del beamformer:

    Y(t)     = w(t-1)^H x(t)                     <- el unico camino critico
    m_out(t) = DTLN( Y(t) )                      <- mascara de la SALIDA
    m_ref(t) = DTLN( x_ref(t) )                  <- mascara del canal crudo
    w(t)     = Souden( SCM( (m_out + m_ref)/2 ) )     <- SIN sustraccion

y la salida se post-filtra con una mascara que NO es la del SCM:

    c(t)   = sigmoide( (iSIR_est(t) - centro) / ancho )
    m_pf   = (1 - c) m_out + c m_ref
    Y_out  = Y * ( smooth + (1 - smooth) m_pf )

POR QUE DOS MASCARAS Y DOS DESTINOS DISTINTOS
---------------------------------------------
Las dos redes fallan en extremos opuestos del iSIR:

    m_ref  buena con iSIR ALTO (el target ya se ve solo); mala con iSIR bajo,
           donde el target esta enterrado.
    m_out  buena con iSIR BAJO (el beamformer ya limpio el interferente); con
           iSIR alto la salida es tan limpia que se va del dominio de
           entrenamiento de la red, la mascara SATURA y, con sharpen_exp=8, la
           rama de RUIDO se queda sin masa ((1-m)^8 -> 0) y Phi_NN queda mal
           condicionada.

El SCM necesita masa en las DOS ramas, asi que le conviene el promedio de las
dos mascaras y punto (medido contra max/min/gmean/bayes/nor y contra cada una
sola). El POST-FILTRO, en cambio, multiplica `Y` frame a frame: ahi la mejor
mascara depende del iSIR y por eso se INTERPOLA, con el cruce agendado por el
estimador ciego. Medido (Delta PESQ, 20 celdas MIRD):

    iSIR            -5      0      5     10     15   | media
    solo m_out     0.618  0.900  1.039  1.040  0.850 | 0.889
    solo m_ref     0.559  0.898  1.087  1.136  0.964 | 0.929
    interpolado    0.613  0.910  1.084  1.130  0.960 | 0.939

o sea que la agenda se queda con el mejor extremo en cada punto en vez de
promediarlos, y ademas conserva el STOI del extremo `m_out` (0.274 contra 0.264
en iSIR -5), que es donde `m_ref` sobre-suprime.

LO UNICO REGULABLE, Y POR QUE
-----------------------------
  smooth        la relajacion del post-filtro, G = smooth + (1-smooth) m_pf. Es
                EL tradeoff PESQ/STOI y depende de para que se use la salida
                (escucha humana vs. reconocimiento), asi que no hay un optimo
                que se pueda clavar. 1.0 = sin post-filtro.
  block_update  P: cada cuantos frames se RECALCULAN los pesos. La estadistica
                (`core.update`) se acumula SIEMPRE, frame a frame; lo que se
                espacia es el `solve`, que es donde estan el eigh de la
                proyeccion PSD y el sistema M x M por bin -- o sea toda la
                cuenta cara del sistema. Con P > 1 el costo por frame baja
                ~P veces y los frames intermedios filtran con los ultimos
                pesos. Es LA palanca de presupuesto del port a ARM, y por eso
                queda expuesta aunque el default medido sea 1.
  alpha, sharpen_exp, min_loading, y la agenda (centro/ancho/alpha/calib):
                barridos y fijados, pero siguen siendo parametros porque
                dependen de la tasa de frames y del modelo de la red.

LO QUE NO ESTA, Y POR QUE
-------------------------
  fuga del canal de referencia (`leak`), perro guardian (`guard`), piso de la
  rama de senal (`mask_floor`): eran las tres defensas contra el SELF-NULLING
  (voz cancelada -> mascara sin voz -> la voz entra a Phi_NN -> mas
  cancelacion). El estado absorbente NO es alcanzable en este esquema: la rama
  de senal del SCM se alimenta de la mascara FUNDIDA, y `m_ref` no depende del
  lazo, asi que Phi_XX nunca deja de ver la voz. Las defensas costaban iSIR de
  entrada a la red sin comprar nada.
  SUSTRACCION DE COVARIANZA Y PROYECCION PSD (`SoudenSubtractCore`): el camino
  anterior estimaba el target restando, Phi_SS = Phi_XX - Phi_NN, y tenia que
  proyectar el resultado al cono PSD con un eigh por bin para que la
  normalizacion de Souden no cambiara de signo. Ese eigh era el ~77 % del
  sistema. Aca la mezcla enmascarada entra directo en la formula: PSD por
  construccion, sin descomposiciones, sin piso de lambda. Empata en PESQ/STOI,
  cuesta 4.5 dB de SIR, gana en la captura real del aro, y el sistema corre
  3.5x mas rapido. Todo el detalle y los numeros, en `SoudenCore`.
  BAN (normalizacion analitica ciega), segunda etapa del DTLN sobre la salida
  (`stage2`), fusion en el dominio de las covarianzas, mascara por canal con
  mediana sobre los M, `mask_warp`, sustraccion de covarianza (`mu`): medidos y
  descartados. El detalle de cada uno esta en `output_feedback.py`.

  OJO con una que NO es una opcion sino un residuo: en `output_feedback.py` el
  coeficiente de fuga arranca en b=1 y se suaviza hacia su valor nominal
  (`leak_smooth=0.5`), asi que aun con `leak=0` los primeros frames la red veia
  una mezcla con el canal crudo (b = 0.5, 0.25, 0.125 ... ; b < 1e-3 desde el
  frame 10). Aca la red ve `Y` puro desde el primer frame. En el frame 0 no hay
  diferencia (w = e_ref -> Y = x_ref), y el efecto de los ~8 frames siguientes
  esta medido en tests/ofb_refactor_equivalence.py: Delta PESQ -0.0005 de
  media y 0.005 en la peor celda contra el camino viejo, o sea ruido de
  medicion.

ESTADO (lo que hay que llevarse al port; K = 257)
-------------------------------------------------
                                            M=8 (MIRD)      M=12 (el aro)
    Num_XX, Num_NN   2 x K x M x M cplx     2 x 263 kB      2 x 592 kB
    Den_XX, Den_NN   2 x K reales           2 x 2.1 kB      2 x 2.1 kB
    w_hold           K x M cplx             32.9 kB         49.3 kB
    estado LSTM      2 redes x el tensor del tflite
    tracker de iSIR  4 reales
(complex128; en complex64 todo se divide por dos: 1.18 MB -> 592 kB los dos
acumuladores con M=12.) Los dos acumuladores de covarianza son TODO el
presupuesto de memoria; el resto es ruido de redondeo al lado de eso.

ESTRUCTURA
----------
    ISIRTracker            el estimador ciego del iSIR de la escena.
    SoudenCore             las dos SCM y los pesos. `update` (todos los frames)
                           y `solve` (cada P) -- el corte que define el port.
    OutputFeedbackMVDR     TODO el estado del sistema + `step(X_frame) -> Y`,
                           o sea un frame. Es la unidad que se porta a C++:
                           el constructor es la allocacion y `step` es la ISR.
    output_feedback_run    driver offline: el `for` sobre los frames de una
                           STFT ya calculada, y el diagnostico. No tiene
                           algoritmo adentro.
"""

import numpy as np

from .blind_feedback import DTLNStream

# Banda donde la mascara del DTLN es informativa. Abajo de 300 Hz manda el
# rumble y arriba de 3400 Hz casi no queda energia: los dos extremos le meten
# varianza al estimador de iSIR sin aportar informacion.
ISIR_BAND_HZ = (300.0, 3400.0)

# Defaults de la agenda, TODOS medidos juntos (tests/ofb_isir_estimator_check.py
# y tests/ofb_auto_benchmark.py). Cambiar uno sin los otros la descalibra.
ISIR_ALPHA = 0.998                  # suavizado del estimador
ISIR_CALIB = (0.496, 3.77)          # hat = g * iSIR_real + b, con este alpha
ISIR_CENTER = 2.2                   # cruce de la agenda, en dB de iSIR REAL
ISIR_WIDTH = 3.5                    # ancho del sigmoide, idem


class ISIRTracker:
    """
    ESTIMADOR CIEGO DEL iSIR DE LA ESCENA, a partir de la mascara del DTLN sobre
    el canal de referencia. Es lo que hace autonoma la agenda del post-filtro:
    sin el, el cruce del sigmoide habria que pasarselo de afuera, o sea saber el
    iSIR de antemano.

    La idea es la unica disponible sin informacion nueva: la mascara `a` ya
    reparte la energia del canal de referencia en "voz" y "resto", asi que el
    cociente de esas dos energias es un correlato del iSIR. No hace falta que
    sea insesgado -- hace falta que sea MONOTONO, porque el sesgo se absorbe en
    la calibracion afin.

    LA FORMA
    --------
    Cociente de potencias POR BIN OCUPADO (las potencias se dividen por la masa
    de mascara acumulada) y restringido a `ISIR_BAND_HZ`. La normalizacion por
    masa es lo que cancela el CICLO DE TRABAJO del locutor: sin ella, en los
    silencios la rama de senal no acumula y la de ruido si, y el estimador
    termina midiendo cuanto habla el locutor adentro del iSIR. Medido, la forma
    cruda da 5.9 dB con un locutor y 14.6 con otro para el MISMO iSIR de +20 dB.

    MEDIDO (8 escenas MIRD no-voz: 2 salas x 2 angulos x 2 locutores, iSIR de
    -10 a +20 dB): monotonia perfecta por escena (Spearman 1.00, 0.98 global),
    dispersion entre escenas de 1.9 dB REALES contra un sigmoide de 3.5 dB de
    ancho, convergencia < 1 s, y ninguna celda del lado equivocado del cruce.
    Contra el ORACULO corriendo la misma agenda: +0.0004 Delta PESQ de media.
    Saber el iSIR de antemano no compra nada.

    LIMITE FUNDAMENTAL (no es de calibracion): la mascara del DTLN detecta VOZ,
    no el TARGET. Con un interferente HABLADO la energia del interferente cae en
    la rama de senal y el iSIR deja de ser observable por esta via (Spearman ~0,
    negativo por escena). El dano esta acotado -- la agenda elige un punto
    arbitrario de una interpolacion cuyos dos extremos son mascaras razonables
    -- pero la ganancia de la agenda se pierde. Sacar el iSIR en ese caso pide
    informacion ESPACIAL, que este estimador no mira.

    COSTO
    -----
    Nada. La mascara `a` ya esta calculada (la usa la fusion), asi que lo unico
    que agrega por frame son, sobre los 99 bins de la banda: UN producto interno
    (99 MAC), DOS reducciones (la rama de ruido sale por identidad, ver
    `update`), una recursion de 4 taps, una division y un log10. Unos 300 MAC y
    un logaritmo. Medido en x86: 5.3 us por frame contra 2135 us del core de
    Souden, o sea el 0.24% del frame.

    CALIBRACION
    -----------
    `calib=(g, b)` invierte el ajuste medido hat = g*iSIR + b, asi que `update`
    devuelve dB de iSIR REAL y el centro/ancho del sigmoide son magnitudes
    fisicas. OJO: (g, b) dependen de `alpha` -- mas suavizado comprime menos y
    sube la pendiente -- asi que los dos van juntos.

    `oracle_db` reemplaza la estimacion por un valor fijo. Es un gancho de
    VALIDACION (el barrido corre la MISMA agenda con el iSIR verdadero para
    medir cuanto cuesta estimarlo) y no una opcion de produccion; vive aca
    adentro justamente para que el lazo no tenga que saber que existe. El estado
    se sigue actualizando igual, asi que `raw_db` deja ver al estimador corriendo
    en paralelo con el oraculo, en la misma corrida.
    """

    def __init__(self, freqs, alpha=ISIR_ALPHA, calib=ISIR_CALIB, oracle_db=None):
        self.alpha = float(np.clip(alpha, 0.0, 0.9999))
        self.calib = (float(calib[0]), float(calib[1]))
        if abs(self.calib[0]) < 1e-6:
            raise ValueError("la pendiente de la calibracion no puede ser 0.")
        self.oracle_db = None if oracle_db is None else float(oracle_db)

        # La banda es CONTIGUA (sale de un par de frecuencias de corte), asi que
        # se guarda como `slice`: indexar con booleanos copia el arreglo, y aca
        # la copia es la mitad del costo.
        f = np.asarray(freqs, dtype=np.float64)
        idx = np.flatnonzero((f >= ISIR_BAND_HZ[0]) & (f <= ISIR_BAND_HZ[1]))
        if not len(idx):
            raise ValueError(f"no hay bins en {ISIR_BAND_HZ} Hz para esta STFT.")
        self.bins = slice(int(idx[0]), int(idx[-1]) + 1)
        self.state = None            # (P_senal, masa_senal, P_ruido, masa_ruido)
        self.raw_db = 0.0            # el estimador, aunque haya oraculo

    def update(self, a, Px):
        """
        Un frame: `a` (K,) mascara del canal de referencia, `Px` (K,) su
        densidad de potencia |X_ref|^2. Devuelve el iSIR en dB REALES.

        La rama de RUIDO no se calcula: sale por identidad de la de senal,

            sum (1-a) Px = sum Px - sum a Px       sum (1-a) = Kb - sum a

        asi que el frame son UN producto interno y DOS reducciones, sin
        materializar (1-a) ni los dos productos elemento a elemento. Es 4 veces
        mas rapido que la forma directa y, en un puerto a hardware, es tambien
        la forma que menos memoria intermedia pide.
        """
        ab, pb = a[self.bins], Px[self.bins]
        s = float(ab @ pb)                       # sum a Px
        ms = float(ab.sum())                     # masa de la rama de senal
        v = np.array([s, ms, float(pb.sum()) - s, float(len(ab)) - ms])
        self.state = v if self.state is None else (self.alpha * self.state +
                                                   (1.0 - self.alpha) * v)
        s, ms, n, mn = self.state
        hat = 10.0 * np.log10(((s / max(ms, 1e-20)) + 1e-20) /
                              ((n / max(mn, 1e-20)) + 1e-20))
        g, b = self.calib
        self.raw_db = float((hat - b) / g)
        return self.raw_db if self.oracle_db is None else self.oracle_db


def _forward_sub(L, B):
    """Resuelve L X = B con L triangular INFERIOR. L (K,M,M), B (K,M,R)."""
    M, R = B.shape[1], B.shape[2]
    X = np.zeros_like(B)
    for i in range(M):
        acc = B[:, i, :]
        if i:
            acc = acc - np.einsum("kj,kjr->kr", L[:, i, :i], X[:, :i, :])
        X[:, i, :] = acc / L[:, i, i][:, None]
    return X


def _back_sub(U, B):
    """Resuelve U X = B con U triangular SUPERIOR. U (K,M,M), B (K,M,R)."""
    M = B.shape[1]
    X = np.zeros_like(B)
    for i in range(M - 1, -1, -1):
        acc = B[:, i, :]
        if i < M - 1:
            acc = acc - np.einsum("kj,kjr->kr", U[:, i, i + 1:], X[:, i + 1:, :])
        X[:, i, :] = acc / U[:, i, i][:, None]
    return X


class SoudenCore:
    """
    NUCLEO DE SOUDEN SIN SUSTRACCION DE COVARIANZA.

    Dos acumuladores, uno por rama de la mascara, y la formula de Souden:

        Phi_XX = sum a^t m_s x x^H / sum a^t m_s      <- MEZCLA enmascarada por
                                                         la rama de voz
        Phi_NN = sum a^t m_n x x^H / sum a^t m_n
        B      = Phi_NN^-1 Phi_XX
        w      = B e_ref / tr(B)

    POR QUE NO SE RESTA (la decision que define este nucleo)
    -------------------------------------------------------
    Phi_XX no es la covarianza del TARGET sino la de la mezcla enmascarada,
    Phi_XX ~= Phi_SS + Phi_NN. El camino anterior (`SoudenSubtractCore`)
    estimaba el target restando, Phi_SS = Phi_XX - Phi_NN, y ese es el origen de
    TODO el costo del sistema: la resta de dos estimaciones ruidosas sale del
    cono PSD en el 99.6 % de los bins, lambda_S = tr(Phi_NN^-1 Phi_SS) queda
    NEGATIVO en el 27 %, y hay que proyectar con un eigh por bin para que la
    normalizacion signifique algo. Ese eigh era el 81 % del `solve` y el ~77 %
    del sistema entero.

    Sin restar, Phi_XX = sum (pesos >= 0) x x^H es PSD POR CONSTRUCCION:

      * no hay autovalores negativos -> NO HAY NADA QUE PROYECTAR: se cae el
        eigh, y con el la unica descomposicion espectral del sistema. Queda un
        Cholesky de M x M por bin y sustituciones -- determinista, sin ramas de
        decision, sin tolerancias de convergencia, sin autovalores repetidos.
        Para un port a mano eso no es solo mas rapido: es una clase entera de
        bugs que desaparece.
      * tr(B) = tr(Phi_NN^-1 Phi_SS) + M = lambda_S + M >= M, asi que el
        denominador NO PUEDE acercarse a cero y el piso `lambda_floor` deja de
        hacer falta.
      * cuando la estimacion es mala (lambda_S << M) el filtro degrada suave a
        e_ref / M, o sea al mic de referencia atenuado, en vez de explotar.

    EL PRECIO, MEDIDO (tests/ofb_psd_variants.py, tests/ofb_lowfreq_bands.py)
    ------------------------------------------------------------------------
    Contra el nucleo con sustraccion y proyeccion:
      arreglo 3-3-3-8-3-3-3, 28 celdas (iSIR -10..+20, interf 45 y 90 grados):
        EMPATA en PESQ (media -0.005, gana 19/28) y en STOI; -4.5 dB de SIR.
      arreglo 8-8-8-8-8-8-8, mas ancho, 21 celdas (3 salas):
        pierde 0.058 de PESQ de media (gana 7/21). El sesgo hacia e_ref tira
        ganancia de arreglo, y un arreglo mas ancho tiene mas para perder.
      captura REAL del aro de 12 mics: GANA en las cinco no intrusivas
        (OVRL 3.13 vs 3.06, SIG 3.56 vs 3.51, BAK 3.80 vs 3.76, segSNR +0.23 dB)
        y el sistema completo corre 3.5x mas rapido (3.6 s contra 12.7 s para
        20 s de audio de 12 canales).
    El colapso de escala en graves que este esquema arrastraria en teoria
    (w -> u/M, -20log10(M) dB) NO aparece: medido por bandas, en 0-130 Hz queda
    +0.42 dB POR ENCIMA del camino con sustraccion. La razon es que el SCM come
    la mascara FUNDIDA, y la rama de atras (m_ref) no depende del lazo, asi que
    la rama de senal nunca se queda sin masa. Ver `souden_mvdr.py` para el
    analisis del colapso en los cores de un solo camino de mascara.

    ESTADO: Num_XX, Num_NN (K, M, M) complejos + Den_XX, Den_NN (K,) reales.
    Son TODO el presupuesto de memoria del sistema.
    """

    def __init__(self, K, M, ref_mic, alpha=0.99, min_loading=1e-9,
                 solve_mode="direct"):
        self.K, self.M, self.ref = int(K), int(M), int(ref_mic)
        self.alpha = float(alpha)
        self.min_loading = float(min_loading)
        # 'direct' = M lados derechos + traza (la forma historica, y la rapida en
        # numpy). 'chol' = la forma del PORT: dos Cholesky, una sustitucion
        # triangular-triangular y UN lado derecho. Ver `_solve_chol`.
        self.solve_mode = solve_mode
        self.eye = np.eye(self.M)[None, :, :]
        self.Num_XX = np.zeros((self.K, self.M, self.M), dtype=np.complex128)
        self.Num_NN = np.zeros((self.K, self.M, self.M), dtype=np.complex128)
        self.Den_XX = np.zeros((self.K, 1, 1), dtype=np.float64)
        self.Den_NN = np.zeros((self.K, 1, 1), dtype=np.float64)

    def update(self, X_frame, m_s, m_n):
        """
        Acumula las dos SCM del frame. Sin inversiones ni descomposiciones, pero
        TOCA TODO EL ESTADO: corre en TODOS los frames, no cada P, y es la parte
        que no se puede amortizar con `block_update`.
        """
        R = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        ms, mn = m_s[:, None, None], m_n[:, None, None]
        self.Num_XX = self.alpha * self.Num_XX + ms * R
        self.Den_XX = self.alpha * self.Den_XX + ms
        self.Num_NN = self.alpha * self.Num_NN + mn * R
        self.Den_NN = self.alpha * self.Den_NN + mn

    def solve(self):
        """
        Los pesos a partir del estado. La unica cuenta pesada del sistema, y es
        un sistema lineal M x M por bin: Phi_NN cargada es hermitiana definida
        positiva, asi que en el port esto es un Cholesky y dos sustituciones.
        """
        Phi_XX = self.Num_XX / (self.Den_XX + 1e-15)
        Phi_NN = self.Num_NN / (self.Den_NN + 1e-15)
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        # Carga diagonal RELATIVA a la traza: invariante a la escala de entrada.
        # OJO EN EL PORT A float32: el default 1e-9 esta POR DEBAJO del epsilon
        # de float (1.2e-7), o sea que en precision simple la carga DESAPARECE en
        # el redondeo y Phi_NN queda sin regularizar. Medido (8 celdas MIRD,
        # Delta PESQ contra 1e-9): 1e-6 cuesta -0.002, 1e-5 cuesta -0.004 (y
        # mejora SIR y SDR), 1e-4 ya cuesta -0.018 y 1e-3 -0.097. En float usar
        # min_loading=1e-5, que queda 84x por encima del epsilon.
        tr = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        Phi_NN = Phi_NN + self.eye * ((self.min_loading * (tr / self.M))[:, None, None]
                                      + 1e-12)

        if self.solve_mode == "chol":
            return self._solve_chol(Phi_XX, Phi_NN)
        B = np.linalg.solve(Phi_NN, Phi_XX)
        # lambda = lambda_S + M >= M por construccion: sin piso, sin sorpresas.
        lam = np.real(np.trace(B, axis1=1, axis2=2))
        return B[:, :, self.ref] / (lam[:, None] + 1e-15)

    def _solve_chol(self, Phi_XX, Phi_NN):
        """
        LA FORMA DEL PORT. Identica a la directa, con 2.3x menos trabajo.

        La forma directa resuelve las M columnas de X = Phi_NN^-1 Phi_XX solo
        para sumarle la diagonal y sacar lambda -- y despues usa UNA sola de esas
        columnas para los pesos. Las otras M-1 se calculan y se tiran.

        Con Phi_NN = LB LB^H y Phi_XX = LA LA^H (las dos Cholesky existen: la de
        Phi_NN porque esta cargada, la de Phi_XX PORQUE NO SE RESTA, ver el
        docstring de la clase):

            lambda = tr(Phi_NN^-1 Phi_XX) = tr(LA^H Phi_NN^-1 LA)
                   = || LB^-1 LA ||_F^2

        o sea una norma de Frobenius sobre una sustitucion triangular con lado
        derecho TRIANGULAR, y los pesos salen de resolver UN solo lado derecho,
        la columna `ref` de Phi_XX:

            actual     chol(B) M^3/6 + M lados derechos M^3   = 1.17 M^3
            esta       chol(B) + chol(A) + tri-tri + 1 RHS    = 0.50 M^3

        La equivalencia es exacta (verificado a 4e-16, tests/ofb_solve_variants.py);
        lo unico que las separa es la carga diagonal que hay que ponerle a
        Phi_XX para que su Cholesky exista siempre -- al arranque Phi_XX tiene
        rango 1 y no es definida positiva hasta acumular M frames.
        """
        M, ref = self.M, self.ref
        tr_A = np.real(np.trace(Phi_XX, axis1=1, axis2=2))
        A = Phi_XX + self.eye * ((self.min_loading * (tr_A / M))[:, None, None] + 1e-30)
        LB = np.linalg.cholesky(Phi_NN)
        LA = np.linalg.cholesky(A)
        Z = _forward_sub(LB, LA)                       # LB Z = LA
        lam = np.real(np.einsum("kij,kij->k", Z.conj(), Z))     # ||Z||_F^2
        y = _forward_sub(LB, Phi_XX[:, :, ref:ref + 1])
        w = _back_sub(np.conj(np.transpose(LB, (0, 2, 1))), y)  # LB^H w = y
        return w[:, :, 0] / (lam[:, None] + 1e-15)

    @staticmethod
    def apply(weights, X_frame):
        """La unica cuenta que queda en el camino critico: y = w^H x."""
        return np.einsum("fm,fm->f", weights.conj(), X_frame)


class OutputFeedbackMVDR:
    """
    EL SISTEMA, por frame. Todo el estado adentro y un solo metodo caliente:

        Y = proc.step(X_frame)           X_frame (K, M) -> Y (K,)

    Esta partido asi porque es la forma que se porta: el constructor es la
    allocacion estatica del port (dos acumuladores de covarianza, los pesos, dos
    estados LSTM y cuatro reales del tracker) y `step` es lo que corre adentro
    de la interrupcion de audio. El `for` sobre los frames vive afuera, en
    `output_feedback_run`, y no tiene algoritmo adentro.

    El frame, en orden (el mismo del port):

        1. Y = w^H x                 <- CAMINO CRITICO: los pesos son de antes
        2. m_out = DTLN(|Y|)         <- las dos redes, mismo modelo, dos estados
           m_ref = DTLN(|x_ref|)
        3. iSIR -> c -> m_pf         <- la agenda
        4. Y *= smooth + (1-smooth) m_pf
        5. core.update(x, m^p, (1-m)^p)          <- estadistica, TODOS los frames
        6. cada P frames: w = core.solve()       <- la cuenta cara

    Los pasos 5 y 6 estan FUERA del camino critico: producen los pesos del frame
    que viene, no del que se acaba de emitir.
    """

    def __init__(self, K, M, ref_mic, freqs, model_path, nperseg,
                 alpha=0.99, sharpen_exp=8.0, min_loading=1e-9,
                 block_update=1, smooth=0.2,
                 isir_center=ISIR_CENTER, isir_width=ISIR_WIDTH,
                 isir_alpha=ISIR_ALPHA, isir_calib=ISIR_CALIB, isir_db=None,
                 solve_mode="direct"):
        """
        K, M: bins y canales. ref_mic: el canal respecto del cual el nucleo es
        distortionless y del que sale `m_ref`. freqs (K,): para la banda del
        estimador. nperseg: largo del bloque (el DTLN come la magnitud SIN
        normalizar de la FFT, o sea |nperseg * X|). El resto, en el docstring
        del modulo.
        """
        if not (0 <= int(ref_mic) < M):
            raise ValueError(f"ref_mic={ref_mic} fuera de rango para M={M}.")
        if isir_width <= 0.0:
            raise ValueError(f"el ancho del sigmoide tiene que ser > 0: {isir_width}")
        self.K, self.M = int(K), int(M)
        self.ref = int(ref_mic)
        self.nperseg = int(nperseg)
        self.p = float(sharpen_exp)
        self.P = max(1, int(block_update))
        # smooth=None (historico) = sin post-filtro, igual que smooth=1.0.
        self.smooth = 1.0 if smooth is None else float(smooth)
        self.isir_center = float(isir_center)
        self.isir_width = float(isir_width)

        # El nucleo, SIN sustraccion de covarianza: ver `SoudenCore` para el
        # porque y para lo que cuesta. Con el se cayeron, ademas del eigh, el
        # piso de lambda, la sustraccion parametrica (mu) y la BAN.
        self.core = SoudenCore(self.K, self.M, self.ref, alpha=alpha,
                               min_loading=min_loading, solve_mode=solve_mode)
        # Las dos redes comparten el MODELO; lo que no comparten es el estado.
        self.dtln_out = DTLNStream(model_path)     # sobre la salida del BF
        self.dtln_ref = DTLNStream(model_path)     # sobre el canal crudo
        self.isir = ISIRTracker(freqs, alpha=isir_alpha, calib=isir_calib,
                                oracle_db=isir_db)

        # Arranque: w = e_ref -> Y(0) = x_ref. El nucleo sobre estado nulo daria
        # w = 0, que dejaria a la red sin nada que mirar en el primer frame.
        self.weights = np.zeros((self.K, self.M), dtype=np.complex128)
        self.weights[:, self.ref] = 1.0
        self.w_used = self.weights          # los del ultimo `step` (diagnostico)
        self.t = 0

        # Ultimo frame, para diagnostico (el lazo no los lee).
        self.m_out = self.m_ref = self.m_pf = np.zeros(self.K)
        self.isir_db = 0.0
        self.c_pf = 0.0

    def step(self, X_frame):
        """X_frame (K, M) complejo -> Y (K,) complejo, ya post-filtrado."""
        w_used = self.weights

        # --- CAMINO CRITICO: la unica combinacion lineal del sistema ---------
        Y = np.einsum("fm,fm->f", w_used.conj(), X_frame)

        # --- las dos mascaras ------------------------------------------------
        x_ref = X_frame[:, self.ref]
        m_out = self._mask(self.dtln_out, Y)
        m_ref = self._mask(self.dtln_ref, x_ref)

        # --- la agenda: que mascara describe mejor a `Y` en este iSIR ---------
        isir_db = self.isir.update(m_ref, np.abs(x_ref) ** 2)
        c = 1.0 / (1.0 + np.exp(-(isir_db - self.isir_center) / self.isir_width))
        m_pf = (1.0 - c) * m_out + c * m_ref

        # --- post-filtro sobre la salida -------------------------------------
        Y = Y * (self.smooth + (1.0 - self.smooth) * m_pf)

        # --- FUERA DEL CAMINO CRITICO: los pesos del frame que viene ---------
        # El SCM come el PROMEDIO de las dos mascaras: necesita masa en las DOS
        # ramas, cosa que el post-filtro -- que solo escala la salida -- no.
        m_scm = 0.5 * (m_out + m_ref)
        self.core.update(X_frame, m_scm ** self.p, (1.0 - m_scm) ** self.p)
        if self.t % self.P == 0:
            self.weights = self.core.solve()
        self.t += 1

        self.m_out, self.m_ref, self.m_pf = m_out, m_ref, m_pf
        self.isir_db, self.c_pf = isir_db, c
        self.w_used = w_used
        return Y

    def _mask(self, net, spectrum):
        """La red come la magnitud SIN normalizar del bloque; sale (K,) en [0,1]."""
        m = net.step(np.abs(self.nperseg * spectrum))
        return np.clip(np.asarray(m, dtype=np.float64), 0.0, 1.0)


def output_feedback_run(X_stft, model_path, nperseg, ref_mic_idx, freqs,
                        alpha=0.99, sharpen_exp=8.0, min_loading=1e-9,
                        block_update=1, smooth=0.2, isir_center=ISIR_CENTER,
                        isir_width=ISIR_WIDTH, isir_alpha=ISIR_ALPHA,
                        isir_calib=ISIR_CALIB, isir_db=None,
                        solve_mode="direct", return_diag=False,
                        return_weights=True):
    """
    Driver offline: corre `OutputFeedbackMVDR` sobre una STFT ya calculada, con
    analisis RECTANGULAR.

    La ventana rectangular no es una opcion: es lo que hace que el frame de la
    STFT y el bloque que ve el DTLN sean LAS MISMAS MUESTRAS, que es la
    condicion para poder meter la red adentro del lazo (ver `blind_feedback`).
    La sintesis con taper la hace el que llama.

    Args:
        X_stft: (K, T, M) complejo, ventana de analisis rectangular.
        model_path: .tflite de la etapa 1 del DTLN (las dos redes, mismo modelo).
        nperseg, ref_mic_idx, freqs, alpha, sharpen_exp, min_loading,
        block_update, smooth, isir_*: ver `OutputFeedbackMVDR`.
        return_weights: los pesos (K, T, M) son DIAGNOSTICO, no algoritmo, y a
            K=257/M=8 son ~60 MB por celda de 15 s. False no los acumula.

    Returns:
        (Y_stft (K,T), W_out (K,T,M) | None) y, con return_diag, un tercer dict
        con las trayectorias del estimador, de la agenda y de las tres mascaras.
    """
    X_stft = np.asarray(X_stft)
    K, T, M = X_stft.shape
    proc = OutputFeedbackMVDR(K, M, ref_mic_idx, freqs, model_path, nperseg,
                              alpha=alpha, sharpen_exp=sharpen_exp,
                              min_loading=min_loading, block_update=block_update,
                              smooth=smooth, isir_center=isir_center,
                              isir_width=isir_width, isir_alpha=isir_alpha,
                              isir_calib=isir_calib, isir_db=isir_db,
                              solve_mode=solve_mode)

    Y_stft = np.zeros((K, T), dtype=np.complex128)
    W_out = np.zeros((K, T, M), dtype=np.complex128) if return_weights else None
    diag = None
    if return_diag:
        diag = {"isir_db": np.zeros(T), "isir_raw_db": np.zeros(T),
                "c_pf": np.zeros(T), "m_out": np.zeros((K, T)),
                "m_ref": np.zeros((K, T)), "m_pf": np.zeros((K, T)),
                "freqs": np.asarray(freqs)}

    for t in range(T):
        Y_stft[:, t] = proc.step(X_stft[:, t, :])
        if W_out is not None:
            W_out[:, t, :] = proc.w_used        # los pesos con los que se filtro
        if diag is not None:
            diag["isir_db"][t] = proc.isir_db
            diag["isir_raw_db"][t] = proc.isir.raw_db
            diag["c_pf"][t] = proc.c_pf
            diag["m_out"][:, t] = proc.m_out
            diag["m_ref"][:, t] = proc.m_ref
            diag["m_pf"][:, t] = proc.m_pf

    return (Y_stft, W_out, diag) if return_diag else (Y_stft, W_out)
