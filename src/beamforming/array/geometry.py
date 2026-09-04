import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_log_array_coords(M: int, d_min: float, d_max: float, room_dims: np.ndarray) -> np.ndarray:
    """
    Generates a 1D logarithmic (geometric) microphone array centered in the room.
    The array is aligned along the X-axis.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if d_min >= d_max:
        raise ValueError("d_min must be strictly less than d_max.")
        
    if M == 2:
        pos_x = np.array([0.0, d_max])
    else:
        pos_x = np.zeros(M)
        pos_x[1:] = np.geomspace(d_min, d_max, num=M-1)
        
    pos_x = pos_x - (d_max / 2.0)
    
    mic_coords = np.zeros((M, 3))
    mic_coords[:, 0] = pos_x
    
    room_center = np.array(room_dims) / 2.0
    mic_coords = mic_coords + room_center
    
    margin = 0.1 
    if np.any(mic_coords < margin) or np.any(mic_coords > (np.array(room_dims) - margin)):
        raise ValueError(f"The array of length {d_max}m does not fit safely.")
        
    return mic_coords


# =============================================================================
# TOPOLOGIAS 2D INSCRIPTAS EN UN CIRCULO
# =============================================================================
# Generadores de geometrias para el estudio comparativo de topologias del
# arreglo (barrido de topologia vs. calidad objetiva del beamformer MVDR).
#
# CONVENCION (identica a generate_mird_linear_array_from_spacing en mird_loader):
#   - Devuelven coordenadas (M, 3) CENTRADAS EN EL ORIGEN, en el plano XY (z=0).
#   - El caller les suma `array_center` para posicionarlas en la sala, p.ej.:
#         mic_coords = generate_circular_array_coords(M, diameter) + array_center
#   - Todas menos la grilla quedan INSCRIPTAS en un circulo del `diameter` dado:
#     ningun microfono cae fuera del circulo, y al menos uno lo toca (se usa el
#     diametro completo), de modo que la apertura fisica sea comparable.
#   - EXCEPCION: la grilla rectangular ('grid' -> generate_grid_array_coords_area)
#     ya NO se inscribe en el circulo. Se la iguala por SUPERFICIE OCUPADA
#     (A_ref = pi * (diameter/2)**2), lo que permite filas != columnas y amplia el
#     rango de M utilizables (6 -> 2x3, 8 -> 2x4, 12 -> 3x4). La version historica
#     inscripta sigue disponible como 'grid_inscribed'.
# =============================================================================

def generate_circular_array_coords(M: int, diameter: float, angle_offset_deg: float = 0.0) -> np.ndarray:
    """
    Uniform Circular Array (UCA): M microphones equally spaced on the
    circumference of a circle of the given `diameter`. Lies on the XY plane
    (z=0), centred on the origin.

    Args:
        M: number of microphones (>= 2).
        diameter: diameter of the inscribing circle in metres. The mics sit
            exactly on this circle, so the array aperture equals `diameter`.
        angle_offset_deg: rigid rotation of the ring about its centre, in
            degrees. Only rotates the pattern; does not change the geometry.

    Returns:
        (M, 3) array of coordinates centred on the origin.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")

    radius = diameter / 2.0
    offset = np.deg2rad(angle_offset_deg)
    angles = offset + np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)

    coords = np.zeros((M, 3))
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    return coords


def generate_circular_center_array_coords(M: int, diameter: float,
                                          angle_offset_deg: float = 0.0) -> np.ndarray:
    """
    UCA CON MICROFONO CENTRAL: un microfono en el centro del arreglo y los otros
    M-1 repartidos uniformemente sobre la circunferencia de `diameter`. Plano XY
    (z=0), centrado en el origen. **El microfono central se cuenta dentro de M**,
    asi que a igual M y a igual diametro esta geometria y `generate_circular_array_coords`
    tienen la misma apertura y la misma superficie: la unica diferencia es que una
    "gasta" un sensor en el centro y la otra lo pone en el anillo.

    Motivacion: los beamformers de la familia Souden (NM-MVDR) proyectan la salida
    sobre UN microfono de referencia. En una UCA pura ese canal es forzosamente un
    mic del anillo, a `diameter` de distancia del mic opuesto; con un mic central,
    la referencia queda a `diameter/2` de TODOS los demas — la maxima diferencia de
    camino acustico hacia la referencia se reduce a la mitad, que es justamente lo
    que dispersa las RTF estimadas. El costo es un sensor menos en el anillo (peor
    muestreo angular). Este generador existe para medir ese canje.

    Orden: el microfono CENTRAL es el indice 0, despues el anillo. Asi
    `select_reference_mic` lo elige naturalmente (es el mas cercano al centroide).

    Args:
        M: numero TOTAL de microfonos (>= 3: 1 central + al menos 2 en el anillo).
        diameter: diametro del anillo en metros (= apertura del arreglo).
        angle_offset_deg: rotacion rigida del anillo; no cambia la geometria.

    Returns:
        (M, 3) coordenadas centradas en el origen, el central primero.
    """
    if M < 3:
        raise ValueError("At least 3 microphones are required (1 centre + 2 on the ring).")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")

    ring = generate_circular_array_coords(M - 1, diameter, angle_offset_deg=angle_offset_deg)
    coords = np.zeros((M, 3))
    coords[1:, :] = ring          # el indice 0 queda en el origen (mic central)
    return coords


def generate_grid_array_coords(M: int, diameter: float) -> np.ndarray:
    """
    Square grid of M microphones inscribed in a circle of the given `diameter`,
    with a microphone at the centre. Lies on the XY plane (z=0), centred on the
    origin.

    The grid is laid out on an `n x n` integer lattice with `n` ODD, so that a
    lattice node falls exactly on the origin (guaranteeing a central microphone
    and an equal number of rows and columns). `n` is the smallest odd integer
    with `n**2 >= M` (n = ceil(sqrt(M)), bumped to the next odd if even). Of the
    `n**2` lattice nodes, the M closest to the centre are kept (nearest-first, so
    the central node is always included); when `M == n**2` the full n x n grid is
    used. The result is uniformly scaled so the outermost microphone lands
    exactly on the inscribing circle (radius = diameter / 2), keeping the
    aperture comparable to the other topologies.

    Note: for M an odd perfect square (9, 25, 49, ...) this is a full, perfectly
    symmetric n x n grid. For other M the outermost shell is partially filled,
    but the central microphone and the square n x n span are preserved.

    Args:
        M: number of microphones (>= 2).
        diameter: diameter of the inscribing circle in metres.

    Returns:
        (M, 3) array of coordinates centred on the origin, central mic first.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")

    # Smallest ODD n with n**2 >= M -> a lattice node sits on the origin.
    n = int(np.ceil(np.sqrt(M)))
    if n % 2 == 0:
        n += 1

    # Centred n x n integer lattice: coordinates are symmetric about 0 and
    # include 0 (n is odd), so (0, 0) is a valid node -> central microphone.
    axis = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(axis, axis)
    lattice = np.column_stack([xx.ravel(), yy.ravel()])  # (n*n, 2)

    # Keep the M nodes closest to the centre (stable sort -> centre node first).
    dist = np.linalg.norm(lattice, axis=1)
    keep = np.argsort(dist, kind="stable")[:M]
    grid = lattice[keep]

    # Scale so the outermost kept mic sits on the inscribing circle.
    max_r = np.max(np.linalg.norm(grid, axis=1))
    if max_r > 0:
        grid = grid * ((diameter / 2.0) / max_r)

    coords = np.zeros((M, 3))
    coords[:, :2] = grid
    return coords


def _near_square_factors(M: int, max_aspect: float = 2.0):
    """
    (rows, cols) para M microfonos, lo mas cuadrado posible.

    1) Si M admite una factorizacion exacta rows*cols == M con aspecto
       cols/rows <= max_aspect, devuelve la mas cuadrada (rows maximo).
       Ej.: 6 -> (2, 3); 8 -> (2, 4); 12 -> (3, 4); 16 -> (4, 4).
    2) Si no (M primo, o toda factorizacion demasiado alargada: 7, 11, 10 con
       max_aspect=2, ...), devuelve el rectangulo mas chico y mas cuadrado con
       rows*cols >= M; el generador rellena solo M nodos (shell externa
       parcialmente llena). Ej.: 7 -> (2, 4); 11 -> (3, 4).
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    best = None
    for rows in range(1, int(np.sqrt(M)) + 1):
        if M % rows == 0:
            cols = M // rows
            if rows >= 2 and (cols / rows) <= max_aspect:
                best = (rows, cols)          # rows crece -> se guarda la mas cuadrada
    if best is not None:
        return best
    # Sin factorizacion aceptable: rectangulo minimo (rows*cols >= M) mas cuadrado.
    rows = int(np.floor(np.sqrt(M)))
    while rows >= 1:
        cols = int(np.ceil(M / rows))
        if rows * cols >= M and (cols / rows) <= max_aspect:
            return (rows, cols)
        rows -= 1
    return (1, M)


def generate_grid_array_coords_area(M: int, diameter: float, area_mode: str = "span",
                                    rows: int = None, cols: int = None,
                                    max_aspect: float = 2.0) -> np.ndarray:
    """
    Grilla rectangular de M microfonos EQUIVALENTE POR AREA al arreglo circular
    (UCA) del mismo `diameter`. Plano XY (z=0), centrada en el origen.

    A diferencia de `generate_grid_array_coords` (legacy), la grilla NO esta
    inscripta en el circulo: se libera esa restriccion y se impone en su lugar
    que la SUPERFICIE OCUPADA sea la del circulo de referencia,

        A_ref = pi * (diameter / 2)**2

    lo que permite filas != columnas y por lo tanto amplia el rango de M
    utilizables con una grilla razonablemente cuadrada (6 -> 2x3, 8 -> 2x4,
    12 -> 3x4), en vez de exigir n x n impar.

    Convenciones de area (`area_mode`):
      - 'span' (default): el RECTANGULO QUE ABARCAN LOS MICROFONOS EXTREMOS
        tiene area A_ref, es decir (cols-1)*d x (rows-1)*d == A_ref. Es la
        lectura literal de "misma superficie que el circulo": el mismo plato
        fisico. Con pocos micros la apertura crece (y con ella el espaciado d,
        que baja la frecuencia de aliasing espacial c/(2d)).
      - 'cell': cada microfono ocupa una celda de A_ref / M, es decir
        M * d**2 == A_ref. Conserva la DENSIDAD de micros del circulo (para M=12
        y D=15 cm da d ~ 3.8 cm, casi el paso del UCA de 12 micros) a costa de
        un footprint total menor.

    El paso `d` es el MISMO en las dos direcciones (celdas cuadradas): la
    relacion de aspecto la fija la factorizacion rows x cols, no un estiramiento.

    Si rows*cols > M (M primo o de factorizacion muy alargada, ver
    `_near_square_factors`) se conservan los M nodos mas cercanos al centro y el
    area se ajusta sobre el rectangulo REALMENTE ocupado por esos M nodos.

    Args:
        M: numero de microfonos (>= 2).
        diameter: diametro del circulo de REFERENCIA (define A_ref), en metros.
            Ojo: la grilla puede extenderse fuera de ese circulo; solo comparte
            el area.
        area_mode: 'span' | 'cell' (ver arriba).
        rows, cols: fuerzan la factorizacion (rows*cols debe ser >= M). Si se
            pasa solo uno, el otro se deriva con ceil(M / el dado).
        max_aspect: aspecto maximo cols/rows tolerado al factorizar M.

    Returns:
        (M, 3) coordenadas centradas en el origen, ordenadas por filas
        (y creciente, luego x creciente).
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")
    if area_mode not in ("span", "cell"):
        raise ValueError("area_mode must be 'span' or 'cell'.")

    if rows is None and cols is None:
        rows, cols = _near_square_factors(M, max_aspect=max_aspect)
    elif rows is None:
        cols = int(cols); rows = int(np.ceil(M / cols))
    elif cols is None:
        rows = int(rows); cols = int(np.ceil(M / rows))
    else:
        rows, cols = int(rows), int(cols)
    if rows * cols < M:
        raise ValueError(f"rows*cols={rows*cols} < M={M}: la grilla no entra.")

    # Lattice unitaria rows x cols centrada en el centro del rectangulo.
    ax = np.arange(cols) - (cols - 1) / 2.0
    ay = np.arange(rows) - (rows - 1) / 2.0
    xx, yy = np.meshgrid(ax, ay)
    lattice = np.column_stack([xx.ravel(), yy.ravel()])       # (rows*cols, 2)

    if lattice.shape[0] > M:
        # Shell externa parcial: se quedan los M nodos mas cercanos al centro.
        keep = np.argsort(np.linalg.norm(lattice, axis=1), kind="stable")[:M]
        lattice = lattice[np.sort(keep)]

    # Recentrar en el centro geometrico del rectangulo ocupado (el "centro del
    # plato"): con la grilla completa coincide con el centroide.
    bbox_lo, bbox_hi = lattice.min(axis=0), lattice.max(axis=0)
    lattice = lattice - (bbox_lo + bbox_hi) / 2.0

    area_ref = np.pi * (diameter / 2.0) ** 2
    if area_mode == "cell":
        d = np.sqrt(area_ref / M)
    else:
        span = lattice.max(axis=0) - lattice.min(axis=0)      # en unidades de paso
        if span[0] <= 0 or span[1] <= 0:
            raise ValueError(
                f"area_mode='span' necesita al menos 2 filas y 2 columnas "
                f"(rows={rows}, cols={cols}, M={M}). Usa area_mode='cell' o "
                f"fija rows/cols."
            )
        d = np.sqrt(area_ref / (span[0] * span[1]))

    grid = lattice * d
    # Orden por filas (y, luego x): lectura natural de la grilla.
    order = np.lexsort((grid[:, 0], grid[:, 1]))
    grid = grid[order]

    coords = np.zeros((M, 3))
    coords[:, :2] = grid
    return coords


def select_reference_mic(mic_coords: np.ndarray) -> int:
    """
    Indice del microfono que mejor representa el CENTRO GEOMETRICO del arreglo:
    el que minimiza la suma de distancias cuadraticas al resto de los micros
    (equivalente a: el mas cercano al centroide de la nube).

    Motivacion: los beamformers tipo Souden (NM-MVDR, Souden-oracle) proyectan la
    salida sobre UN microfono de referencia; ese canal fija el "punto de escucha"
    del filtro espacial y la RTF a la que se normalizan los pesos. Elegirlo en el
    centro del arreglo minimiza la maxima diferencia de camino acustico hacia los
    demas micros, lo que reduce la varianza espacial de las RTF estimadas. El
    indice M//2 (default historico) solo cae en el centro por accidente y depende
    del ORDEN en que cada topologia enumera sus micros.

    Empates (p.ej. UCA, o grilla par x par, donde varios micros equidistan del
    centroide) se resuelven por el indice mas bajo (argmin estable).

    Args:
        mic_coords: (M, 3) o (M, 2) coordenadas de los microfonos (absolutas o
            centradas: el criterio es invariante a traslacion).

    Returns:
        int: indice del microfono de referencia en [0, M).
    """
    P = np.asarray(mic_coords, dtype=float)
    if P.ndim != 2 or P.shape[0] < 1:
        raise ValueError("mic_coords must be a (M, D) array with M >= 1.")
    centroid = P.mean(axis=0)
    dist = np.linalg.norm(P - centroid, axis=1)
    # Tolerancia para que los empates EXACTOS por simetria (UCA: todos los micros
    # equidistan del centro) no los desempate el ruido de punto flotante: se toma
    # el indice mas bajo dentro de la tolerancia -> resultado reproducible.
    tol = 1e-9 + 1e-6 * float(np.max(dist))
    return int(np.flatnonzero(dist <= dist.min() + tol)[0])


def sample_torus_specs(n: int, r_major: float, r_tube: float,
                       array_height: float = 0.0, z_min: float = 0.05,
                       rng=None, volume_uniform: bool = True) -> np.ndarray:
    """
    Muestrea `n` posiciones dentro de un TOROIDE APOYADO SOBRE EL PISO, concentrico
    al arreglo, y las devuelve como specs (azimut, elevacion, distancia slant)
    listas para `place_spherical` / el motor del benchmark.

    Modelo de uso: el arreglo esta apoyado sobre una mesa/piso (plano z=0) y los
    locutores estan a ~`r_tube` de altura sobre ese plano, alrededor del arreglo.
    El toroide tiene su circulo generador de radio `r_major` a la altura
    z = r_tube, de modo que el tubo es TANGENTE al piso (z va de 0 a 2*r_tube) y
    el centro de la dispersion en altura cae en r_tube. Reemplaza al domo/medio
    cascaron: la fuente ya no puede estar en el cenit ni pegada al plano lejos,
    sino en el anillo realista alrededor del dispositivo.

    Muestreo: azimut uniforme en [0, 360); dentro de la seccion circular del tubo
    (radio r_tube) se muestrea uniforme por area (sqrt del radio). Con
    `volume_uniform=True` se aplica ademas rechazo proporcional a
    (r_major + drho) / (r_major + r_tube), que corrige la densidad para que el
    muestreo sea uniforme en el VOLUMEN del toroide (la parte externa del tubo
    abarca mas volumen que la interna). Se rechazan los puntos con
    z < `z_min` (una fuente en z<=0 rompe el ISM y ademas no es fisica).

    Args:
        n: cantidad de posiciones.
        r_major: radio mayor [m] (centro del arreglo -> centro del tubo, horizontal).
        r_tube: radio menor [m] (del tubo). Fija la altura media de las fuentes.
        array_height: altura del PLANO DEL ARREGLO [m] sobre el piso. La elevacion
            se mide respecto de ese plano (es el 0 de elevacion del benchmark).
        z_min: altura minima admitida [m] sobre el piso.
        rng: np.random.Generator o semilla (default_rng).
        volume_uniform: True -> uniforme en volumen; False -> uniforme en la
            seccion del tubo (mas peso relativo al lado interno).

    Returns:
        (n, 3) array de (azimut_deg, elevacion_deg, distancia_slant_m) respecto
        del centro del arreglo.
    """
    if r_tube <= 0 or r_major <= 0:
        raise ValueError("r_major y r_tube deben ser estrictamente positivos.")
    if r_tube >= r_major:
        raise ValueError("r_tube debe ser menor que r_major (toroide no degenerado).")
    rng = np.random.default_rng(rng)

    specs = np.zeros((n, 3))
    k = 0
    while k < n:
        phi = rng.uniform(0.0, 2.0 * np.pi)                  # azimut
        # Punto uniforme por area en la seccion circular del tubo.
        rad = r_tube * np.sqrt(rng.random())
        ang = rng.uniform(0.0, 2.0 * np.pi)
        drho, dz = rad * np.cos(ang), rad * np.sin(ang)

        rho = r_major + drho                                  # distancia horizontal
        z = r_tube + dz                                       # altura sobre el piso
        if z < z_min:
            continue
        if volume_uniform and rng.random() > rho / (r_major + r_tube):
            continue                                          # correccion de volumen

        dz_arr = z - array_height                             # altura sobre el arreglo
        dist = float(np.hypot(rho, dz_arr))
        el = float(np.rad2deg(np.arctan2(dz_arr, rho)))
        az = float(np.rad2deg(phi))
        specs[k] = (az, el, dist)
        k += 1
    return specs


def generate_powerlaw_grid_array_coords(M: int, diameter: float, exponent: float = 2.0,
                                        area_mode: str = "span", rows: int = None,
                                        cols: int = None, max_aspect: float = 2.0) -> np.ndarray:
    """
    Grilla rectangular de M microfonos con distribucion por LEY DE POTENCIAS en
    cada eje, inscripta en el MISMO rectangulo que `generate_grid_array_coords_area`
    (misma superficie, mismas filas x columnas, misma apertura). Plano XY (z=0),
    centrada en el origen.

    Sirve para aislar UNA sola variable: si la UNIFORMIDAD del reparto ayuda o no.
    Se parte de la grilla uniforme equivalente por area y se deforma cada eje por
    separado, en coordenadas normalizadas u en [-1, 1]:

        u  ->  sign(u) * |u| ** exponent

    Los extremos (|u| = 1) quedan fijos, asi que el RECTANGULO ENVOLVENTE -y por lo
    tanto la superficie ocupada y la apertura maxima- es identico al de la grilla
    uniforme; lo unico que cambia es como se reparten los micros adentro. Como la
    deformacion es separable (misma para toda una fila / columna), la estructura de
    producto tensorial de la grilla se conserva.

      - exponent = 1   -> identidad: EXACTAMENTE la grilla uniforme (control).
      - exponent > 1   -> micros apretados hacia el CENTRO y ralos en los bordes.
        El co-arreglo gana espaciados chicos (mejor comportamiento en alta
        frecuencia / menos aliasing espacial) sin perder apertura, a costa de
        canales mas correlacionados en el centro.
      - exponent < 1   -> micros empujados hacia los BORDES (tipo dos sub-arreglos
        separados): mas resolucion angular, peor muestreo del campo cercano.

    Nota: un eje con solo 2 nodos (p.ej. las 2 filas de la grilla 2x3 de M=6) no
    cambia -sus micros ya estan en los extremos-, asi que con M chico la ley de
    potencias actua sobre el eje largo unicamente.

    Args:
        M: numero de microfonos (>= 2).
        diameter: diametro del circulo de REFERENCIA que fija la superficie
            (A_ref = pi (diameter/2)^2), igual que en la grilla uniforme.
        exponent: exponente de la ley de potencias (> 0).
        area_mode, rows, cols, max_aspect: identicos a
            `generate_grid_array_coords_area` (definen el rectangulo de base).

    Returns:
        (M, 3) coordenadas centradas en el origen, mismo orden por filas que la
        grilla uniforme (comparables mic a mic).
    """
    if exponent <= 0:
        raise ValueError("exponent must be strictly positive.")

    base = generate_grid_array_coords_area(M, diameter, area_mode=area_mode,
                                           rows=rows, cols=cols, max_aspect=max_aspect)
    xy = base[:, :2]
    warped = xy.copy()
    for ax in (0, 1):
        half = float(np.abs(xy[:, ax]).max())        # semi-span del eje
        if half <= 0:
            continue                                  # eje degenerado (1 sola linea)
        u = xy[:, ax] / half                          # normalizado a [-1, 1]
        warped[:, ax] = half * np.sign(u) * np.abs(u) ** exponent

    coords = np.zeros((M, 3))
    coords[:, :2] = warped
    return coords


def generate_spiral_array_coords(M: int, diameter: float, n_turns: float = 2.0) -> np.ndarray:
    """
    Archimedean spiral of M microphones inscribed in a circle of the given
    `diameter`. Lies on the XY plane (z=0), centred on the origin.

    Microphones are placed at equal angular increments along a spiral whose
    radius grows linearly with angle, from the centre (first mic at r=0) out to
    the inscribing circle (last mic at r = diameter / 2). This yields an
    irregular, non-redundant co-array that spans many inter-sensor spacings.

    Args:
        M: number of microphones (>= 2).
        diameter: diameter of the inscribing circle in metres.
        n_turns: number of full turns of the spiral from centre to edge.

    Returns:
        (M, 3) array of coordinates centred on the origin.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")
    if n_turns <= 0:
        raise ValueError("n_turns must be strictly positive.")

    theta = np.linspace(0.0, 2.0 * np.pi * n_turns, M)
    # Radius linear in angle, normalised so the outermost point hits diameter/2.
    radius = (diameter / 2.0) * (theta / theta[-1])

    coords = np.zeros((M, 3))
    coords[:, 0] = radius * np.cos(theta)
    coords[:, 1] = radius * np.sin(theta)
    return coords


def generate_concentric_array_coords(M: int, diameter: float, inner_ratio: float = 0.5,
                                     stagger: bool = True) -> np.ndarray:
    """
    Two concentric uniform circular rings inscribed in a circle of the given
    `diameter`. Lies on the XY plane (z=0), centred on the origin.

    The M microphones are split between an inner ring (radius
    inner_ratio * diameter / 2) and an outer ring (radius diameter / 2). The
    outer ring receives the extra mic when M is odd, since it has more room.
    The inner ring can be angularly staggered (offset by half its angular step)
    to interleave the two rings and improve spatial sampling.

    Args:
        M: number of microphones (>= 2).
        diameter: diameter of the inscribing (outer) circle in metres.
        inner_ratio: radius of the inner ring as a fraction of the outer radius,
            in (0, 1). Default 0.5.
        stagger: if True, rotate the inner ring by half of its angular step so
            its mics fall between the outer-ring mics.

    Returns:
        (M, 3) array of coordinates centred on the origin. The outer-ring mics
        come first, followed by the inner-ring mics.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")
    if not (0.0 < inner_ratio < 1.0):
        raise ValueError("inner_ratio must lie strictly between 0 and 1.")

    n_outer = M - M // 2  # ceil(M/2): outer ring gets the extra mic if M is odd
    n_inner = M // 2      # floor(M/2)

    outer_r = diameter / 2.0
    inner_r = inner_ratio * outer_r

    outer_ang = np.linspace(0.0, 2.0 * np.pi, n_outer, endpoint=False)
    inner_ang = np.linspace(0.0, 2.0 * np.pi, n_inner, endpoint=False)
    if stagger and n_inner > 0:
        inner_ang = inner_ang + (np.pi / n_inner)  # half angular step

    coords = np.zeros((M, 3))
    coords[:n_outer, 0] = outer_r * np.cos(outer_ang)
    coords[:n_outer, 1] = outer_r * np.sin(outer_ang)
    coords[n_outer:, 0] = inner_r * np.cos(inner_ang)
    coords[n_outer:, 1] = inner_r * np.sin(inner_ang)
    return coords


def generate_random_array_coords(M: int, diameter: float, seed=None,
                                 min_dist_ratio: float = 0.6, max_attempts: int = 2000) -> np.ndarray:
    """
    Random microphone layout inscribed in a circle of the given `diameter`.
    Lies on the XY plane (z=0), centred on the origin.

    Microphones are drawn uniformly (by area) inside the disk with a soft
    minimum-separation constraint (Poisson-disk-like rejection sampling) so no
    two mics coincide or crowd, which would make the spatial covariance
    ill-conditioned. The cloud is then centred on its centroid and uniformly
    scaled so the outermost microphone lands exactly on the inscribing circle
    (radius = diameter / 2), matching the aperture convention of the other
    topologies.

    DETERMINISM: the layout is a pure function of (M, diameter, seed). Pass a
    stable `seed` (e.g. derived from the scene via compute_scene_seed in the
    benchmark) so the SAME random array is reproduced across runs and reused
    identically by every processor of a given experiment. With seed=None the
    layout is non-reproducible (fresh entropy each call) -> avoid in benchmarks.

    Args:
        M: number of microphones (>= 2).
        diameter: diameter of the inscribing circle in metres.
        seed: seed / SeedSequence / Generator for numpy's default_rng. Fixing it
            makes the geometry reproducible.
        min_dist_ratio: target minimum inter-mic spacing as a fraction of the
            nominal spacing R / sqrt(M) (R = diameter/2). Higher -> more evenly
            spread but harder to place; the constraint is relaxed automatically
            if placement stalls, so it never fails to return M mics.
        max_attempts: rejection-sampling attempts before relaxing the constraint.

    Returns:
        (M, 3) array of coordinates centred on the origin.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive.")

    rng = np.random.default_rng(seed)
    R = diameter / 2.0
    min_dist = min_dist_ratio * R / np.sqrt(M)

    pts = []
    attempts = 0
    while len(pts) < M:
        # Uniform-by-area sample in the disk of radius R.
        rad = R * np.sqrt(rng.random())
        ang = 2.0 * np.pi * rng.random()
        cand = np.array([rad * np.cos(ang), rad * np.sin(ang)])

        if all(np.linalg.norm(cand - p) >= min_dist for p in pts):
            pts.append(cand)
            attempts = 0
        else:
            attempts += 1
            if attempts >= max_attempts:
                # Placement stalled: relax the spacing constraint and retry.
                min_dist *= 0.5
                attempts = 0

    grid = np.asarray(pts)
    # Centre on centroid and scale so the outermost mic sits on the circle.
    grid -= grid.mean(axis=0)
    max_r = np.max(np.linalg.norm(grid, axis=1))
    if max_r > 0:
        grid *= R / max_r

    coords = np.zeros((M, 3))
    coords[:, :2] = grid
    return coords


# Dispatcher: mapea un nombre de topologia -> su generador. Comodo para barrer
# topologias en el benchmark (p.ej. for topo in TOPOLOGY_GENERATORS: ...).
TOPOLOGY_GENERATORS = {
    "circular": generate_circular_array_coords,
    # UCA con un mic en el centro (contado dentro de M): misma apertura y area que
    # 'circular', pero la referencia del beamformer queda equidistante de todos.
    "circular_center": generate_circular_center_array_coords,
    # 'grid' = grilla EQUIVALENTE POR AREA al circulo de referencia (filas != columnas
    # permitidas). La version historica inscripta en el circulo (n x n impar) queda
    # disponible como 'grid_inscribed' para reproducir corridas viejas.
    "grid": generate_grid_array_coords_area,
    "grid_inscribed": generate_grid_array_coords,
    # Mismo rectangulo que 'grid' pero reparto por ley de potencias en cada eje
    # (exponent=1 -> identico a 'grid'): aisla el efecto de la UNIFORMIDAD.
    "powerlaw": generate_powerlaw_grid_array_coords,
    "spiral": generate_spiral_array_coords,
    "concentric": generate_concentric_array_coords,
    "random": generate_random_array_coords,
}


def generate_array_coords(topology: str, M: int, diameter: float, **kwargs) -> np.ndarray:
    """
    Dispatcher over the 2D topologies inscribed in a circle. Returns (M, 3)
    coordinates centred on the origin (z=0), ready to be offset by array_center.

    Args:
        topology: one of {"circular", "grid", "spiral", "concentric"}.
        M: number of microphones.
        diameter: diameter of the inscribing circle in metres.
        **kwargs: topology-specific options (e.g. n_turns, inner_ratio).

    Returns:
        (M, 3) array of coordinates centred on the origin.
    """
    key = topology.lower()
    if key not in TOPOLOGY_GENERATORS:
        raise ValueError(f"Unknown topology '{topology}'. "
                         f"Valid options: {sorted(TOPOLOGY_GENERATORS)}")
    return TOPOLOGY_GENERATORS[key](M=M, diameter=diameter, **kwargs)


def sample_box_uniform_specs(n: int, lo, hi, min_dist: float = 0.5,
                             rng=None) -> np.ndarray:
    """
    Muestrea `n` puntos UNIFORMES EN EL VOLUMEN de una caja (la sala, recortada por
    el margen de pared) y los devuelve como specs (azimut, elevacion, distancia
    slant) listas para `place_spherical` / el motor del benchmark.

    `lo` y `hi` son las esquinas de la caja EXPRESADAS RESPECTO DEL CENTRO DEL
    ARREGLO (y con z medida desde el PLANO del arreglo, que es el 0 de elevacion).
    Al trabajar en coordenadas relativas, la misma tanda de specs sirve para varias
    salas: basta tomar la INTERSECCION de sus cajas utiles (ver el notebook del
    barrido de topologias) y ninguna posicion necesita recortarse despues.

    A diferencia de un muestreo (azimut, elevacion, distancia) uniforme por
    separado -que concentra las fuentes en un cascaron alrededor del arreglo y deja
    las esquinas vacias-, esto llena la sala: la densidad es constante por unidad de
    volumen, asi que la mayoria de las posiciones cae lejos (el volumen crece con
    r^2) y a baja elevacion (una sala es mucho mas ancha que alta).

    Se rechazan los puntos a menos de `min_dist` del centro del arreglo (una fuente
    encima de los microfonos no es fisica y ademas degenera el modelo de campo
    cercano). `lo[2]` fija implicitamente la elevacion minima: con lo[2] >= 0 nunca
    hay fuentes por debajo del plano del arreglo (necesario si el arreglo se apoya
    sobre un bafle/piso).

    Args:
        n: cantidad de posiciones.
        lo, hi: (3,) esquinas minima y maxima de la caja, relativas al centro del
            arreglo [m].
        min_dist: distancia minima admitida al centro del arreglo [m].
        rng: np.random.Generator o semilla (default_rng).

    Returns:
        (n, 3) array de (azimut_deg, elevacion_deg, distancia_slant_m).
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("lo y hi deben ser vectores de 3 componentes.")
    if np.any(hi <= lo):
        raise ValueError(f"Caja vacia o invertida: lo={lo}, hi={hi}.")
    rng = np.random.default_rng(rng)

    specs = np.zeros((n, 3))
    k = 0
    while k < n:
        p = rng.uniform(lo, hi)                      # uniforme en el volumen
        hd = float(np.hypot(p[0], p[1]))
        dist = float(np.sqrt(hd ** 2 + p[2] ** 2))
        if dist < min_dist:
            continue
        specs[k] = (float(np.rad2deg(np.arctan2(p[0], p[1])) % 360.0),
                    float(np.rad2deg(np.arctan2(p[2], hd))),
                    dist)
        k += 1
    return specs


def place_spherical(azimuth_deg: float, elevation_deg: float, distance: float,
                    array_center: np.ndarray) -> np.ndarray:
    """
    Absolute (x, y, z) position from a (azimuth, elevation, slant distance) spec
    relative to `array_center`. Convention (matches the topology arrays, which
    face +Y):

      - azimuth 0 deg = front (+Y); positive azimuth rotates toward +X.
      - elevation 0 deg = the array's horizontal plane; positive = ABOVE it.
      - distance = slant range (array_center -> point).

    The out-of-plane height offset is distance * sin(elevation), so the elevation
    angle directly controls how far the point sits above/below the array plane —
    the axis a planar array discriminates worst, i.e. the intended stressor.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    hd = distance * np.cos(el)  # horizontal (ground-projected) distance
    array_center = np.asarray(array_center, dtype=float)
    return np.array([
        array_center[0] + hd * np.sin(az),
        array_center[1] + hd * np.cos(az),
        array_center[2] + distance * np.sin(el),
    ])


def max_distance_in_room(azimuth_deg: float, elevation_deg: float, array_center: np.ndarray,
                         room_dims: np.ndarray, margin: float = 0.3) -> float:
    """
    Largest slant distance along the (azimuth, elevation) ray from `array_center`
    that keeps the resulting point inside the room shrunk by `margin` on every
    wall. Used to clamp source/interference distances so a spec can never place a
    point outside (or hugging) a wall. Returns +inf if the ray is degenerate on
    all three axes.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    d = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    p0 = np.asarray(array_center, dtype=float)
    room_dims = np.asarray(room_dims, dtype=float)
    lo = np.full(3, margin)
    hi = room_dims - margin

    t_max = np.inf
    for k in range(3):
        if d[k] > 1e-9:
            t_max = min(t_max, (hi[k] - p0[k]) / d[k])
        elif d[k] < -1e-9:
            t_max = min(t_max, (lo[k] - p0[k]) / d[k])
    return max(0.0, t_max)


def generate_source_and_interferences(N_interferences: int, radius_source: float, radius_interf: float, delta_ang_deg: float, array_center: np.ndarray) -> tuple:
    """
    Generates the 3D coordinates for the target source and N interferences.
    The target source is fixed at broadside (90 degrees, perpendicular to X-axis array).
    Interferences are placed alternately at +/- multiples of delta_ang relative to the source.
    """
    delta_ang_rad = np.deg2rad(delta_ang_deg)
    
    # The array is on the X-axis. Broadside (perpendicular) is the Y-axis (90 degrees or pi/2)
    ref_angle_rad = np.pi / 2.0
    
    # Calculate target source position
    source_pos = np.copy(array_center)
    source_pos[0] += radius_source * np.cos(ref_angle_rad) # Evaluates to ~0 offset in X
    source_pos[1] += radius_source * np.sin(ref_angle_rad) # Evaluates to radius_source in Y
    
    interferences_pos = np.zeros((N_interferences, 3))
    
    for i in range(N_interferences):
        multiplier = (i // 2) + 1
        sign = 1 if i % 2 == 0 else -1
        
        # Calculate angle relative to the broadside reference
        angle_rad = ref_angle_rad + (sign * multiplier * delta_ang_rad)
        
        # Calculate cartesian coordinates
        x = array_center[0] + radius_interf * np.cos(angle_rad)
        y = array_center[1] + radius_interf * np.sin(angle_rad)
        z = array_center[2]
        
        interferences_pos[i] = [x, y, z]
        
    return source_pos, interferences_pos


def _plot_topology_gallery(M: int = 12, diameter: float = 0.30):
    """Quick visual sanity-check of the four 2D topologies inscribed in a
    circle of the given diameter. Each panel draws the inscribing circle and
    the resulting microphone positions."""
    specs = [
        ("Circular (UCA)", generate_circular_array_coords(M, diameter)),
        ("Grid (=area, span)", generate_grid_array_coords_area(M, diameter)),
        ("Grid (=area, cell)", generate_grid_array_coords_area(M, diameter, area_mode="cell")),
        ("Grid (inscripta, legacy)", generate_grid_array_coords(M, diameter)),
        ("Spiral (Archimedean)", generate_spiral_array_coords(M, diameter)),
        ("Concentric rings", generate_concentric_array_coords(M, diameter)),
        ("Random (seed=0)", generate_random_array_coords(M, diameter, seed=0)),
    ]
    radius = diameter / 2.0

    fig, axes = plt.subplots(1, len(specs), figsize=(4.5 * len(specs), 5))
    for ax, (name, coords) in zip(axes, specs):
        circle = patches.Circle((0, 0), radius, linewidth=1.5, edgecolor='gray',
                                 facecolor='none', linestyle='--')
        ax.add_patch(circle)
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', marker='x', s=60)
        for k, (x, y, _) in enumerate(coords):
            ax.text(x + 0.005, y + 0.005, str(k), fontsize=8, color='navy')
        # La grilla equivalente por area puede salirse del circulo de referencia:
        # el limite se toma de las coordenadas, no del radio.
        lim = max(radius, float(np.abs(coords[:, :2]).max())) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n(M={coords.shape[0]}, D={diameter} m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Topology gallery: eyeball the four 2D topologies (M, diameter) ---
    _plot_topology_gallery(M=12, diameter=0.30)

    # 1. Define room and array setup parameters
    room_dims = np.array([6.0, 5.0, 2.5])
    M = 8
    d_min = 0.02
    d_max = 0.30
    
    # Source and interference setup parameters
    radius_source = 1.0   # Fuente más cerca del arreglo
    radius_interf = 1.8   # Interferencias más alejadas
    delta_ang_deg = 30.0  # Espaciado angular
    
    # 2. Compute array coordinates and center
    mic_coords = generate_log_array_coords(M, d_min, d_max, room_dims)
    array_center = room_dims / 2.0
    
    # 3. Create the Matplotlib figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    interference_cases = [1, 2, 3]
    
    for ax, n_int in zip(axes, interference_cases):
        # Generate sources for the current case using the updated function
        source_pos, interferences_pos = generate_source_and_interferences(
            N_interferences=n_int, 
            radius_source=radius_source,
            radius_interf=radius_interf, 
            delta_ang_deg=delta_ang_deg, 
            array_center=array_center
        )
        
        # Plot room boundary (Rectangle)
        room_patch = patches.Rectangle(
            (0, 0), room_dims[0], room_dims[1], 
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(room_patch)
        
        # Plot the microphone array
        ax.scatter(mic_coords[:, 0], mic_coords[:, 1], c='blue', marker='x', label='Mic Array')
        
        # Plot the target source (Green)
        ax.scatter(source_pos[0], source_pos[1], c='green', marker='o', s=100, label='Target Source')
        
        # Plot the interferences and add their tags
        for i in range(n_int):
            ax.scatter(interferences_pos[i, 0], interferences_pos[i, 1], c='red', marker='v', s=80)
            
            # Tag text positioning (slightly offset from the point)
            offset_x = 0.1
            offset_y = 0.1
            ax.text(
                interferences_pos[i, 0] + offset_x, 
                interferences_pos[i, 1] + offset_y, 
                f'Int {i+1}', 
                fontsize=10, color='red', weight='bold'
            )
            
        # Plot formatting
        ax.set_xlim(-0.5, room_dims[0] + 0.5)
        ax.set_ylim(-0.5, room_dims[1] + 0.5)
        ax.set_aspect('equal') # Keep physical proportions true
        ax.set_title(f'{n_int} Interference(s)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, linestyle=':', alpha=0.7)
        
        if n_int == 1:
            ax.legend(loc='upper left')
            
    plt.tight_layout()
    plt.show()