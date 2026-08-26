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
#   - Todas quedan INSCRIPTAS en un circulo del `diameter` dado: ningun microfono
#     cae fuera del circulo, y al menos uno lo toca (se usa el diametro completo),
#     de modo que la apertura fisica sea comparable entre topologias.
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
    "grid": generate_grid_array_coords,
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
        ("Rectangular grid", generate_grid_array_coords(M, diameter)),
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
        lim = radius * 1.2
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