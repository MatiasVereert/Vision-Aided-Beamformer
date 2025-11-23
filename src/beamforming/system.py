import numpy as np 
from scipy.spatial import KDTree

# Asegúrate que estos imports existan en tu proyecto
from utils.geometry import spatial_grid
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.algorithms.weights import compute_fixed_weights_optimized

class AdaptiveBeamformer:

    def __init__(self, MicArrayObj, K: int, fs: int, fmin, fmax):
        # 1. Extraer datos del objeto MicArray
        # Guardamos las coordenadas crudas para la matemática
        self.mic_coords = MicArrayObj.coordinates 
        self.M = self.mic_coords.shape[0] # Definimos M PRIMERO
        
        self.K = K
        self.fs = fs 
        self.fmin = fmin
        self.fmax = fmax

        # 2. Estado del Procesamiento
        # AHORA SÍ podemos usar self.M porque ya existe
        self.buffer = np.zeros((self.M, K), dtype=np.float32)
        self.wa = None 
        
        self.current_wq = None
        self.current_Ca = None
        
        # 3. Estado de Foco (Coordenadas)
        self.active_coords = np.zeros(3)  # Dónde estamos mirando realmente
        self.target_coords = None         # Dónde dice la cámara

        # 4. Banco de Datos
        self.grid_points = None
        self.weight_bank = None
        self.grid_tree = None

    def generate_bank(self, radius, azimut, inclination, points, center):
        """
        Calcula el banco de pesos (Offline).
        radius, azimut, inclination: Son los 'deltas' (anchos) del ROI.
        """
        print("[SISTEMA] Generando banco de pesos...")
        
        # 1. Generar Grilla
        grid_points = spatial_grid(
            delta_radius=radius, 
            delta_azimut=azimut, 
            delta_inclination=inclination, 
            center=center, 
            points=points, 
            mode='cart' # Importante pedir cartesianas para el KDTree
        )

        # Parámetros de robustez fijos (puedes pasarlos como args si quieres)
        delta_r_robust = 0.15
        delta_az_robust = np.deg2rad(5)
        delta_inc_robust = np.deg2rad(5)

        weights_storage = []
        total = len(grid_points) # CORREGIDO: Usamos la variable, no la función

        for i, point in enumerate(grid_points):
            
            if i % 10 == 0: print(f"Calculando: {i}/{total}", end='\r')

            C, h, Ca = build_region_constraints(
                      Rs=point,
                      delta_r=delta_r_robust,
                      delta_azimut=delta_az_robust,
                      delta_elevation=delta_inc_robust,
                      mic_array=self.mic_coords, # Usamos las coordenadas extraídas
                      fs=self.fs,
                      K=self.K,
                      f_min=self.fmin,
                      f_max=self.fmax
                      )
            
            w_q = compute_fixed_weights_optimized(C, h)
            
            # CORREGIDO: Append de tupla a la lista
            weights_storage.append( (w_q.flatten(), Ca) )

        # Guardar atributos
        self.grid_points = grid_points
        self.weight_bank = weights_storage
        self.grid_tree = KDTree(grid_points)
        print(f"\n[LISTO] Banco generado con {total} puntos.")

    def update_focal_point(self, camera_coords):
        """Interfaz pública: Recibe coordenadas y decide si actualizar."""
        self.target_coords = np.array(camera_coords)
        
        # CORREGIDO: Lógica de activación (Histéresis)
        # Si no tenemos foco activo o nos movimos > 5cm, actualizamos
        dist = np.linalg.norm(self.target_coords - self.active_coords)
        
        if self.current_wq is None or dist > 0.05: 
            self._refresh_weights()

    def _refresh_weights(self):
        """(Privado) Actualiza los filtros."""
        if self.grid_tree is None:
            raise RuntimeError("WeightsBank not generated. Call generate_bank() first.")

        # 1. Buscar
        _, idx = self.grid_tree.query(self.target_coords)

        # 2. Recuperar
        new_wq, new_Ca = self.weight_bank[idx]
        
        # 3. Protección de Memoria (wa)
        cols_Ca = new_Ca.shape[1]
        
        if self.wa is None or self.wa.shape[0] != cols_Ca:
            self.wa = np.zeros(cols_Ca, dtype=np.complex64) # O float32
            # print("DEBUG: Reset adaptativo")

        # 4. Asignar
        self.current_wq = new_wq
        self.current_Ca = new_Ca
        self.active_coords = self.grid_points[idx]