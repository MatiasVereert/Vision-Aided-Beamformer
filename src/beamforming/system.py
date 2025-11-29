import numpy as np 
from scipy.spatial import KDTree
import os 
# Asegúrate que estos imports existan en tu proyecto
from utils.geometry import spatial_grid
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.algorithms.weights import compute_fixed_weights_optimized

class AdaptiveBeamformer:

    def __init__(self, mic_array, K: int, fs: int, fmin, fmax):
        # 1. Extraer datos del objeto MicArray
        # Guardamos las coordenadas crudas para la matemática
        self.mic_coords = mic_array 
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
        self.current_wa = None
        
        # 3. Estado de Foco (Coordenadas)
        self.active_coords = np.zeros(3)  # Dónde estamos mirando realmente
        self.target_coords = None         # Dónde dice la cámara

        # 4. Banco de Datos
        self.grid_points = None
        self.weight_bank = None
        self.grid_tree = None

        # Recroding Atributes
        self.FPS = 30
        self.data_log = {}
       

        #adaptive
        self.MU = .65
        self.EPSILON = 10e-12

        self.bank_metadata = {}

    def generate_bank(self, r_spam, az_spam, inc_spam, points, center):
        """
        Calcula el banco de pesos (Offline).
        radius, azimut, inclination: Son los 'deltas' (anchos) del ROI.
        """
        print("[SISTEMA] Generando banco de pesos...")
        
        # 1. Generar Grilla
        grid_points = spatial_grid(
            delta_radius= r_spam, 
            delta_azimut= az_spam, 
            delta_inclination= inc_spam , 
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

        # Save the metadata 
        self.bank_metadata['r_spam'] = r_spam
        self.bank_metadata['az_spam'] = az_spam
        self.bank_metadata['inc_spam'] = inc_spam
        self.bank_metadata['points'] = points
        self.bank_metadata['center'] = center


    def save_weights_to_disk(self, filename = "weights_bank.npz", folder = "weights_storage" ):
        #Verifies that the bank has been computed
        if self.weight_bank is None:
            print('ERROR: Please generate bank before saving.')
            return
        
        full_path = os.path.join(folder, filename)

        #Creates new folder if its necesary 
        if not os.path.exists(folder):
            print(f'[System]: creating new folder as {folder}')
            os.makedirs(folder)
        
        weight_bank_obj = np.array(self.weight_bank, dtype = object)

        
        print('Saving Filters')
        np.savez_compressed(file=full_path,
                            metadata = self.bank_metadata,
                            weights_bank = weight_bank_obj,
                            grid_points = self.grid_points
                            )
        print(f'The bank has been saved into: {filename}')

    def load_weights_from_disk(self, full_path ):
        if not os.path.exists(full_path):
            print(f"[System]: the file does not exist")
            return False
        
        try:
            data = np.load(full_path, allow_pickle= True)

            self.grid_points = data['grid_points']
            self.weight_bank = data['weights_bank']
            self.bank_metadata = data['metadata'].item()

            self.grid_tree = KDTree(self.grid_points)   

            print("The bank has been loaded")

        except Exception as e:
            print(f'[Critical Error] Loading the file {e}')
            


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
        
        if self.current_wa is None or self.current_wa.shape[0] != cols_Ca:
            self.current_wa = np.zeros(cols_Ca, dtype=np.float32) # O float32
            # print("DEBUG: Reset adaptativo")

        # 4. Asignar
        self.current_wq = new_wq
        self.current_Ca = new_Ca
        self.active_coords = self.grid_points[idx]

    def process_block(self, input_signal, record_weights = False):
        # 1. Definir dimensiones (M filas, N columnas)
        # Asumimos que input_signal viene como (M_mics, Tot_samples)
        M, tot_samples = input_signal.shape
        data = {}
        
        # Validar que M coincida con tu array
        if M != self.M:
             raise ValueError(f"Input dimension mismatch. Expected {self.M} rows, got {M}")

        output = np.zeros(tot_samples, dtype=np.float32)

        # Buffer size ya está definido en __init__ (self.buffer)    
        if record_weights == True:
            interval = int(self.fs/self.FPS)
            self.FPS = self.fs / interval

            print("Recroding Activated with: ", self.FPS, "FPS")


        for i in range(tot_samples):
            # 2. Shift & Insert (CORREGIDO)
            self.buffer[:, 1:] = self.buffer[:, :-1]
            
            # ¡OJO AQUÍ! Usamos [:, i] para agarrar la columna (muestra actual de todos los mics)
            self.buffer[:, 0] = input_signal[:, i] 

            # Adapt dimensions
            u_k = self.buffer.flatten()

            # --- RAMAS (Sin np.conj porque es Real) ---
            
            # Fixed output branch
            # Si wq es real, dot(wq.T, u_k) es suficiente
            y_q = np.dot(self.current_wq, u_k) 

            # Blocking matrix
            # x_a = Ca^T * u_k
            x_a = np.dot(self.current_Ca.T, u_k)

            # Initialize wa (Lazy)
            if self.current_wa is None:
                self.current_wa = np.zeros_like(x_a) 

            # Adaptive Branch
            # y_a = wa^T * x_a
            y_a = np.dot(self.current_wa, x_a)

            # Error (OUTPUT)
            error = y_q - y_a 
            output[i] = error

            # --- NLMS Update (Versión Real) ---
            # Energía = x_a dot x_a (Suma de cuadrados)
            energy = np.dot(x_a, x_a) + self.EPSILON

            # Update: mu * error * vector / energia
            update = self.MU * error * x_a / energy

            self.current_wa += update

            #Recording
            if record_weights == True and (i%interval ==0):
                time_sec = i * (1/self.fs)
                snapshot = {"time": time_sec,
                            "wq" : self.current_wq.copy,
                            "wa" : self.current_wa.copy,
                            "ca" : self.current_Ca.copy
                        }
                self.data_log.append(snapshot)
            
        return output


