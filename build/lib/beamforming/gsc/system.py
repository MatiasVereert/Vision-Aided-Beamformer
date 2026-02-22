import numpy as np 
from scipy.spatial import KDTree
import os 
# Asegúrate que estos imports existan en tu proyecto
from utils.geometry import spatial_grid
from beamforming.gsc.region_constriant import build_region_constraints
from beamforming.gsc.weights import compute_fixed_weights_optimized

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
        self.data_log = []
       

        #adaptive
        self.MU = .1
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
                            "wq" : self.current_wq.copy(),
                            "wa" : self.current_wa.copy(),
                            "ca" : self.current_Ca.copy()
                        }
                self.data_log.append(snapshot)
            
        return output

    def process_block_vad(self, input_signal, record_weights=False):
            """
            Procesa un bloque de muestras realizando el filtrado GSC paso a paso.
            Incluye un VAD energético simple para robustez contra cancelación del target.
            """
            # 1. Definir dimensiones
            M, tot_samples = input_signal.shape
            
            # Validar dimensiones
            if M != self.M:
                raise ValueError(f"Input dimension mismatch. Expected {self.M} rows, got {M}")

            output = np.zeros(tot_samples, dtype=np.float32)

            # Configuración de grabación (Logging)
            interval = 0
            if record_weights:
                # Asegurar que FPS no sea 0 o infinito
                if self.FPS <= 0: self.FPS = 30
                interval = int(self.fs / self.FPS)
                if interval < 1: interval = 1
                print(f"Recording Activated with: {self.FPS} FPS (Interval: {interval})")

            # --- PARÁMETROS VAD INTERNO (Robustez) ---
            # Estimadores de energía para decidir cuándo adaptar
            avg_energy = 0.0
            # Umbral relativo: Si la señal instantánea supera X veces al promedio, 
            # asumimos que es voz fuerte (target) y congelamos.
            vad_threshold_ratio = 3.0 
            
            # Factor de "olvido" suave para evitar deriva de pesos (Leaky NLMS)
            leakage = 0.9999 

            for i in range(tot_samples):
                # 2. Shift & Insert (TDL Update)
                # Desplazar buffer hacia la derecha (taps antiguos)
                self.buffer[:, 1:] = self.buffer[:, :-1]
                # Insertar nueva muestra en la posición 0
                self.buffer[:, 0] = input_signal[:, i] 

                # Aplanar para producto punto (M*K, 1)
                u_k = self.buffer.flatten()

                # --- ESTRUCTURA GSC ---
                
                # A) Rama Fija (Fixed Beamformer) -> Referencia del Target
                # y_q = w_q^T * u(k)
                y_q = np.dot(self.current_wq, u_k) 

                # B) Matriz de Bloqueo -> Referencia de Ruido/Interferencia
                # x_a = Ca^T * u_k
                x_a = np.dot(self.current_Ca.T, u_k)

                # Inicializar wa si es la primera vez (Lazy Init)
                if self.current_wa is None:
                    self.current_wa = np.zeros_like(x_a) 

                # C) Rama Adaptativa
                # y_a = wa^T * x_a
                y_a = np.dot(self.current_wa, x_a)

                # D) Salida del Sistema (Error de cancelación)
                # e = y_q - y_a
                error = y_q - y_a 
                output[i] = error

                # --- LÓGICA DE CONTROL DE ADAPTACIÓN (VAD) ---
                # Calculamos energía instantánea de la salida fija (proxy del target)
                inst_power = y_q**2
                
                # Promedio exponencial (Slow attack, slow decay)
                avg_energy = 0.99 * avg_energy + 0.01 * inst_power
                
                # Decisión: ¿Adaptar o Congelar?
                # Si la energía actual es mucho mayor que el promedio reciente, 
                # probablemente entró el locutor -> NO adaptar para no cancelarlo.
                if inst_power > (avg_energy * vad_threshold_ratio):
                    effective_mu = 0.0 # Congelar
                else:
                    effective_mu = self.MU # Adaptar (aprender ruido)

                # --- ACTUALIZACIÓN DE PESOS (NLMS) ---
                if effective_mu > 0:
                    # Energía del vector de referencia de ruido
                    energy_xa = np.dot(x_a, x_a) + self.EPSILON
                    
                    # Update rule: w(n+1) = w(n) + mu * e * x / |x|^2
                    update = effective_mu * error * x_a / energy_xa
                    
                    # Aplicamos update + leakage (estabilidad a largo plazo)
                    self.current_wa = (self.current_wa * leakage) + update

                # --- GRABACIÓN DE ESTADO ---
                if record_weights and (i % interval == 0):
                    time_sec = i * (1/self.fs)
                    # IMPORTANTE: Usar .copy() con paréntesis, sino guarda la referencia al método
                    snapshot = {
                        "time": time_sec,
                        "wq" : self.current_wq.copy(), 
                        "wa" : self.current_wa.copy(),
                        "ca" : self.current_Ca.copy(),
                        "mu_eff": effective_mu # Útil para debuguear el VAD
                    }
                    self.data_log.append(snapshot)
                
            return output


