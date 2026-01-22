import numpy as np 
from matplotlib import pyplot as plt
from beamforming.signal_model import near_field_steering_vector, near_field_steering_vector_multi
from scipy import signal, fft
import numpy as np
from propagation.free_field import space_delay

#Constants 
fs = 48000
f_test = 3000.0 
C_SOUND = 343
fmin = 200
fmax= 10000

#Array dimentions 
mic_spacing = 0.01
M = 12
lenght = M * mic_spacing
x_coords = (np.arange(M) - (M-1)/2)* mic_spacing

y_coords = z_coords = np.zeros(M)
mic_coords = np.stack([x_coords, y_coords, z_coords ], axis = 1) # M, 3

#Source Signal and ubicaiton
source_pos = [1.5 , 1.5, 0]
f = 1000
time = 1 #s
t = np.arange(0, 1 , 1/fs)
source_signal = np.sin(2 * np.pi * t * f)

#Noise signal
noise_pos = [-1.5 , 1.5, 0]
noise = np.random.rand(len(t)) - 0.5


plt.figure()
plt.plot(t, source_signal)
plt.plot(t, noise, color = 'r')
plt.show()

#obtain the inputs of the array
inp_signal  = (source_signal, fs, source_pos, mic_coords)
inp_noise  = (noise, fs, noise_pos, mic_coords)
array_input = inp_signal + inp_noise



# ---- Low Rank Beamforming Sum delay Design ---------
#Settings

M1 = 3
M2 = 4
P = 1

#Firstly, we obtain de steering vector in the position of the source
sv = near_field_steering_vector_multi(f , source_pos, fs, mic_coords, 1)

#We obtain te coeficients by lowering the rank of the 

steering_m = sv.reshape(M1, M2)
U, S, Vh = np.linalg.svd(steering_m, full_matrices= False)

I1 = np.identity(M1)
I2 = np.identity(M2)

H2_underlined = np.zeros((M, P * M1), dtype=complex)
H1_underlined = np.zeros((M, P * M2), dtype=complex)

h2_underlined = np.zeros((P * M2), dtype=complex)
h1_underlined = np.zeros((P * M1), dtype=complex)

#Compose the h weights for each
for p in range(P):
    sigma_p_sqrt = np.sqrt(S[p]) 
    
    # 1. Definir Vectores Columna (M1, 1) y (M2, 1)
    h1_p = (U[:, p] * sigma_p_sqrt).reshape(-1, 1)
    h2_p = (Vh[p, :] * sigma_p_sqrt).reshape(-1, 1)

    # 2. Construir Bloques Kronecker
    # H2_p: (M2, 1) kron (M1, M1) -> Bloque (M, M1) [Ancho M1]
    H2_p = np.kron(h2_p.conj(), I1) 
    
    # H1_p: (M2, M2) kron (M1, 1) -> Bloque (M, M2) [Ancho M2]
    H1_p = np.kron(I2, h1_p)

    # --- GUARDADO EN MATRICES GIGANTES (Usan índices cruzados) ---
    
    # Matriz H2: Sus bloques tienen ancho M1
    idx_start_mat_2 = p * M1
    idx_end_mat_2   = (p + 1) * M1
    H2_underlined[:, idx_start_mat_2 : idx_end_mat_2] = H2_p
    
    # Matriz H1: Sus bloques tienen ancho M2 (¡OJO AQUÍ!)
    idx_start_mat_1 = p * M2
    idx_end_mat_1   = (p + 1) * M2
    H1_underlined[:, idx_start_mat_1 : idx_end_mat_1] = H1_p
    
    # --- GUARDADO EN VECTORES COLUMNA (Usan sus dimensiones naturales) ---
    
    # Vector h2: Tiene largo M2
    idx_start_vec_2 = p * M2
    idx_end_vec_2   = (p + 1) * M2
    # Asignamos directo (sin .T porque ya es columna, y cuidado con el conj si el libro lo pide)
    h2_underlined[idx_start_vec_2 : idx_end_vec_2] = h2_p.flatten()
    
    # Vector h1: Tiene largo M1
    idx_start_vec_1 = p * M1
    idx_end_vec_1   = (p + 1) * M1
    h1_underlined[idx_start_vec_1 : idx_end_vec_1] = h1_p.flatten()


# ------- Procces the Signals ---------
# singal->convert to frecuency-> process-> invert to time 


n_window = 1024
n_overlap = 512

# 1. Transformada al Dominio de la Frecuencia (STFT)
# X shape: (M, Freqs, Time) -> (12, 513, N_frames)
f_axis, t_axis, X = signal.stft(x=array_input, 
                                fs=fs, 
                                nperseg=n_window, 
                                noverlap=n_overlap, 
                                window='hann', 
                                axis=1)

# Matriz para guardar el resultado (Frecuencia x Tiempo)
# Ya no tiene dimensión de micrófonos porque el beamformer los colapsa a 1.
Y_stft = np.zeros((X.shape[1], X.shape[2]), dtype=complex)

print("Procesando señal Banda Ancha (Low-Rank por frecuencia)...")

# --- 2. BUCLE DE PROCESAMIENTO (Banda Ancha) ---
# Iteramos sobre cada bin de frecuencia k
for k, freq_val in enumerate(f_axis):
    
    # Evitamos procesar DC (0Hz) o Nyquist si dan problemas numéricos,
    # y limitamos a fmax para no procesar ruido ultrasónico innecesario.
    if freq_val < fmin or freq_val > fmax:
        continue

    # -----------------------------------------------------------
    # PASO A: Recalcular la Matemática Narrowband para ESTA frecuencia
    # -----------------------------------------------------------
    
    # 1. Steering Vector actual (depende de freq_val)
    # Nota: Usamos tu función con freq_val, NO con la f fija de 1000
    sv_k = near_field_steering_vector_multi(freq_val, source_pos, fs, mic_coords, 1)
    
    # 2. Reshape y SVD
    steering_m = sv_k.reshape(M1, M2)
    U, S, Vh = np.linalg.svd(steering_m, full_matrices=False)
    
    # -----------------------------------------------------------
    # PASO B: Construir los Filtros Separables (h1 y h2)
    # -----------------------------------------------------------
    # Usamos P=1 como definiste, pero sumamos las ramas si P>1
    
    # Tomamos la "foto" de los micrófonos en esta frecuencia k
    # X_snapshot shape: (M, Time) -> lo pasamos a (M1, M2, Time)
    X_k_snapshot = X[:, k, :].reshape(M1, M2, -1)
    
    output_frec_k = np.zeros(X_k_snapshot.shape[2], dtype=complex)
    
    for p in range(P):
        sigma_sqrt = np.sqrt(S[p])
        
        # Vectores h para el modo p (Conjugados porque son filtros W = h*)
        # h1 (Filas): (M1,)
        w1_p = (U[:, p] * sigma_sqrt).conj()
        # h2 (Columnas): (M2,)
        w2_p = (Vh[p, :] * sigma_sqrt).conj()
        
        # -----------------------------------------------------------
        # PASO C: Aplicar Filtro Separable (Eficiencia Pura)
        # -----------------------------------------------------------
        # En vez de Kronecker gigante, filtramos columnas y luego filas.
        
        # 1. Filtrar dimensión M2 (Columnas)
        # X_k_snapshot (M1, M2, Time) @ w2_p (M2,) -> Intermedio (M1, Time)
        # Usamos np.tensordot o einsum. Einsum es más explícito:
        # "ijk, j -> ik": Indices (M1, M2, Time), (M2) -> (M1, Time)
        intermedio = np.einsum('ijk, j -> ik', X_k_snapshot, w2_p)
        
        # 2. Filtrar dimensión M1 (Filas)
        # w1_p (M1,) @ Intermedio (M1, Time) -> Salida (Time)
        salida_rama = w1_p @ intermedio
        
        # Sumamos la contribución de este rango p
        output_frec_k += salida_rama
        
    # Guardamos la fila de frecuencia procesada en la matriz final
    Y_stft[k, :] = output_frec_k

# --- 3. INVERSA AL TIEMPO (ISTFT) ---
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap, window='hann')

# Ajuste de longitud por padding
min_len = min(len(t), len(signal_out))
signal_out = signal_out[:min_len]

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.title("Señal Original (Sucia) - Mic 1")
plt.plot(t[:min_len], array_input[0, :min_len].real, label='Mic 1')
plt.legend()

plt.subplot(3,1,2)
plt.title("Salida Beamformer Low-Rank")
plt.plot(t[:min_len], signal_out.real, color='g', label='Salida')
plt.legend()

plt.subplot(3,1,3)
plt.title("Comparación Espectral")
plt.psd(array_input[0, :], Fs=fs, NFFT=1024, label='Input (Mic 1)')
plt.psd(signal_out, Fs=fs, NFFT=1024, label='Output (Beamformed)')
plt.legend()

plt.tight_layout()
plt.show()





