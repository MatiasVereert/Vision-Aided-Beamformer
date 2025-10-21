import numpy as np 
# from matplotlib import pytplot as plt 

def near_field_steering_vector(f, Rs,fs, mic_array, K =1, c=343):
    """
    (Código de la función sin modificar la lógica)
    """

    #Calculate the distance and fase relativet to the array, asuming is centered in the origin.
    source_distance = np.linalg.norm(Rs)
    phase_reference = np.exp( 1j * 2 * np.pi * f * source_distance / c)
    normalization_factor = source_distance / phase_reference

    #Distances of each element respect to the source (shape Mx1)
    distances = np.linalg.norm( Rs - mic_array, axis = 1).reshape(-1, 1)

    #Propagation delay for each microphone (shape Mx1)
    mic_delay = distances/c 

    #tabs delays shape 1xK  
    T = 1/fs
    tabs = np.arange(K)
    tab_delay = tabs * T

    #Defining the propagation vector (Brodcast (Mx1)(1xK)->(MxK))
    steering_vector = np.exp(1j * 2 *np.pi *f * (mic_delay -tab_delay) ) / distances
    normalized_steering_vector = normalization_factor * steering_vector

    #Colapsing the matriz into colums 

    normalized_steering_vector = normalized_steering_vector.reshape(-1,1)


    return normalized_steering_vector, mic_array, K


# --- CONFIGURACIÓN DE PRUEBA (Externo a la función) ---
M = 3
d = 0.05  
mic_array_test = np.array([[-d, 0, 0], [0, 0, 0], [d, 0, 0]])
Rs_test = np.array([1.0, 0.0, 0.0])
f_test = 1000.0
fs_test = 8000
K_test = 1 # K = 1 para banda estrecha
c_test = 343

# Ejecutar la función (recuperando variables para el análisis)
a_mk, array, K_calc = near_field_steering_vector(f_test, Rs_test, fs_test, mic_array_test, K_test, c_test)

# --- ANÁLISIS ---
central_mic_index = 1 
central_element_value = a_mk[central_mic_index, 0]

# A. Prueba de la Normalización de Fase
phase_deg = np.angle(central_element_value) * 180 / np.pi
print("--- Prueba de Normalización de Fase (K=1) ---")
print(f"Fase del Micrófono Central: {phase_deg:.4f} grados")
# Criterio: La fase debe ser muy cercana a 0.
print(f"Estado: OK (Fase cercana a 0)") if np.abs(phase_deg) < 1e-4 else print("Estado: FALLO (Fase no nula)")


# B. Prueba de la Normalización de Amplitud
amplitude = np.abs(central_element_value)

print("\n--- Prueba de Normalización de Amplitud (K=1) ---")
print(f"Amplitud del Micrófono Central: {amplitude:.4f}")
# Criterio: La amplitud debe ser muy cercana a 1.0.
print(f"Estado: OK (Amplitud cercana a 1.0)") if np.abs(amplitude - 1.0) < 1e-4 else print("Estado: FALLO (Amplitud no unitaria)")
