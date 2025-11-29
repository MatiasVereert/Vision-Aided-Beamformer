import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necesario para plots 3D

# Imports del sistema
from beamforming.array.mic_array import ULA 
from beamforming.system import AdaptiveBeamformer

def test_pipeline():
    print("=== INICIANDO TEST DE SISTEMA ADAPTATIVO ===")

    # ---------------------------------------------------------
    # 1. CONFIGURACIÓN FÍSICA
    # ---------------------------------------------------------
    fs = 48000
    K = 25
    c = 343.0
    
    # Frecuencias de diseño (Broadband)
    f_min = 500.0
    f_max = 4000.0

    # Definir el Array (ULA de 7 micrófonos, espaciado 4cm)
    print("\n[1] Inicializando Arreglo de Micrófonos...")
    mic_array = ULA(M=7, d=0.04)
    mic_array.Plot_Geometry() # Esto debería abrir una ventanita 2D
    plt.close() # Cerramos para seguir

    mic_array = mic_array.coordinates
    # ---------------------------------------------------------
    # 2. INSTANCIAR EL MOTOR (Beamformer)
    # ---------------------------------------------------------
    print("[2] Inicializando Motor Beamformer...")
    bf = AdaptiveBeamformer(
        mic_array=mic_array, 
        K=K, 
        fs=fs, 
        fmin=f_min, 
        fmax=f_max
    )
    
    # Verificamos estado inicial
    print(f"    -> Buffer inicializado: {bf.buffer.shape}")
    print(f"    -> Pesos actuales: {bf.current_wq}") # Debería ser None

    # ---------------------------------------------------------
    # 3. GENERACIÓN DE BANCO DE PESOS (Offline Simulation)
    # ---------------------------------------------------------
    print("\n[3] Generando Banco de Pesos (Grilla Espacial)...")
    
    # Definimos un ROI (Región de Interés)
    # Centro: 1 metro de distancia, 90° Inclinación (Plano Horizontal), 90° Azimut (Frente)
    # Nota: Usamos radianes porque tus funciones trigonométricas lo esperan
    roi_center_sph = np.array([1.0, np.pi/2, np.pi/2]) 
    
    # Deltas (Ancho del ROI)
    delta_r = 0.4           # +/- 20 cm
    delta_inc = np.deg2rad(10) # +/- 5 grados verticales
    delta_az = np.deg2rad(60)  # +/- 30 grados horizontales (Barrido amplio)
    
    # Resolución
    points_per_dim = 6 # Total puntos = 6*6*6 = 216 puntos
    
    bf.generate_bank(
        r_spam=delta_r,
        az_spam=delta_az, 
        inc_spam=delta_inc,
        points=points_per_dim, 
        center=roi_center_sph
    )
    
    # Verificación
    if bf.grid_tree is not None:
        print(f"    -> ¡ÉXITO! Banco generado con {len(bf.weight_bank)} filtros.")
    else:
        print("    -> ERROR: El banco no se generó.")
        return

    # ---------------------------------------------------------
    # 4. SIMULACIÓN DE RUNTIME (Cámara -> Beamformer)
    # ---------------------------------------------------------
    print("\n[4] Probando Actualización de Foco (Runtime)...")
    
    # Simulamos que la cámara detecta una persona en (0.9m, un poco a la izquierda)
    # Convertimos una posición de prueba a cartesianas para simular la cámara
    test_r = 0.95
    test_az = np.deg2rad(110) # 20 grados a la izquierda
    test_el = np.pi/2         # Plano horizontal
    
    cam_x = test_r * np.sin(test_el) * np.cos(test_az)
    cam_y = test_r * np.sin(test_el) * np.sin(test_az)
    cam_z = test_r * np.cos(test_el)
    
    camera_target = np.array([cam_x, cam_y, cam_z])
    print(f"    -> Cámara detecta objetivo en: {camera_target.round(2)}")
    
    # LLAMADA CLAVE: Actualizar foco
    bf.update_focal_point(camera_target)
    
    # Verificar si se cargaron los pesos
    if bf.current_wq is not None:
        print(f"    -> Sistema Actualizado.")
        print(f"    -> Coordenada Activa (Grid): {bf.active_coords.round(2)}")
        print(f"    -> Filtro Fijo (wq) cargado. Forma: {bf.current_wq.shape}")
        print(f"    -> Filtro Adaptativo (wa) inicializado. Forma: {bf.wa.shape}")
    else:
        print("    -> ERROR: Los pesos no se actualizaron.")

    # ---------------------------------------------------------
    # 5. VISUALIZACIÓN (Debug Gráfico)
    # ---------------------------------------------------------
    print("\n[5] Graficando ROI...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Puntos de la grilla generada (Azules)
    grid = bf.grid_points
    ax.scatter(grid[:,0], grid[:,1], grid[:,2], c='blue', marker='.', alpha=0.3, label='Banco de Pesos')
    
    # Micrófonos (Negros)
    mics = mic_array.coordinates
    ax.scatter(mics[:,0], mics[:,1], mics[:,2], c='black', marker='s', s=50, label='Micrófonos')
    
    # Objetivo Cámara (Rojo)
    ax.scatter(camera_target[0], camera_target[1], camera_target[2], c='red', marker='x', s=100, label='Cámara (Target)')
    
    # Foco Real del Beamformer (Verde) - El vecino más cercano encontrado
    ax.scatter(bf.active_coords[0], bf.active_coords[1], bf.active_coords[2], c='lime', marker='o', s=80, label='Beamformer (Activo)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Visualización del Sistema: Banco vs. Realidad')
    ax.legend()
    
    # Ajustar vista para que parezca planta (desde arriba)
    ax.view_init(elev=90, azim=-90) 
    plt.show()

if __name__ == "__main__":
    test_pipeline()