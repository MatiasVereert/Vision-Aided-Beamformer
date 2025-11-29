# tools/data_loader.py

import h5py
import numpy as np
from scipy.spatial import KDTree
from numpy.linalg import norm
from typing import Tuple, List, Dict

def load_sriracha_selection(
    ruta_archivo: str,  
    n_mics_select: int = 8,
    interference_quartiles: List[float] = [1.0, 0.75, 0.50, 0.25]
) -> Dict:
    """
    Carga y selecciona los índices y coordenadas esenciales del dataset SRIRACHA.
    La selección se basa en los 8 micrófonos centrales y fuentes espaciadas por distancia.

    Args:
        ruta_archivo (str): Ruta completa al archivo H5 (e.g., 'SR1-C1.h5').
        n_mics_select (int): Número de micrófonos centrales a seleccionar.
        interference_quartiles (List[float]): Coeficientes de distancia para interferencias.

    Returns:
        Dict: Diccionario con coordenadas y arrays de índices.
    """
    try:
        with h5py.File(ruta_archivo, 'r') as f:
            
            # --- Carga inicial de datos ---
            mics = f['data/location/receiver'][:]
            fuentes = f['data/location/source'][:]
            fs = f['metadata/sampling_rate'][()]
            
            mics[:, 0] = -1 * mics[:, 0]
            fuentes[:, 0] = -1 * fuentes[:, 0]
            
            # 1. SELECCIÓN DE MICRÓFONOS (8 Centrales)
            mics_tree = KDTree(mics)
            _, mic_idx_unsorted = mics_tree.query([0, 0, 0], k=n_mics_select) 
            
            mic_idx_selected = np.sort(mic_idx_unsorted)
            # 2. SELECCIÓN DE FUENTE TARGET (Más Cercana al centro)
            distances = norm(fuentes, axis=1)
            target_idx = np.argmin(distances) 
            
            # 3. SELECCIÓN DE INTERFERENCIAS (Cuartiles de Distancia)
            D_max = np.max(distances)
            interference_indices_list = []
            
            for Q in interference_quartiles:
                D_target = Q * D_max
                idx_found = np.argmin(np.abs(distances - D_target))
                interference_indices_list.append(idx_found)

            # 4. LIMPIEZA Y CONSOLIDACIÓN DE ÍNDICES
            interference_indices_raw = np.unique(interference_indices_list)
            # Excluir el índice del target
            final_interference_indices = interference_indices_raw[interference_indices_raw != target_idx]
            
            return {
                'fs': int(fs),
                'mic_coords_full': mics,
                'fuentes_coords_full': fuentes,
                # --- Índices y Coordenadas para el procesamiento ---
                'mic_idx': mic_idx_selected,
                'target_idx': target_idx,
                'interference_idx': final_interference_indices,
                'mic_coords_final': mics[mic_idx_selected],
                'target_pos': fuentes[target_idx],
                'interference_pos': fuentes[final_interference_indices],
            }

    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo H5 no encontrado en la ruta: {ruta_archivo}")
    except Exception as e:
        raise Exception(f"Error procesando H5: {e}")
    
# src/utils/data_loader.py (Adición a tu módulo)

def get_rir_data_arrays(ruta_archivo: str, data_selection: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Lee las RIRs del archivo H5 usando los índices pre-calculados.

    Args:
        ruta_archivo (str): Ruta al archivo H5.
        data_selection (Dict): Diccionario que contiene 'mic_idx', 'target_idx', 'interference_idx', etc.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (RIR_Target, RIR_Interferencia, FS_nativo).
    """
    mic_idx_selected = data_selection['mic_idx']
    target_idx = data_selection['target_idx']
    final_interference_indices = data_selection['interference_idx']
    print("Comenzando impotacion de RIRs")
    try:
            with h5py.File(ruta_archivo, 'r') as f:
                rir_dataset = f['data/impulse_response']
                fs_native = f['metadata/sampling_rate'][()]
                
                # --- CORRECCIÓN PARA EL TARGET ---
                # target_idx es un escalar (int), así que h5py lo maneja bien combinado con una lista.
                # No obstante, para máxima seguridad, hacemos lo mismo:
                
                # 1. Cargar la fuente completa (1, 64, N)
                rir_target_full_mics = rir_dataset[target_idx, :, :] 
                
                # 2. Recortar micrófonos en RAM (NumPy)
                rir_target = rir_target_full_mics[mic_idx_selected, :]
                
                
                # --- CORRECCIÓN PARA INTERFERENCIAS (Aquí fallaba) ---
                
                # 1. Paso H5 (Disco -> RAM): 
                # Traemos las fuentes de interferencia, pero TODOS los micrófonos (usando :)
                rirs_interf_temp = rir_dataset[final_interference_indices, :, :]
                
                # 2. Paso NumPy (RAM):
                # Ahora que 'rirs_interf_temp' está en memoria, recortamos los micros.
                rirs_interference = rirs_interf_temp[:, mic_idx_selected, :]
                print("Finalizando importacion de RIRs")
                return rir_target, rirs_interference, float(fs_native)
                
    except Exception as e:
        raise Exception(f"Error cargando arrays de RIRs: {e}") 