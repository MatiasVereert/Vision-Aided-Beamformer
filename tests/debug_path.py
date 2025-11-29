import os
import h5py

target = "data/dEchorate/dEchorate_rirs_gzip7.hdf5"
abs_path = os.path.abspath(target)

print(f"--- DIAGNÓSTICO DE ARCHIVO ---")
print(f"Ruta absoluta: {abs_path}")

if os.path.exists(target):
    size_bytes = os.path.getsize(target)
    print(f"Tamaño en disco (Python): {size_bytes} bytes ({size_bytes / (1024**3):.2f} GB)")
    
    try:
        # Intentamos abrir con modo 'latest' por si es un tema de versión de formato
        with h5py.File(target, 'r', libver='latest') as f:
            print("¡Éxito abriendo HDF5!")
            print(f"Claves: {list(f.keys())}")
            # Intentar acceder al grupo problemático
            if 'rir' in f:
                print(f"Grupo 'rir' accesible. Elementos: {len(f['rir'])}")
    except Exception as e:
        print(f"\n[FALLO INTERNO HDF5]: {e}")
else:
    print("¡El archivo no existe en la ruta relativa!")
    