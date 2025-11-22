import os
from datetime import datetime

# --- CONFIGURACIÓN ---
# Carpeta donde se guardarán los historiales
OUTPUT_DIR = 'snapshots_contexto'

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'.py', '.toml', '.md', '.json'}

# Carpetas a ignorar
IGNORE_DIRS = {
    '.git', 
    '__pycache__', 
    'venv', 
    'env', 
    '.idea', 
    '.vscode',
    'build',
    'dist',
    'beamfomig_python.egg-info'
}

def is_ignored(path_part):
    """Verifica si una carpeta debe ser ignorada."""
    return path_part in IGNORE_DIRS or path_part.endswith('.egg-info')

def main():
    # 1. Crear directorio de snapshots si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Carpeta '{OUTPUT_DIR}' creada.")

    # 2. Generar nombre de archivo con fecha y hora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"contexto_{timestamp}.txt"
    full_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Iniciando exportación a: {full_path}...")
    
    try:
        with open(full_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"SNAPSHOT DEL PROYECTO - FECHA: {timestamp}\n")
            out_f.write("===========================================\n\n")

            for root, dirs, files in os.walk("."):
                # Evitar entrar en la propia carpeta de backups para no leer los txt viejos
                if OUTPUT_DIR in root:
                    continue

                # Filtrar directorios in-place
                dirs[:] = [d for d in dirs if not d.startswith('.') and not is_ignored(d)]
                
                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext in ALLOWED_EXTENSIONS:
                        file_path = os.path.join(root, file)
                        
                        # Encabezado de archivo
                        out_f.write(f"\n{'='*60}\n")
                        out_f.write(f"ARCHIVO: {os.path.relpath(file_path, '.')}\n")
                        out_f.write(f"{'='*60}\n")
                        
                        # Contenido
                        try:
                            with open(file_path, 'r', encoding='utf-8') as in_f:
                                out_f.write(in_f.read())
                                out_f.write("\n") 
                            print(f"--> Agregado: {os.path.relpath(file_path, '.')}")
                        except Exception as e:
                            print(f"[!] Error leyendo {file_path}: {e}")
                            out_f.write(f"[ERROR LEYENDO ARCHIVO: {e}]\n")

        print(f"\n[OK] Snapshot guardado exitosamente en: {full_path}")

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] No se pudo escribir el archivo: {e}")

if __name__ == "__main__":
    main()