import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

def procesar_video_svd(input_path, output_path, n_components=2):
    """
    Lee un video, elimina el fondo estático (paredes) usando SVD y guarda el resultado.
    
    Args:
        input_path (str): Ruta al archivo de video de entrada (.mp4, .avi).
        output_path (str): Ruta donde se guardará el video filtrado.
        n_components (int): Número de componentes a eliminar (fuerza del filtro).
                            1-2 para fondos muy estáticos.
                            3-5 si la cámara se mueve un poco.
    """
    
    # --- 1. Cargar el video ---
    print(f"Abriendo video: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        return

    # Obtener propiedades del video original
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_buffer = []

    # Leer todos los frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir a escala de grises (SVD trabaja con intensidad, 1 canal)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_buffer.append(gray_frame)
    
    cap.release()
    
    # Convertir lista a array numpy (N_frames, Alto, Ancho)
    video_stack = np.array(frames_buffer)
    print(f"Video cargado en memoria RAM. Dimensiones: {video_stack.shape}")

    # --- 2. Procesamiento SVD ---
    print("Iniciando descomposición SVD (esto puede tardar unos segundos)...")
    
    n_frames, h, w = video_stack.shape
    
    # Aplanar: (Frames, Pixeles)
    # Nota: Usamos float32 para precisión matemática
    X = video_stack.reshape(n_frames, -1).astype(np.float32)
    
    # Calcular SVD Truncada (Encontrar el fondo)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)
    
    # Reconstruir el fondo
    X_background = svd.inverse_transform(svd.transform(X))
    
    # Restar el fondo (Obtener la señal dinámica/raíces)
    X_filtered = X - X_background
    
    # --- 3. Post-Procesamiento (Normalización para Video) ---
    print("Renderizando video de salida...")

    # A. Clip negativo: Lo que es más oscuro que el fondo se vuelve 0
    # (Si quieres ver sombras oscuras, usa np.abs(X_filtered) en su lugar)
    X_filtered = np.maximum(X_filtered, 0)
    
    # B. Normalizar a 0-255 para formato de video
    # Evitamos dividir por cero si la imagen es negra
    max_val = np.max(X_filtered)
    if max_val > 0:
        X_filtered = (X_filtered / max_val) * 255.0
    
    # Convertir a uint8 (formato de imagen estándar)
    video_limpio = X_filtered.astype(np.uint8).reshape(n_frames, h, w)

    # --- 4. Guardar Video ---
    # Usamos codec MJPG para .avi o mp4v para .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    for frame in video_limpio:
        out.write(frame)
        
    out.release()
    print(f"¡Éxito! Video filtrado guardado en: {output_path}")

# --- Bloque de Ejecución ---
if __name__ == "__main__":
    # CAMBIA ESTAS RUTAS POR LAS TUYAS
    ruta_entrada = "data/videos/001.AVI" 
    ruta_salida = "001_filtered.mp4"
    
    # Si el archivo existe, ejecutar
    if os.path.exists(ruta_entrada):
        procesar_video_svd(ruta_entrada, ruta_salida, n_components=5)
    else:
        # Ejemplo dummy para que no falle si copias y pegas sin cambiar la ruta
        print("Por favor, edita la variable 'ruta_entrada' con la ubicación real de tu video.")