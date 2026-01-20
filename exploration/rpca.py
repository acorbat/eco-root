
import cv2
import numpy as np
import os

def robust_pca(M):
    """
    Implementación de Robust PCA usando Inexact ALM (Augmented Lagrange Multiplier).
    Separa M en L (Low-Rank / Fondo) y S (Sparse / Raíces).
    
    Ref: Candès, Li, Ma, and Wright (2011). Robust Principal Component Analysis?
    """
    # Parámetros automáticos basados en la teoría
    n1, n2 = M.shape
    lambda_param = 1 / np.sqrt(max(n1, n2))
    
    # Inicialización
    Y = M / np.max(np.abs(M)) # Matriz de Lagrange
    S = np.zeros_like(M)      # Matriz Sparse (Raíces)
    L = np.zeros_like(M)      # Matriz Low-Rank (Pared)
    mu = 1.25 / np.linalg.norm(Y, 2) # Tasa de aprendizaje
    rho = 1.5                        # Factor de incremento de mu
    tol = 1e-7                       # Tolerancia de convergencia
    max_iter = 100                   # Límite de seguridad
    
    print("Iniciando iteraciones RPCA...")
    
    for i in range(max_iter):
        # 1. Actualizar L (Fondo) usando SVD y Thresholding
        # L = D_epsilon(M - S + Y/mu)
        temp_L = M - S + (1/mu) * Y
        U, Sigma, Vt = np.linalg.svd(temp_L, full_matrices=False)
        
        # Operador de umbralización para valores singulares (Soft Thresholding)
        thresh = 1/mu
        Sigma_new = np.maximum(Sigma - thresh, 0)
        
        L_new = np.dot(U * Sigma_new, Vt)
        
        # 2. Actualizar S (Raíces) usando Soft Thresholding
        # S = S_epsilon(M - L + Y/mu)
        temp_S = M - L_new + (1/mu) * Y
        thresh_s = lambda_param / mu
        # Soft thresholding element-wise
        S_new = np.sign(temp_S) * np.maximum(np.abs(temp_S) - thresh_s, 0)
        
        # 3. Actualizar Lagrange Multiplier y mu
        Z = M - L_new - S_new
        Y = Y + mu * Z
        mu = min(mu * rho, 1e7)
        
        # Chequeo de convergencia
        error = np.linalg.norm(Z, 'fro') / np.linalg.norm(M, 'fro')
        
        # Imprimir progreso cada 10 iteraciones
        if i % 10 == 0:
            print(f"Iteración {i}: error={error:.5f}")
            
        if error < tol:
            print(f"Convergió en iteración {i}")
            break
            
        L = L_new
        S = S_new

    return L, S

def procesar_video_rpca(input_path, output_path):
    # --- 1. Cargar Video ---
    print(f"Cargando: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Convertir a float32 y escala de grises [0, 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255.0)
        
    cap.release()
    
    video_stack = np.array(frames) # (T, H, W)
    t, height, width = video_stack.shape
    
    # Aplanar: (Time, Pixels) -> RPCA espera (Dimension, Samples) usualmente, 
    # pero aquí usaremos (Features=Pixels, Samples=Time)
    # Matriz D: Columnas son frames, Filas son píxeles
    D = video_stack.reshape(t, -1).T 
    
    print(f"Matriz de datos: {D.shape} (Pixeles x Frames)")
    print("ADVERTENCIA: Si el video es muy largo (>500 frames) esto consumirá mucha RAM.")
    
    # --- 2. Ejecutar RPCA ---
    L_matrix, S_matrix = robust_pca(D)
    
    # --- 3. Reconstrucción ---
    # S_matrix contiene las raíces. Puede tener valores negativos por el ajuste matemático.
    # Tomamos el valor absoluto para ver la "energía" de la raíz.
    S_video = np.abs(S_matrix).T.reshape(t, height, width)
    
    # Normalizar para video (0-255)
    print("Guardando resultado...")
    S_video = np.clip(S_video * 255, 0, 255).astype(np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)
    
    for frame in S_video:
        out.write(frame)
    out.release()
    print(f"Listo: {output_path}")

# --- Ejecutar ---
if __name__ == "__main__":
    # Ajusta tu ruta aquí
    video_in = "data/videos/001.AVI"
    video_out = "resultado_rpca.mp4"
    
    if os.path.exists(video_in):
        procesar_video_rpca(video_in, video_out)
    else:
        print("Edita la variable video_in con la ruta correcta")