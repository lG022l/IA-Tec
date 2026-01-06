import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread

# --- CONFIGURACIÓN ---
MODELO_PATH = 'mejor_modelo.h5'
ANCHO = 64
ALTO = 64

# 1. CARGAR MODELO
print("⏳ Cargando modelo...")
try:
    model = load_model(MODELO_PATH)
    if os.path.exists('clases.npy'):
        clases = np.load('clases.npy')
    else:
        # Asegúrate que estas sean las mismas clases con las que entrenaste
        clases = ['gato', 'hormiga', 'ladybug', 'perro', 'tortuga']
    print("✅ Modelo cargado.")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    # Definimos clases dummy por si falla la carga para que no crashee la UI al abrir
    clases = ['clase1', 'clase2', 'clase3', 'clase4', 'clase5']
    model = None

def analizar_imagen():
    if model is None:
        messagebox.showerror("Error", "El modelo no se cargó correctamente.")
        return

    ruta_imagen = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not ruta_imagen:
        return

    try:
        # --- A. MOSTRAR IMAGEN EN UI ---
        imagen_pil = Image.open(ruta_imagen)
        imagen_pil.thumbnail((300, 250), Image.Resampling.LANCZOS) 
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        lbl_imagen.config(image=imagen_tk, width=300, height=250, bg="#e0e0e0")
        lbl_imagen.image = imagen_tk 
        
        # --- B. PROCESAMIENTO Y PREDICCIÓN ---
        img = imread(ruta_imagen)
        if len(img.shape) == 3 and img.shape[2] == 4: img = img[:,:,:3] # Sin transparencia
        
        # Redimensionar a 64x64 (Lo que ve la IA)
        img_resized = resize(img, (ALTO, ANCHO), anti_aliasing=True, preserve_range=True)
        X_input = np.array([img_resized], dtype=np.float32) / 255.0
        
        # Predecir
        prediccion = model.predict(X_input)
        vector_prediccion = prediccion[0] # El array de probabilidades [0.1, 0.8, ...]
        
        # Obtener ganadora
        indice = np.argmax(vector_prediccion)
        clase_ganadora = clases[indice]
        confianza = np.max(vector_prediccion) * 100
        
        # --- C. RESULTADOS PRINCIPALES ---
        lbl_resultado.config(text=f"{clase_ganadora.upper()}", fg="#27ae60")
        
        # Color según confianza
        if confianza > 80: color_conf = "#27ae60" 
        elif confianza > 50: color_conf = "#f39c12" 
        else: color_conf = "#c0392b" 
        
        lbl_confianza.config(text=f"Ganador: {confianza:.1f}%", fg=color_conf)

        # --- D. MOSTRAR TODAS LAS CLASES (NUEVO) ---
        # 1. Crear una lista de tuplas (nombre, porcentaje)
        resultados_lista = []
        for i in range(len(clases)):
            probabilidad = vector_prediccion[i] * 100
            resultados_lista.append((clases[i], probabilidad))
        
        # 2. Ordenar de mayor a menor probabilidad
        resultados_lista.sort(key=lambda x: x[1], reverse=True)

        # 3. Construir el texto para mostrar
        texto_detalle = ""
        for clase, score in resultados_lista:
            # Formato:  Gato: 85.0%
            texto_detalle += f"{clase.capitalize()}: {score:.2f}%\n"

        # 4. Actualizar la etiqueta de detalles
        lbl_detalles.config(text=texto_detalle)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error procesando imagen: {e}")
        print(e)

# 3. DISEÑO DE VENTANA
ventana = tk.Tk()
ventana.title("Clasificador CNN")

# Aumenté un poco el alto para que quepa la lista
ventana.geometry("380x680") 
ventana.resizable(True, True)
ventana.config(bg="#f4f6f7")

# --- HEADER ---
frame_top = tk.Frame(ventana, bg="#2c3e50", height=60)
frame_top.pack(fill="x")
lbl_titulo = tk.Label(frame_top, text="Detector IA", font=("Segoe UI", 14, "bold"), bg="#2c3e50", fg="white")
lbl_titulo.pack(pady=15)

# --- ÁREA DE IMAGEN ---
frame_img = tk.Frame(ventana, bg="#bdc3c7", bd=2, relief="sunken")
frame_img.pack(pady=15)

lbl_imagen = tk.Label(frame_img, text="Sin imagen", font=("Arial", 10), bg="#e0e0e0", fg="#7f8c8d", width=40, height=15)
lbl_imagen.pack()

# --- RESULTADO GANADOR ---
lbl_resultado = tk.Label(ventana, text="...", font=("Segoe UI", 16, "bold"), bg="#f4f6f7", fg="#34495e")
lbl_resultado.pack(pady=(5, 0))

lbl_confianza = tk.Label(ventana, text="", font=("Segoe UI", 10), bg="#f4f6f7", fg="#7f8c8d")
lbl_confianza.pack(pady=(0, 10))

# --- LISTA DE DETALLES (NUEVO) ---
# Creamos un LabelFrame o simplemente un Label para listar los porcentajes
frame_detalles = tk.Frame(ventana, bg="#f4f6f7")
frame_detalles.pack(pady=5)

lbl_detalles = tk.Label(frame_detalles, text="Probabilidades por clase aparecerán aquí", 
                        font=("Consolas", 10), bg="#f4f6f7", fg="#2c3e50", justify="left")
lbl_detalles.pack()

# --- BOTÓN ---
btn_cargar = tk.Button(ventana, text="📷 ANALIZAR IMAGEN", command=analizar_imagen, 
                        font=("Segoe UI", 10, "bold"), bg="#3498db", fg="white", 
                        activebackground="#2980b9", activeforeground="white", 
                        relief="flat", padx=20, pady=8, cursor="hand2")
btn_cargar.pack(side="bottom", pady=20)

ventana.mainloop()