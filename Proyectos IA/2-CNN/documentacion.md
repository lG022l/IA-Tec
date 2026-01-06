Documentación CNN
=================

1\. Introducción y Objetivos
----------------------------

El presente proyecto tiene como objetivo el desarrollo de un sistema integral de **Visión Artificial** (Computer Vision) capaz de identificar y clasificar objetos en imágenes de manera automática. El sistema abarca el ciclo de vida completo de una aplicación de _Machine Learning_: desde la recolección automatizada de datos (dataset), pasando por el preprocesamiento y entrenamiento de una Red Neuronal Convolucional (CNN), hasta la implementación de una interfaz gráfica de usuario (GUI) para pruebas en tiempo real.

El núcleo del sistema se basa en librerías de alto rendimiento como **TensorFlow/Keras** para el modelado profundo, **Scikit-learn** para la gestión de datos y **Tkinter** para la interacción con el usuario final.

2\. Arquitectura del Sistema
----------------------------

El proyecto se divide en tres módulos funcionales independientes que interactúan a través del sistema de archivos:

1.  **Módulo de Adquisición de Datos (download.py)**: Crawler automatizado para descargar imágenes de internet.
    
2.  **Módulo de Entrenamiento (entrenamiento.ipynb)**: Script de procesamiento, definición de la red neuronal y entrenamiento.
    
3.  **Módulo de Inferencia (test.py)**: Aplicación de escritorio para probar el modelo con imágenes nuevas.
    

3\. Descripción Detallada de los Módulos
----------------------------------------

### 3.1. Módulo de Adquisición de Datos (download.py)

Este script automatiza la creación del dataset, eliminando la necesidad de buscar y guardar imágenes manualmente.

*   **Librería Principal**: icrawler (específicamente BingImageCrawler).
    
*   **Funcionamiento**:
    
    *   Utiliza un diccionario (mapa\_busqueda) donde las claves son los nombres de las carpetas (clases) y los valores son listas de términos de búsqueda ("keywords").
        
    *   **Estrategia de Variación**: Para obtener un dataset robusto y evitar imágenes repetidas, se utilizan múltiples variaciones de búsqueda para una misma clase (ej. para 'hormigas' se busca: "ant insect", "ant macro", "ant close-up").
        
    *   **Almacenamiento**: Crea automáticamente la estructura de directorios necesaria en ./dataset/{nombre\_clase}.
        
*   **Configuración**:
    
    *   IMAGENES\_POR\_KEYWORD: Define cuántas imágenes se intentarán descargar por cada variación de palabra clave.
        

### 3.2. Módulo de Entrenamiento (entrenamiento.ipynb)

Es el núcleo lógico del proyecto. Realiza la carga de imágenes, normalización, aumento de datos y entrenamiento de la red.

#### 3.2.1. Preprocesamiento de Datos

*   **Redimensionado**: Todas las imágenes se redimensionan a **64x64 píxeles** con 3 canales de color (RGB). Esto estandariza la entrada (Input Shape) para la red neuronal.
    
*   **Normalización**: Los valores de los píxeles (0-255) se dividen entre 255.0 para obtener valores flotantes entre 0 y 1, lo que facilita la convergencia del algoritmo de optimización.
    
*   **Codificación de Etiquetas**: Se utiliza _One-Hot Encoding_ (to\_categorical) para transformar las etiquetas de texto en vectores binarios.
    

#### 3.2.2. Data Augmentation (Aumento de Datos)

Para combatir el _overfitting_ (sobreajuste), se implementa ImageDataGenerator. Esto genera nuevas imágenes sintéticas en tiempo real durante el entrenamiento aplicando:

*   Rotaciones aleatorias (20 grados).
    
*   Desplazamientos horizontales y verticales.
    
*   Zoom y deformaciones (Shear).
    
*   Volteo horizontal (Horizontal Flip).
    

#### 3.2.3. Arquitectura de la Red Neuronal (CNN)

El modelo es de tipo **Secuencial** y consta de tres bloques convolucionales principales, diseñados para extraer características de menor a mayor complejidad:

1.  **Bloque 1 (Extracción Básica)**:
    
    *   Conv2D (32 filtros): Detecta bordes y texturas simples.
        
    *   BatchNormalization: Estabiliza el aprendizaje normalizando las activaciones internas.
        
    *   LeakyReLU: Función de activación que permite un pequeño gradiente negativo (evita el problema de "neuronas muertas").
        
    *   MaxPooling2D: Reduce la dimensionalidad espacial (de 64x64 a 32x32).
        
    *   Dropout (0.25): Apaga aleatoriamente el 25% de las neuronas para forzar a la red a aprender caminos redundantes (robustez).
        
2.  **Bloque 2 (Extracción Intermedia)**:
    
    *   Similar al anterior pero con **64 filtros**.
        
3.  **Bloque 3 (Extracción Compleja)**:
    
    *   Aumenta a **128 filtros** para capturar patrones complejos del objeto.
        
    *   Dropout aumenta a 0.3 para mayor regularización.
        
4.  **Clasificador (Top Model)**:
    
    *   Flatten: Aplana los mapas de características 2D a un vector 1D.
        
    *   Dense (128 neuronas): Capa totalmente conectada para razonamiento.
        
    *   Dense (Salida): Capa final con activación **Softmax**, que devuelve la probabilidad de pertenencia a cada clase.
        

#### 3.2.4. Callbacks (Optimización del Entrenamiento)

El entrenamiento incluye mecanismos de control automático:

*   **EarlyStopping**: Detiene el entrenamiento si la pérdida en validación (val\_loss) no mejora en 10 épocas.
    
*   **ModelCheckpoint**: Guarda automáticamente el archivo mejor\_modelo.h5 solo cuando se detecta una mejora en la validación.
    
*   **ReduceLROnPlateau**: Reduce la tasa de aprendizaje si el modelo se estanca, permitiendo un ajuste más fino ("fine-tuning").
    

### 3.3. Módulo de Inferencia / Interfaz Gráfica (test.py)

Provee una herramienta visual para validar el modelo fuera del entorno de programación.

*   **Tecnología**: Python Tkinter.
    
*   **Flujo de Trabajo**:
    
    1.  Carga el modelo entrenado (mejor\_modelo.h5) y el diccionario de clases (clases.npy).
        
    2.  Permite al usuario seleccionar una imagen desde su disco duro.
        
    3.  Preprocesa la imagen (mismo redimensionado y normalización que en el entrenamiento).
        
    4.  Muestra la predicción ganadora y un desglose de confianza.
        
*   **Visualización de Resultados**:
    
    *   Muestra la etiqueta ganadora en color verde.
        
    *   Indica el nivel de confianza mediante un código de colores (Verde > 80%, Naranja > 50%, Rojo < 50%).
        
    *   Desglosa las probabilidades de todas las clases posibles ordenadas de mayor a menor.
        

4\. Requisitos del Sistema
--------------------------

Para ejecutar este proyecto, se requiere el siguiente entorno de software:

*   **Lenguaje**: Python 3.8 o superior (Probado en Python 3.11).
    
*   **Librerías Clave**:
    
    *   tensorflow: Framework de Deep Learning.
        
    *   numpy: Cálculo numérico.
        
    *   scikit-learn: Utilidades de Machine Learning.
        
    *   scikit-image: Procesamiento de imágenes.
        
    *   matplotlib: Gráficas.
        
    *   pillow (PIL): Manejo de imágenes en la GUI.
        
    *   icrawler: Descarga de imágenes.
        

5\. Manual de Uso
-----------------

### Paso 1: Generación del Dataset

1.  Abra el archivo download.py.
    
2.  Pythonmapa\_busqueda = { 'gato': \['cat animal', 'gato domestico'\], 'perro': \['dog animal', 'perro raza'\]}
    
3.  Ejecute el script. Las imágenes se guardarán en la carpeta ./dataset.
    

### Paso 2: Entrenamiento del Modelo

1.  Abra y ejecute el notebook entrenamiento.ipynb (puede usar Jupyter Notebook, JupyterLab o VS Code).
    
2.  Ejecute todas las celdas secuencialmente.
    
3.  Al finalizar, verifique las gráficas de _Accuracy_ y _Loss_ para asegurar que no hubo sobreajuste.
    
4.  Se generarán dos archivos críticos: mejor\_modelo.h5 (los pesos neuronales) y clases.npy (los nombres de las categorías).
    

### Paso 3: Pruebas y Predicción

1.  Asegúrese de que mejor\_modelo.h5, clases.npy y test.py estén en el mismo directorio.
    
2.  Bashpython test.py
    
3.  En la ventana emergente, haga clic en **"ANALIZAR IMAGEN"**, seleccione un archivo y observe los resultados.
    

6\. Conclusiones y Trabajo Futuro
---------------------------------

El sistema implementado demuestra la eficacia de las Redes Neuronales Convolucionales para tareas de clasificación de imágenes. La inclusión de técnicas modernas como _Batch Normalization_, _Data Augmentation_ y _Callbacks_ inteligentes asegura que el modelo sea capaz de generalizar correctamente ante datos nuevos, evitando la simple memorización de las imágenes de entrenamiento.

**Posibles mejoras futuras:**

*   Implementar _Transfer Learning_ (usando MobileNet o VGG16) para mejorar la precisión con menos datos.
    
*   Añadir una matriz de confusión en el notebook para analizar en detalle qué clases confunde el modelo.
    
*   Migrar la interfaz gráfica a una aplicación web usando Flask o Streamlit.