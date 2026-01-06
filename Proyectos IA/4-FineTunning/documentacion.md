Documentación: Fine-Tuning
====================================================================

1\. Resumen 
---------------------

Este proyecto implementa un sistema de Inteligencia Artificial Generativa especializado en la enseñanza de algoritmos. Se utiliza una técnica de Fine-Tuning sobre el modelo de lenguaje **Microsoft Phi-2**.

Para lograr el entrenamiento en hardware de consumo (específicamente una GPU NVIDIA RTX 3050 con memoria limitada), se emplearon técnicas de optimización avanzadas como **QLoRA** (Quantized Low-Rank Adaptation), reduciendo la precisión del modelo base a 4-bit y entrenando únicamente un conjunto reducido de adaptadores. El resultado es un modelo ligero, capaz de responder preguntas técnicas con un formato pedagógico predefinido.

2\. Introducción y Objetivos
----------------------------

### 2.1 Contexto

Los Modelos Grandes de Lenguaje (LLMs) generalistas suelen tener un conocimiento amplio pero superficial. En el ámbito educativo, específicamente en ciencias de la computación, se requiere precisión técnica y un estilo de explicación didáctico. El entrenamiento completo de un LLM es costoso e inviable para estudiantes; sin embargo, el _Fine-Tuning_ permite adaptar modelos pre-entrenados a tareas específicas.

### 2.2 Objetivos

1.  **Objetivo General:** Desarrollar un asistente virtual experto en algoritmos que pueda ejecutarse y entrenarse en hardware local de gama media/baja.
    
2.  **Objetivos Específicos:**
    
    *   Curar un dataset de pares instrucción-respuesta sobre temas de computación (grafos, árboles, ordenamiento).
        
    *   Implementar un pipeline de entrenamiento utilizando la biblioteca transformers y peft.
        
    *   Optimizar el consumo de VRAM utilizando cuantización de 4 bits (bitsandbytes).
        
    *   Desarrollar un script de inferencia para interactuar con el modelo final.
        

3\. Marco Teórico
-----------------

Para comprender la implementación, es necesario definir las tecnologías clave utilizadas en trainv2.py.

### 3.1 Microsoft Phi-2

Phi-2 es un "Small Language Model" (SLM) de Microsoft con 2.7 billones de parámetros. A diferencia de modelos como Llama-2 (7B) o GPT-4, Phi-2 fue entrenado con "libros de texto" sintéticos de alta calidad, lo que le otorga una capacidad de razonamiento lógico superior a su tamaño. Su elección para este proyecto se justifica por su equilibrio entre rendimiento y requisitos de memoria (cabe en ~6GB de VRAM en 4-bit).

### 3.2 Fine-Tuning Supervisado (SFT)

El SFT implica entrenar el modelo con un conjunto de datos etiquetado donde se le muestra explícitamente la entrada (pregunta del estudiante) y la salida deseada (explicación del tutor). Esto ajusta los pesos probabilísticos del modelo para favorecer este tipo de respuestas.

### 3.3 PEFT y LoRA (Low-Rank Adaptation)

Entrenar los 2.7 billones de parámetros requeriría más de 40GB de VRAM. **PEFT** (Parameter-Efficient Fine-Tuning) es una librería que habilita métodos como **LoRA**.

*   **Concepto:** En lugar de modificar todos los pesos $W$ de la red neuronal, LoRA inyecta matrices pequeñas de rango bajo $A$ y $B$ en las capas de atención.
    
*   **Matemática:** $W' = W + \\Delta W = W + BA$. Durante el entrenamiento, $W$ se congela y solo se actualizan $A$ y $B$, reduciendo los parámetros entrenables en un 99%.
    

### 3.4 QLoRA (Quantized LoRA)

Es la técnica que hace posible este proyecto en una RTX 3050.

1.  **Cuantización NF4:** El modelo base se carga en 4 bits (Normal Float 4), reduciendo su tamaño en memoria en un 75%.
    
2.  **Double Quantization:** Cuantiza también las constantes de cuantización para ahorrar más memoria.
    
3.  **Paged Optimizers:** Usa la memoria RAM del sistema si la VRAM de la GPU se llena.
    

4\. Dataset y Preprocesamiento
------------------------------

El archivo tutor\_dataset.jsonl es el combustible del modelo.

### 4.1 Estructura del Dataset

El formato utilizado es **JSONL** (JSON Lines), donde cada línea es un objeto independiente. Se sigue el esquema estándar de chat de OpenAI/HuggingFace.

### 4.2 Análisis del Contenido

Basado en los archivos recuperados, el dataset contiene:

*   **System Prompt:** Define la personalidad ("Tutor experto", "Explicas de forma clara").
    
*   **Instrucciones (User):** Preguntas sobre estructuras de datos (Colas, Pilas, Árboles, Grafos, Big O).
    
*   **Respuestas (Assistant):** Explicaciones estructuradas con emojis, viñetas y ejemplos de código o analogías.
    

### 4.3 Formateo (Chat Templates)

En trainv2.py, la función format\_chat\_template transforma el JSON estructurado en una sola cadena de texto que el modelo puede leer secuencialmente. Se utilizan tokens especiales para delimitar turnos:

*   **Entrada:** Objeto JSON.
    
*   <|system|>\\nEres un tutor... <|user|>\\n¿Qué es un grafo? <|assistant|>\\nUn grafo es... <|end|>\\n
    

Esto enseña al modelo cuándo debe dejar de hablar (<|end|>).

5\. Implementación Técnica (Análisis de Código)
-----------------------------------------------

Esta sección desglosa los scripts desarrollados.

### 5.1 Entrenamiento (trainv2.py)

El script de entrenamiento es el núcleo del proyecto. Se divide en fases lógicas:

#### A. Configuración de Entorno y Hardware

El script verifica la disponibilidad de CUDA (GPU NVIDIA). Es crucial para asegurar que bitsandbytes funcione correctamente.

#### B. Carga y Cuantización del Modelo (BitsAndBytes)

Aquí reside la optimización clave.

Esto permite cargar Phi-2 ocupando el mínimo espacio posible.

#### C. Tokenización

Se utiliza el AutoTokenizer de Microsoft.

*   **Padding:** Se configura padding\_side="right" para prevenir problemas con la atención en modelos causales.
    
*   **Tokens especiales:** Se asegura que el modelo reconozca el token de fin de secuencia (eos\_token).
    

#### D. Configuración de LoRA (Adaptadores)

Se atacan los módulos de proyección de _Query, Key, Value_ (q\_proj, k\_proj, v\_proj) y la capa densa. Esto permite que el modelo aprenda nuevos patrones de razonamiento (algoritmos) en lugar de solo vocabulario.

#### E. El Trainer

Se utiliza la clase Trainer de HuggingFace, que abstrae el bucle de entrenamiento (forward pass, loss calculation, backward pass, optimizer step).

*   **Gradient Checkpointing:** Se habilita explícitamente (model.gradient\_checkpointing\_enable()). Esto reduce el uso de VRAM drásticamente al no guardar todas las activaciones intermedias, recalculándolas cuando es necesario durante el _backward pass_.
    

### 5.2 Inferencia y Pruebas (try.py)

Este script permite probar el modelo una vez entrenado.

1.  **Carga del Modelo Base:** Carga nuevamente microsoft/phi-2 en memoria.
    
2.  **Inyección de Adaptadores:** Utiliza PeftModel.from\_pretrained para fusionar dinámicamente los pesos aprendidos (guardados en la carpeta tutor-algoritmos-rtx3050) con el modelo base.
    
3.  **Generación:**
    
    *   La función preguntar() envuelve la consulta del usuario en el formato <|user|>\\n....
        
    *   Usa model.generate con parámetros de decodificación (aunque no explícitos en el snippet, se asume greedy o sampling básico).
        
    *   **Gestión de Tensores:** El script mueve explícitamente los inputs al dispositivo (.to(device)), un paso crítico para evitar errores de _device mismatch_ entre CPU y GPU.
        

6\. Estructura de Archivos del Modelo (tutor-algoritmos-rtx3050)
----------------------------------------------------------------

Al finalizar el entrenamiento, se genera una carpeta que no contiene el modelo completo (que pesaría gigabytes), sino solo los cambios:

1.  **adapter\_model.safetensors**: Archivo binario que contiene los pesos de las matrices $A$ y $B$ de LoRA. Es muy ligero (probablemente <100MB).
    
2.  **adapter\_config.json**: Metadatos de la configuración LoRA (rango, alpha, módulos objetivo).
    
3.  **tokenizer.json / vocab.json**: El vocabulario necesario para convertir texto a números.
    
4.  **added\_tokens.json**: Si agregamos tokens especiales (como los de formato de chat), aparecen aquí.
    

Esta arquitectura modular permite compartir el modelo fácilmente, ya que solo se necesita enviar la carpeta de adaptadores, y el usuario final descarga el modelo base de Microsoft por su cuenta.

7\. Requisitos de Instalación
-----------------------------

Para replicar este entorno en otro equipo universitario, se requiere un entorno Python 3.10+ y las siguientes librerías:


**Nota sobre Hardware:**

*   **Mínimo:** GPU NVIDIA con 6GB VRAM (GTX 1660 Super / RTX 3050 Laptop).
    
*   **Recomendado:** RTX 3060 (12GB) o superior.
    
*   **RAM:** 16GB.
    

8\. Guía de Ejecución
---------------------

### Paso 1: Preparación de Datos

Asegúrese de que tutor\_dataset.jsonl esté en la misma carpeta que los scripts. El formato debe ser válido.

### Paso 2: Entrenamiento

_El proceso mostrará una barra de progreso. En una RTX 3050, esto puede tardar entre 1 y 4 horas dependiendo de la cantidad de "epochs" configuradas._

### Paso 3: Pruebas

Una vez finalizado, se creará la carpeta tutor-algoritmos-rtx3050. Ejecutar:

Modificar la variable pregunta dentro de try.py o implementar un input() para conversar interactivamente.

9\. Conclusiones
---------------------------------

### Conclusiones

1.  **Viabilidad:** Se demostró que es posible realizar _fine-tuning_ de modelos de lenguaje modernos en hardware accesible para estudiantes universitarios utilizando QLoRA.
    
2.  **Especialización:** El modelo resultante deja de divagar con información general y adopta el rol específico de un tutor de programación, gracias al dataset curado.
    
3.  **Eficiencia:** El uso de adaptadores LoRA permite tener múltiples "personalidades" (tutor de algoritmos, tutor de historia, experto en SQL) usando el mismo modelo base, ahorrando espacio en disco.
    

### Limitaciones

*   **Ventana de Contexto:** Phi-2 tiene una ventana limitada (2048 tokens), lo que impide analizar códigos muy extensos.
    
*   **Alucinaciones:** Como todo LLM, puede generar código sintácticamente correcto pero lógicamente erróneo si el dataset no cubre casos borde.
    
