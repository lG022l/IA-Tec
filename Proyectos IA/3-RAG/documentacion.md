Sistema de Generación Aumentada por Recuperación (RAG) para el Análisis de la Identidad Digital
=========================================================================================================================


1\. Introducción
---------------------------------------

### 1.1. Visión General

El presente proyecto de investigación técnica tiene como objetivo principal el desarrollo, implementación y validación de una arquitectura de Inteligencia Artificial basada en Generación Aumentada por Recuperación (RAG). El propósito funcional del sistema es actuar como un motor de inferencia capaz de analizar fenómenos sociológicos complejos —específicamente la crisis de identidad, la ansiedad y el vacío existencial en la Generación Z— contrastando evidencia empírica extraída de redes sociales con marcos teóricos filosóficos formales.

Para garantizar la robustez de los resultados y evaluar diferentes paradigmas de implementación, el proyecto se estructuró en dos vertientes arquitectónicas complementarias que operan sobre la misma base de conocimiento:

1.  **Arquitectura Programática (Ad-Hoc):** Un pipeline de procesamiento de datos construido mediante scripts de Python, utilizando las librerías LangChain y ChromaDB.
    
2.  **Arquitectura Orquestada (Low-Code):** Una implementación gestionada a través de la interfaz AnythingLLM, integrando modelos especializados como DeepSeek y motores de vectorización Nomic.
    

### 1.2. Objetivos Técnicos

El sistema busca resolver el problema de la "alucinación" en los Modelos de Lenguaje Grande (LLM) al restringir sus respuestas a un contexto cerrado y curado. Los objetivos específicos incluyen:

*   Automatizar la hermenéutica digital mediante la ingestión de fuentes heterogéneas (PDFs académicos y datasets sociales).
    
*   Comparar el rendimiento de inferencia entre modelos generalistas (Gemma 3) y modelos de razonamiento (DeepSeek-R1).
    
*   Validar los mecanismos de _Grounding_ (anclaje a la realidad) para asegurar que la IA reconozca los límites de su propio conocimiento.
    

2\. Marco Teórico y Base de Conocimiento
----------------------------------------

El núcleo del sistema RAG reside en su base de datos vectorial. A diferencia de un chatbot estándar, este sistema consulta una "memoria externa" antes de generar respuestas. Esta memoria se construyó a partir de dos categorías documentales que alimentan ambas implementaciones del sistema.

### 2.1. Corpus Filosófico (Marco Teórico)

Se seleccionaron y procesaron digitalmente textos clave de la filosofía contemporánea y la ética de la tecnología. Estos documentos actúan como el "lente" a través del cual la IA debe interpretar la realidad. Los archivos principales incluyen:

*   **Foucault y el Biopoder:** Documentación centrada en la vigilancia y las sociedades disciplinarias, utilizada para interpretar los algoritmos de recomendación como panópticos digitales.
    
*   **Habermas y la Esfera Pública:** Textos que analizan la erosión del debate racional, fundamentales para entender las cámaras de eco en redes sociales.
    
*   **Heidegger y la Técnica:** Ensayos sobre el "Desocultamiento" y la estructura de emplazamiento (_Gestell_), aplicados a la cosificación del usuario como dato.
    
*   **Dilemas Éticos en IA:** Un documento técnico (dilemaseticosia.pdf) que aborda la responsabilidad de los diseñadores de algoritmos y la manipulación autónoma.
    

### 2.2. Corpus Social (Evidencia Empírica)

Para contrastar la teoría con la realidad, se generó un dataset dinámico proveniente de la minería de datos en plataformas digitales.

*   **Comentarios de YouTube:** Extracción de opiniones reales sobre videos relacionados con "soledad", "adicción al móvil" y "dopamina", capturando la voz auténtica de la Generación Z.
    
*   **Dataset Sintético Aumentado:** Un archivo CSV generado mediante técnicas de aumento de datos, diseñado para ampliar el espectro de análisis de sentimiento y proporcionar volumen estadístico al modelo.
    

3\. Arquitectura A: Implementación Programática (Python/LangChain)
------------------------------------------------------------------

Esta primera fase del proyecto se centró en el control granular del flujo de datos mediante código, permitiendo una personalización total de las estrategias de fragmentación (_chunking_) y recuperación.

### 3.1. Módulos de Adquisición y ETL

El proceso de Extracción, Transformación y Carga (ETL) se gestionó mediante scripts dedicados:

*   **Módulo filosofia.py:** Este script implementa un crawler web que accede a fuentes académicas predefinidas. Una innovación clave en este módulo es el uso de un "Meta-Resumen" mediante IA; en lugar de guardar texto crudo, el script invoca a un modelo local para sintetizar y estructurar los conceptos filosóficos antes de guardarlos, optimizando la densidad de información.
    
*   **Módulo recollection.py:** Utilizando librerías de scraping no oficial de YouTube, este componente descarga miles de comentarios, aplica filtros de limpieza (eliminando spam y textos cortos) y normaliza el formato para su posterior ingesta.
    
*   **Módulo extraccion.ipynb:** Un cuaderno de Jupyter encargado de la manipulación de datos estructurados. Aquí se fusionan los datos reales con datos sintéticos para equilibrar las clases de sentimiento (positivo/negativo/neutral) y se exporta el corpus final.
    

### 3.2. Motor de Indexación (index.py)

La transformación de texto a vectores matemáticos es el corazón del sistema.

*   **Carga:** Se utiliza TextLoader para ingerir tanto los archivos de texto filosófico como los registros sociales.
    
*   **Fragmentación:** Se emplea RecursiveCharacterTextSplitter con un tamaño de fragmento de 600 caracteres y un solapamiento (_overlap_) de 100 caracteres. Esta configuración asegura que las ideas complejas no se corten abruptamente, manteniendo el contexto semántico necesario para el análisis filosófico.
    
*   **Vectorización:** Se implementó el modelo de embeddings sentence-transformers/all-MiniLM-L6-v2 a través de la librería HuggingFace. Este modelo convierte cada fragmento de texto en un vector denso de 384 dimensiones.
    
*   **Persistencia:** Los vectores resultantes se almacenan en una instancia local de ChromaDB (SQLite), permitiendo consultas rápidas sin necesidad de recalcular los vectores en cada ejecución.
    

### 3.3. Orquestación de Inferencia (rag.py)

El script final conecta la base de datos con el modelo de lenguaje.

*   **Recuperador (Retriever):** Se configura ChromaDB para recuperar los 5 documentos más similares (k=5) a la consulta del usuario.
    
*   **Modelo Generativo:** Se utiliza ChatOllama invocando al modelo gemma3.
    
*   **Ingeniería de Prompts:** Se diseñó una plantilla de sistema estricta ("Persona") que instruye al modelo a actuar como un investigador social experto en autores específicos (Han, Foucault, Bauman), forzando un tono analítico y académico en las respuestas.
    
*   **Ciclo de Análisis:** El script itera sobre una lista de 20 preguntas de investigación predefinidas y genera un informe automático en formato Markdown (PREG.md), citando implícitamente el contexto recuperado.
    

4\. Arquitectura B: Implementación Orquestada (AnythingLLM)
-----------------------------------------------------------

Como complemento y validación de la arquitectura programática, se desarrolló una segunda versión del sistema utilizando AnythingLLM. Esta implementación se enfoca en la usabilidad, la gestión visual del conocimiento y la integración de modelos de razonamiento avanzado.

### 4.1. Configuración del Orquestador

Se utilizó AnythingLLM como la interfaz de gestión y orquestación del sistema RAG. Esta plataforma permite una abstracción de la complejidad del código, facilitando la administración de documentos y modelos. La configuración técnica se realizó vinculando la aplicación con el servidor local de Ollama mediante los siguientes parámetros:

*   **Proveedor de LLM (Inferencia):** Se configuró el endpoint local para utilizar deepseek-r1:1.5b. La elección de este modelo específico responde a su capacidad optimizada de razonamiento ("Chain of Thought"), ideal para desglosar las consultas filosóficas complejas del proyecto.
    
*   **Proveedor de Vectores (Embeddings):** Se estableció Nomic Embed Text como el motor encargado de indexar la documentación. Este modelo de embeddings ofrece una mayor ventana de contexto y una mejor comprensión semántica comparada con modelos más ligeros, mejorando la precisión en la recuperación de documentos académicos densos.
    

### 4.2. Ingesta y Vectorización en Workspace

Para dotar a la IA de contexto específico, se creó un "Workspace" (Espacio de Trabajo) dedicado dentro de la aplicación. El proceso de ingesta de datos (Data Ingestion) fue híbrido, integrando fuentes heterogéneas en una memoria unificada:

1.  **Fuentes Académicas:** Se cargaron directamente los artículos académicos en formato PDF (dilemaseticosia.pdf, habermas.pdf, foucault.pdf, etc.). El orquestador se encargó automáticamente del _parsing_ (lectura) y la limpieza de estos archivos.
    
2.  **Fuentes Sociales:** Se integró el dataset generado en la Fase A (archivo CSV/TXT), aportando el análisis de sentimiento y la realidad social actual.
    

Posteriormente, se ejecutó el proceso de Vectorización gestionado por el orquestador. Durante esta etapa, el sistema fragmentó ambos tipos de documentos y los convirtió en vectores almacenados en la base de datos vectorial interna de AnythingLLM (LancedB por defecto). Esto permite que, ante una pregunta del usuario, el sistema recupere fragmentos tanto de la teoría (PDFs) como de la opinión pública de manera transparente.

5\. Pruebas y Validación de Resultados
--------------------------------------

Para verificar la fiabilidad industrial y académica del sistema RAG, se realizaron pruebas de inferencia orientadas a evaluar dos capacidades críticas: la recuperación de información relevante y el discernimiento de los límites de la base de conocimiento (_Grounding_). Estas pruebas se documentaron exhaustivamente sobre la implementación en AnythingLLM.

### 5.1. Primer Caso de Prueba: Validación de Grounding (Anti-Alucinación)

El objetivo de esta prueba fue someter al sistema a una consulta compleja que requería cruzar información sobre terminología generacional (slang) con teoría social. El propósito era verificar si la IA "alucinaba" o inventaba información inexistente en el dataset académico.

*   **Protocolo de Prueba:** Se introdujo el prompt: _"¿Qué expresiones o términos utiliza la Gen Z para describir el vacío existencial en redes sociales?"_.
    
*   **Comportamiento del Sistema:** El modelo realizó una búsqueda semántica en la base vectorial indexada. Al analizar los documentos recuperados (dilemaseticosia.pdf, habermas.pdf, foucault.pdf), el sistema detectó que, aunque los textos hablaban de vacío existencial y teorías sociales, no contenían una lista de jerga o "slang" específico de la Generación Z.
    
*   **Respuesta Generada:** El sistema actuó correctamente. Identificó el enfoque temático de los documentos (teorías sociales y filosofía) y notificó al usuario la ausencia de terminología específica en dichos textos académicos, en lugar de inventar términos falsos.
    
*   **Análisis de Integridad:** Este resultado se considera un éxito técnico. La funcionalidad de _Context Awareness_ (Conciencia de Contexto) previno la generación de información falsa, priorizando la veracidad de la fuente (los PDFs cargados) sobre la creatividad del modelo. La pestaña de "Citas" confirmó que el modelo leyó 1 referencia de Foucault y 2 de Habermas antes de concluir que no tenía la respuesta exacta sobre el "slang", demostrando una comprensión lectora efectiva.
    

### 5.2. Segundo Caso de Prueba: Recuperación Semántica y Síntesis

A diferencia de la prueba anterior (que validaba lo que el sistema _no_ sabe), este caso evaluó la capacidad positiva para localizar conceptos específicos dispersos y sintetizarlos.

*   **Protocolo de Prueba:** Se introdujo el prompt: _"¿Qué menciones aparecen sobre libertad, control o manipulación algorítmica?"_.
    
*   **Respuesta del Sistema:** El modelo deepseek-r1:1.5b logró recuperar fragmentos de texto altamente relevantes. Específicamente, extrajo información sobre la ética de la IA, las "armas autónomas" y la responsabilidad inherente de los "diseñadores de algoritmos".
    
*   **Análisis de Citas y Recuperación:** El desglose de referencias evidenció el funcionamiento correcto de los embeddings de Nomic.
    
    *   Del documento dilemaseticosia.pdf, el sistema extrajo tres bloques de texto distintos, identificando párrafos clave sobre "objetivos inadmisibles" y "maniobras políticas".
        
    *   Del documento habermas.pdf, correlacionó la consulta con teoría sociológica sobre la estructura social.
        
*   **Interpretación Técnica:** La respuesta generada no fue una simple copia textual. El modelo realizó una abstracción cognitiva. Tomó citas crudas como "son los diseñadores de algoritmos... quienes manipulan" y las reformuló explicativamente para el usuario, demostrando capacidad de razonamiento lógico sobre los datos vectorizados. Esto confirma que la arquitectura RAG cumple con su objetivo de asistir en la investigación cualitativa.
    

6\. Resultados del Análisis Automático (PREG.md)
------------------------------------------------

Como producto final de la arquitectura programática (Fase A), se generó el archivo PREG.md. Este documento contiene el análisis exhaustivo de 20 preguntas de investigación. Los hallazgos principales generados por el sistema incluyen:

1.  **Sobre la Identidad Líquida:** El sistema vinculó exitosamente los patrones de comportamiento de la Gen Z en redes sociales (cambios rápidos de estética, uso de múltiples pronombres) con el concepto de Zygmunt Bauman, identificando una identidad "efímera y performativa".
    
2.  **Sobre la Sociedad del Rendimiento:** El análisis detectó términos como "hustle culture" y "obsesión por la productividad" en los comentarios sociales y los conectó directamente con la teoría de Byung-Chul Han, diagnosticando una autoexplotación voluntaria.
    
3.  **Sobre el Panóptico Digital:** El sistema interpretó la aceptación de los algoritmos de recomendación no como una simple preferencia de consumo, sino como una sumisión al biopoder descrito por Foucault, donde los usuarios aceptan ser vigilados a cambio de conectividad.
    

7\. Guía de Despliegue y Requisitos del Sistema
-----------------------------------------------

Para replicar este entorno de investigación, se deben seguir los siguientes procedimientos técnicos unificados.

### 7.1. Requisitos de Software y Hardware

*   **Entorno de Ejecución:** Python 3.10 o superior.
    
*   **Servidor de Modelos:** Ollama (servicio en segundo plano) con los modelos gemma3 y deepseek-r1:1.5b descargados (ollama pull ...).
    
*   **Orquestador:** AnythingLLM (Versión Desktop) instalado.
    
*   **Recursos:** Mínimo 16GB de RAM recomendado para la ejecución fluida de la vectorización y la inferencia concurrente.
    

### 7.2. Ejecución de la Fase Programática

1.  Instalar dependencias: pip install langchain langchain-community chromadb sentence-transformers youtube-comment-downloader.
    
2.  Ejecutar scripts de recolección: python recollection.py y python filosofia.py para poblar la carpeta datos/.
    
3.  Generar la base de datos vectorial: python index.py. Esto creará el directorio chroma\_db.
    
4.  Iniciar el análisis: python rag.py. El sistema generará el reporte en tiempo real.
    

### 7.3. Configuración de la Fase Orquestada

1.  Abrir AnythingLLM y navegar a Settings > AI Providers.
    
2.  Seleccionar **Ollama** como proveedor de Chat y **Nomic** (o Ollama) como proveedor de Embeddings.
    
3.  Crear un nuevo Workspace llamado "Investigación Gen Z".
    
4.  Subir los archivos PDF (foucault.pdf, habermas.pdf, etc.) y el CSV social mediante la interfaz de "Upload".
    
5.  Pulsar "Move to Workspace" y luego "Save and Embed" para iniciar la vectorización gráfica.
    
6.  Utilizar la interfaz de chat para realizar consultas de validación y visualizar las citas en el panel lateral derecho.
    

8\. Conclusiones
---------------------------------

El proyecto ha demostrado con éxito la viabilidad de utilizar arquitecturas RAG para la investigación sociológica avanzada. La combinación de una implementación programática (Python) para el procesamiento masivo de datos y una implementación orquestada (AnythingLLM) para la exploración interactiva y la validación, resultó ser una estrategia óptima.

**Hallazgos Técnicos Clave:**

*   El modelo **DeepSeek-R1** mostró capacidades superiores de síntesis y razonamiento lógico comparado con modelos más generalistas, especialmente al tratar con textos densos de filosofía.
    
*   La arquitectura RAG eliminó eficazmente las alucinaciones; cuando el sistema carecía de datos sobre jerga específica, lo admitió en lugar de confabular, un rasgo crítico para la integridad académica.
    
*   La vectorización híbrida (Teoría + Práctica) permitió descubrir correlaciones no evidentes entre el comportamiento digital moderno y las teorías filosóficas del siglo XX.
    

9\. Referencias Bibliográficas
------------------------------

*   Cárdenas Arenas, J. C. (2005). Filosofía de la tecnología en Martin Heidegger. _Praxis Filosófica_, (21), 97-110.
    
*   González Arencibia, M., & Martínez Cardero, D. (2020). Dilemas éticos en el escenario de la inteligencia artificial. _Economía y Sociedad_, _25_(57), 1-17.
    
*   Jaramillo Marín, J. (2010). El espacio de lo político en Habermas: Alcances y límites de las nociones de esfera pública y política deliberativa. _Jurídicas_, _7_(1), 55-73.
    
*   Toscano López, D. G. (2008). El bio-poder en Michel Foucault. _Universitas Philosophica_, _25_(51), 39-57.