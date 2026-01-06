import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PREGUNTAS = [
    "¿Qué expresiones o términos utiliza la Gen Z para describir el vacío existencial en redes sociales?",
    "¿Cómo influyen los algoritmos de recomendación en la construcción de su identidad?",
    "¿Qué emociones aparecen con mayor frecuencia cuando se habla de burnout o presión digital?",
    "¿La Gen Z percibe la autonomía como algo propio o como algo condicionado por la tecnología?",
    "¿Qué diferencias hay entre discursos auténticos vs discursos performativos en plataformas como TikTok?",
    "¿Existen patrones de lenguaje que indiquen crisis de sentido o desorientación vital?",
    "¿Cómo se refleja la idea de 'identidad líquida' en los datos recuperados?",
    "¿Qué menciones aparecen sobre libertad, control o manipulación algorítmica?",
    "¿Se observan señales de que los algoritmos crean deseos o hábitos?",
    "¿Qué temas o preocupaciones predominan en la conversación digital sobre propósito de vida?",
    "¿Hay evidencia de rechazo a los metarrelatos o valores tradicionales?",
    "¿Cómo aparece la figura del 'yo digital' en los textos analizados?",
    "¿Qué ejemplos concretos muestran pérdida del pensamiento crítico por efecto de la burbuja de filtros?",
    "¿Existen contrastes entre la visión que la Gen Z tiene de sí misma y lo que los datos sugieren?",
    "¿Qué rol juega la hiperconectividad en la ansiedad o depresión mencionada?",
    "¿Se observan patrones que apoyen las ideas de Byung-Chul Han sobre rendimiento y autoexplotación?",
    "¿Cómo interpretaría Foucault el régimen de vigilancia algorítmica detectado?",
    "¿Qué evidencias hay de que la tecnología 'desoculta' y transforma la vida según Heidegger?",
    "¿El espacio público digital está debilitado como afirma Habermas? ¿Qué muestran los datos?",
    "¿Cuáles son los principales miedos, frustraciones y esperanzas de la Gen Z frente al futuro?"
]

def analizar_todo():
    print("Cargando base de datos...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(model="gemma3", temperature=0.3)

    template = """Eres un investigador social experto en filosofía digital (Han, Foucault, Bauman).
    Usa el siguiente contexto real extraído de redes sociales para responder la pregunta.
    
    Contexto: {context}
    
    Pregunta: {question}
    
    Instrucciones:
    - Responde de forma analítica.
    - Cita ejemplos del texto si existen.
    - Si el contexto no tiene información suficiente, dilo claramente.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Ejecutar el bucle de preguntas
    print(f"Iniciando análisis de {len(PREGUNTAS)} preguntas...\n")
    
    with open("PREG.md", "w", encoding="utf-8") as f:
        f.write("# Informe de Análisis: Gen Z y Algoritmos\n\n")
        
        for i, p in enumerate(PREGUNTAS, 1):
            print(f"[{i}/{len(PREGUNTAS)}] Analizando: {p}...")
            respuesta = chain.invoke(p)
            
            # Escribir en el archivo
            f.write(f"## {i}. {p}\n\n")
            f.write(respuesta + "\n\n")
            f.write("---\n")

    print("\n¡Análisis completado! Revisa el archivo 'PREG.md'.")

if __name__ == "__main__":
    analizar_todo()