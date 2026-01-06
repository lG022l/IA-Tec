import requests
from bs4 import BeautifulSoup
import ollama
import os

URLS_FILOSOFIA = [
    "https://iep.utm.edu/existent/",
    "https://reviewbooku.com/review/byung-chul-han-the-burnout-society-4981180",
    "https://iep.utm.edu/habermas/",
    "https://iep.utm.edu/heidegge/",
    "https://iep.utm.edu/foucault/",
    "https://iep.utm.edu/postmodernism/",
    "https://traversingtradition.com/2025/11/03/liquid-identities-solid-critiques-review-of-zygmunt-baumans-culture-in-a-liquid-modern-world/",
]

OUTPUT_FILE = "data/filosofia.txt"
os.makedirs("data", exist_ok=True)

def extraer_texto_web(url):
    """Descarga el texto principal de una página web."""
    print(f"Descargando: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        parrafos = soup.find_all('p')
        texto_limpio = "\n".join([p.get_text() for p in parrafos])
        
        # Reducimos un poco el límite por URL para que quepan todas en el contexto de la IA
        return texto_limpio[:4000] 
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return ""

def estructurar_con_ia(texto_completo):
    """Analiza TODO el texto acumulado y genera un resumen único."""
    print("Analizando y sintetizando todos los textos con IA (esto puede tardar un poco)...")
    
    prompt = f"""
    Actúa como un experto filósofo y analista de sistemas. Tienes abajo una recopilación de textos de varias fuentes.
    
    Tu tarea es generar UNA ÚNICA entrada consolidada para CADA UNO de los siguientes 7 conceptos (No repitas conceptos, unifica la información):

    1. Existencialismo (Sartre, Camus): vacío existencial.
    2. Posmodernidad (Lyotard): fin de los metarrelatos.
    3. Identidad líquida (Bauman).
    4. Cultura del rendimiento (Byung-Chul Han).
    5. Vigilancia y biopoder (Foucault).
    6. Desocultamiento y tecnificación del ser (Heidegger).
    7. Erosión del espacio público (Habermas).

    INSTRUCCIONES:
    - Si encuentras información contradictoria, sintetízala.
    - El formato debe ser estrictamente el siguiente para cada uno de los 7 conceptos:

    CONCEPTO: [Nombre del Concepto y Autor]
    DEFINICION: [Síntesis clara del concepto filosófico basada en los textos]
    RELACION CON GEN Z: [Cómo esto explica el comportamiento de la Gen Z, redes sociales, ansiedad o tecnología]
    ---

    TEXTOS FUENTE ACUMULADOS:
    {texto_completo}
    """

    # Nota: Asegúrate de que tu modelo gemma3 soporte suficiente contexto. 
    # Si falla, prueba con un modelo con mayor ventana de contexto o reduce el texto.
    response = ollama.chat(model='gemma3', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    texto_acumulado = ""

    # 1. FASE DE RECOLECCIÓN (Bucle solo para descargar)
    for url in URLS_FILOSOFIA:
        texto = extraer_texto_web(url)
        if texto and len(texto) > 200:
            texto_acumulado += f"\n\n--- TEXTO DE {url} ---\n{texto}"
            print("Texto agregado a memoria.")
        else:
            print("URL con contenido insuficiente.")

    # 2. FASE DE PROCESAMIENTO (Una sola llamada a la IA)
    if texto_acumulado:
        resultado_final = estructurar_con_ia(texto_acumulado)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("TITULO: Marco Filosófico Unificado\n\n")
            f.write(resultado_final)

        print("Archivo creado")
    else:
        print("No se pudo recolectar suficiente texto para procesar.")

if __name__ == "__main__":
    main()