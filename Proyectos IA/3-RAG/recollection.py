import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
from itertools import islice

# Configuración
OUTPUT_FILE = "datos/cyoutube.txt"
LIMIT_PER_VIDEO = 70

# Videos seleccionados (Temas: Adicción a redes, Soledad, Burnout Gen Z)
# Puedes añadir más URLs aquí si quieres.
# Ejemplo de lista expandida
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=luTkEGY5ixM",
    "https://www.youtube.com/watch?v=3xosRVxzjgA",
    "https://www.youtube.com/watch?v=ZMQALttUvyM",
    "https://www.youtube.com/watch?v=5qpPPM5NLNw",
    "https://www.youtube.com/watch?v=YXxBpGhlAeY",
    "https://www.youtube.com/watch?v=o2k7VjBUAtA",
    "https://www.youtube.com/watch?v=KlFCmHvWxlU",
    "https://www.youtube.com/watch?v=x7-sVvW2-5Y", 
    "https://www.youtube.com/watch?v=QA9Wfyh0mFQ", 
    "https://www.youtube.com/watch?v=ZKk5sJ6S5rY", 
    "https://www.youtube.com/watch?v=0ggK1Qz7HnI", 
    "https://www.youtube.com/watch?v=7Pq-S557XQU", 
]



def descargar_comentarios():
    downloader = YoutubeCommentDownloader()
    all_comments = []

    print("--- Iniciando descarga de comentarios ---")
    
    for url in VIDEO_URLS:
        print(f"Procesando: {url}")
        try:
            # Obtener comentarios (generador)
            comments = downloader.get_comments_from_url(url, sort_by=0) 
            
            # Tomar solo los primeros N comentarios
            for comment in islice(comments, LIMIT_PER_VIDEO):
                text = comment['text']
                # Limpieza básica rápida: eliminar saltos de línea excesivos
                text_clean = text.replace('\n', ' ').strip()
                
                if len(text_clean) > 30: # Ignorar comentarios muy cortos como "lol"
                    all_comments.append(f"FUENTE: YOUTUBE_VIDEO | COMENTARIO: {text_clean}")
                    
        except Exception as e:
            print(f"Error en video {url}: {e}")

    # Guardar en txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in all_comments:
            f.write(c + "\n")
            
    print(f"\n--- Éxito! ---")
    print(f"Se guardaron {len(all_comments)} comentarios en: {OUTPUT_FILE}")

if __name__ == "__main__":
    descargar_comentarios()