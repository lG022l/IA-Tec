import os
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

# Configuración
CARPETA_BASE = './dataset'
IMAGENES_POR_KEYWORD = 500  # Bajamos el target por palabra, pero usamos muchas palabras

# Mapeo: Carpeta -> Lista de variaciones de búsqueda
# Esto engaña al buscador pidiendo cosas "diferentes" para llenar la misma carpeta
mapa_busqueda = {
    'hormigas': [
        'ant insect', 'ant red', 'ant black ', 'ant macro', 'ant close-up'
    ]
}

def descargar_todo():
    for carpeta, keywords in mapa_busqueda.items():
        ruta_final = os.path.join(CARPETA_BASE, carpeta)
        if not os.path.exists(ruta_final):
            os.makedirs(ruta_final)
            
        print(f"\n🦁 Llenando carpeta: {carpeta}...")
        
        for keyword in keywords:
            print(f"   🔍 Buscando variación: '{keyword}'...")
            
            # Usamos Bing porque suele bloquear menos que Google
            crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
                storage={'root_dir': ruta_final},
                log_level='ERROR' # Para que no llene la pantalla de texto
            )
            
            crawler.crawl(
                keyword=keyword, 
                max_num=IMAGENES_POR_KEYWORD, 
                file_idx_offset='auto' # Importante: no sobreescribe las anteriores
            )

if __name__ == '__main__':
    descargar_todo()
    print("\n✅ ¡Descarga masiva terminada! Revisa las carpetas.")