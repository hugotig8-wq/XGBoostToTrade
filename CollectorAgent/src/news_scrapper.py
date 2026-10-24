import os
import re
import time
import requests
import yfinance as yf
from datetime import datetime
from urllib.parse import urlparse

# --- CONFIGURACIÓN ---
ASSETS = ["BZ=F", "WTI", "XOM", "CVX", "IBE.MC", "NEE"]
BASE_DIR = "/home/humberto/dataset_noticias"
TEXT_DIR = os.path.join(BASE_DIR, "textos")
IMG_DIR = os.path.join(BASE_DIR, "imagenes")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def setup_directories():
    """Crea las carpetas necesarias si no existen."""
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"📁 Directorios listos en: {BASE_DIR}")

def clean_filename(text):
    """Limpia el texto para usarlo como nombre de archivo seguro."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text)[:50]

def download_image(img_url, filename):
    """Descarga la imagen desde la URL y la guarda."""
    try:
        response = requests.get(img_url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            # Detectar la extensión del archivo
            ext = ".jpg"
            if ".png" in img_url.lower(): ext = ".png"
            if ".webp" in img_url.lower(): ext = ".webp"
            
            filepath = os.path.join(IMG_DIR, f"{filename}{ext}")
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
    except Exception as e:
        print(f"   ⚠️ Error descargando imagen: {e}")
    return None

def fetch_news_for_assets():
    """Obtiene noticias de Yahoo Finance para los activos seleccionados."""
    print(f"\n🚀 Iniciando recolección de noticias para: {', '.join(ASSETS)}")
    today_str = datetime.now().strftime("%Y-%m-%d")
    total_news_saved = 0

    for ticker in ASSETS:
        print(f"\n🔍 Buscando noticias para {ticker}...")
        try:
            # yfinance devuelve un diccionario con las noticias recientes
            ticker_obj = yf.Ticker(ticker)
            news_items = ticker_obj.news
            
            if not news_items:
                print(f"   ⚠️ No se encontraron noticias recientes para {ticker}.")
                continue

            for i, news in enumerate(news_items):
                content = news.get('content', {})
                title = content.get('title', 'Sin Titular')
                summary = content.get('summary', '')
                pub_date = content.get('pubDate', '')
                provider = content.get('provider', {}).get('displayName', 'Desconocido')
                
                # Obtener URL de la imagen (si existe)
                thumbnail_url = None
                thumbnail_data = content.get('thumbnail', {})
                if thumbnail_data and thumbnail_data.get('resolutions'):
                    thumbnail_url = thumbnail_data['resolutions'][0].get('url')

                # Crear un nombre de archivo único: 2026-10-24_TICKER_Titular
                safe_title = clean_filename(title)
                base_filename = f"{today_str}_{ticker}_{safe_title}"
                txt_path = os.path.join(TEXT_DIR, f"{base_filename}.txt")

                # Comprobar si ya lo descargamos antes para no duplicar
                if os.path.exists(txt_path):
                    continue

                # 1. Guardar el texto en un .txt
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"TICKER: {ticker}\n")
                    f.write(f"TITULAR: {title}\n")
                    f.write(f"PROVEEDOR: {provider}\n")
                    f.write(f"FECHA: {pub_date}\n")
                    f.write(f"RESUMEN: {summary}\n")
                    f.write(f"URL: {news.get('link', 'N/A')}\n")
                
                print(f"   📝 Guardado texto: {title[:50]}...")

                # 2. Descargar la imagen si existe
                if thumbnail_url:
                    img_path = download_image(thumbnail_url, base_filename)
                    if img_path:
                        print(f"   🖼️ Imagen guardada: {os.path.basename(img_path)}")
                
                total_news_saved += 1
                
                # Pausa de cortesía para no saturar el servidor de Yahoo
                time.sleep(1) 
                
        except Exception as e:
            print(f"   ❌ Error procesando {ticker}: {e}")

    print(f"\n✅ Proceso completado. Se guardaron {total_news_saved} noticias nuevas.")

if __name__ == "__main__":
    setup_directories()
    fetch_news_for_assets()
