import os
import json
import sys
from openai import OpenAI

# Inicializar cliente de NVIDIA NIM con la URL de la API v1 de integración
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# Archivos del bot de predicciones que Nemotron mantendrá, estructurará y mejorará
ARCHIVOS_OBJETIVO = [
    "requirements.txt",
    "src/scraper_espn.py",
    "src/odds_client.py",
    "src/predictor_agent.py",
    "src/main.py"
]

def leer_todo_el_repositorio():
    contexto_repo = {}
    for ruta in ARCHIVOS_OBJETIVO:
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                contexto_repo[ruta] = f.read()
        else:
            # Si el archivo no existe, le pasamos una directiva para que lo cree
            contexto_repo[ruta] = f"# Implementar aquí la lógica completa para {ruta}"
    return contexto_repo

def main():
    if not os.environ.get("NVIDIA_API_KEY"):
        print("❌ Error: La variable de entorno NVIDIA_API_KEY no está configurada.")
        sys.exit(1)

    print("📦 Leyendo el estado actual del repositorio...")
    repo_actual = leer_todo_el_repositorio()
    
    prompt_sistema = (
        "Eres Nemotron-3 Ultra (550B), un Ingeniero de Software Principal de nivel mundial.\n"
        "Tu tarea es recibir el mapa de archivos de un bot de predicciones deportivas de valor (que consume ESPN y The Odds API) "
        "y reescribir TODOS los archivos especificados para optimizarlos al máximo nivel de producción.\n\n"
        "REQUISITOS TÉCNICOS DE LOS ARCHIVOS:\n"
        "1. `requirements.txt`: Debe incluir requests, beautifulsoup4, pandas, openai y lxml.\n"
        "2. `src/scraper_espn.py`: Scraping robusto usando BeautifulSoup y Headers (User-Agent) simulación para extraer estadísticas e historial de equipos de fútbol desde ESPN.\n"
        "3. `src/odds_client.py`: Cliente funcional que consuma la API de cuotas '/v4/sports/soccer/odds' de The Odds API usando la variable de entorno ODDS_API_KEY.\n"
        "4. `src/predictor_agent.py`: Motor matemático que cruza las estadísticas y calcula cuotas teóricas (Poisson/frecuencia). Aplica la fórmula de valor: (Cuota Casa * Probabilidad Agente) - 1 > 0.\n"
        "5. `src/main.py`: Ejecutor principal que orquesta los módulos y genera un archivo markdown llamado 'predictions_report.md' con los resultados.\n\n"
        "REGLA ESTRICTA DE SALIDA:\n"
        "Debes responder EXCLUSIVAMENTE con un objeto JSON válido que contenga el mapa de los archivos. "
        "No incluyas texto explicativo, saludos, ni bloques de código formateados con triple comilla (```json). Solo el JSON puro.\n\n"
        "Estructura exacta del JSON esperado:\n"
        "{\n"
        '  "requirements.txt": "contenido...",\n'
        '  "src/scraper_espn.py": "contenido...",\n'
        '  "src/odds_client.py": "contenido...",\n'
        '  "src/predictor_agent.py": "contenido...",\n'
        '  "src/main.py": "contenido..."\n'
        "}"
    )

    print("🧠 Enviando repositorio completo a Nemotron-3 Ultra 550B para refactorización profunda...")
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Aquí está mi repositorio actual en formato JSON:\n\n{json.dumps(repo_actual, indent=2)}"}
            ],
            temperature=0.1,  # Minimiza alucinaciones matemáticas y de sintaxis
            max_tokens=4096
        )
        
        respuesta_cruda = completion.choices[0].message.content.strip()
        
        # Limpieza de seguridad en caso de que el modelo use delimitadores de markdown involuntariamente
        if respuesta_cruda.startswith("```json"):
            respuesta_cruda = respuesta_cruda.split("```json")[1].split("```")[0].strip()
        elif respuesta_cruda.startswith("```"):
            respuesta_cruda = respuesta_cruda.split("```")[1].split("```")[0].strip()

        archivos_mejorados = json.loads(respuesta_cruda)
        
        print("💾 Nemotron ha generado el código optimizado de forma exitosa. Escribiendo archivos locales...")
        for ruta, contenido in archivos_mejorados.items():
            # Crear directorios src/ si no existen previamente
            directorio = os.path.dirname(ruta)
            if directorio:
                os.makedirs(directorio, exist_ok=True)
                
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)
            print(f"✅ {ruta} actualizado y optimizado.")
            
        print("🚀 Refactorización completa realizada por Nemotron con éxito.")

    except json.JSONDecodeError as je:
        print(f"❌ Error al decodificar el JSON de Nemotron. Respuesta recibida:\n{respuesta_cruda}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error crítico durante el escaneo y actualización: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
