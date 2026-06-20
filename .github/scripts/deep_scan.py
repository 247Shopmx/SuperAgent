import os
import json
import sys
from openai import OpenAI

# Inicializar cliente de NVIDIA NIM utilizando la URL base de integración oficial
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# Estructura del repositorio 247Shopmx/SuperAgent que la IA debe generar/optimizar
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
            contexto_repo[ruta] = f"# Crear la estructura inicial de código para {ruta}"
    return contexto_repo

def main():
    if not os.environ.get("NVIDIA_API_KEY"):
        print("❌ Error: Falta la variable de entorno NVIDIA_API_KEY en los Secrets de GitHub.")
        sys.exit(1)

    # Utilizando el modelo verificado y disponible en tu Tier de NVIDIA
    MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-instruct"

    print(f"📦 Mapeando el repositorio actual (247Shopmx/SuperAgent)...")
    repo_actual = leer_todo_el_repositorio()
    
    prompt_sistema = (
        "Eres un Ingeniero de Software experto en Algoritmos Cuantitativos y APIs de NVIDIA.\n"
        "Tu tarea es generar el código completo de un bot de Value Betting en fútbol que haga scraping de ESPN y consuma The Odds API.\n\n"
        "REQUISITOS ARQUITECTÓNICOS MÍNIMOS:\n"
        "1. `requirements.txt`: Debe incluir requests, beautifulsoup4, pandas, openai, lxml.\n"
        "2. `src/scraper_espn.py`: Scraping usando BeautifulSoup simulando cabeceras (User-Agent) para extraer estadísticas, goles y xG de ESPN.\n"
        "3. `src/odds_client.py`: Cliente para consumir la API de cuotas '/v4/sports/soccer/odds' usando la variable de entorno ODDS_API_KEY.\n"
        "4. `src/predictor_agent.py`: Motor matemático que calcula probabilidades implícitas y determina si hay valor mediante: (Cuota Casa * Probabilidad Bot) - 1 > 0.\n"
        "5. `src/main.py`: Orquestador que ejecuta el flujo y escribe un reporte markdown llamado 'predictions_report.md'.\n\n"
        "REGLA DE FORMATO ABSOLUTA:\n"
        "Responde ÚNICAMENTE con un objeto JSON crudo y válido que contenga las rutas como llaves y el código completo de producción como valores. "
        "No incluyas texto descriptivo, explicaciones, ni bloques de código formateados con caracteres de markdown (```json). Solo JSON puro.\n\n"
        "Formato del JSON:\n"
        "{\n"
        '  "requirements.txt": "codigo...",\n'
        '  "src/scraper_espn.py": "codigo...",\n'
        '  "src/odds_client.py": "codigo...",\n'
        '  "src/predictor_agent.py": "codigo...",\n'
        '  "src/main.py": "codigo..."\n'
        "}"
    )

    print(f"🧠 Enviando solicitud a NVIDIA NIM utilizando el modelo: {MODEL_NAME}...")
    try:
        # Forzar respuesta en formato JSON utilizando el parámetro response_format si el endpoint lo soporta
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Estructura del repositorio actual:\n\n{json.dumps(repo_actual, indent=2)}"}
            ],
            temperature=0.1,
            max_tokens=3500
        )
        
        respuesta_cruda = completion.choices[0].message.content.strip()
        
        # Limpieza de marcado por si el modelo incluye marcas de bloque JSON
        if respuesta_cruda.startswith("```json"):
            respuesta_cruda = respuesta_cruda.split("```json")[1].split("```")[0].strip()
        elif respuesta_cruda.startswith("```"):
            respuesta_cruda = respuesta_cruda.split("```")[1].split("```")[0].strip()

        archivos_mejorados = json.loads(respuesta_cruda)
        
        print("💾 Inyectando código optimizado generado por la IA...")
        for ruta, contenido in archivos_mejorados.items():
            directorio = os.path.dirname(ruta)
            if directorio:
                os.makedirs(directorio, exist_ok=True)
                
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)
            print(f"✅ Archivo configurado con éxito: {ruta}")
            
        print("🚀 Proceso de actualización del repositorio completado de forma autónoma.")

    except json.JSONDecodeError:
        print("❌ Error: La respuesta de la API no contiene un JSON limpio. Respuesta obtenida:")
        print(respuesta_cruda)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error crítico en la llamada de inferencia a {MODEL_NAME}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
