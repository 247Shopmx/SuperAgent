import os
import json
import sys
from openai import OpenAI

# Inicializar cliente de NVIDIA NIM para Nemotron-3 Ultra 550B
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# Definición de la estructura limpia de tu repositorio SuperAgent
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
            # Plantilla inicial para indicarle a Nemotron qué construir desde cero
            contexto_repo[ruta] = f"# Crear la estructura inicial de código para {ruta}"
    return contexto_repo

def main():
    if not os.environ.get("NVIDIA_API_KEY"):
        print("❌ Error: Falta la variable de entorno NVIDIA_API_KEY en los Secrets del repositorio.")
        sys.exit(1)

    print("📦 Mapeando la estructura del repositorio 247Shopmx/SuperAgent...")
    repo_actual = leer_todo_el_repositorio()
    
    prompt_sistema = (
        "Eres Nemotron-3 Ultra (550B), un Ingeniero de Software Principal de NVIDIA y experto en Algoritmos Cuantitativos.\n"
        "Tu tarea es sobrescribir y crear la lógica completa para un bot autónomo de Value Betting (Apuestas de valor) en fútbol.\n\n"
        "REQUISITOS ARQUITECTÓNICOS MÍNIMOS:\n"
        "1. `requirements.txt`: Debe congelar dependencias estables: requests, beautifulsoup4, pandas, openai, lxml.\n"
        "2. `src/scraper_espn.py`: Scraping modular usando BeautifulSoup con simulación de cabeceras (User-Agent) para extraer estadísticas históricas, goles a favor/en contra y xG de ESPN.\n"
        "3. `src/odds_client.py`: Integración limpia con The Odds API ('/v4/sports/soccer/odds') usando la variable de entorno ODDS_API_KEY para jalar cuotas en tiempo real.\n"
        "4. `src/predictor_agent.py`: Cerebro matemático que cruza datos de ESPN contra las cuotas reales. Debe calcular probabilidades implícitas y determinar si hay valor mediante: (Cuota de Casa * Probabilidad del Bot) - 1 > 0.\n"
        "5. `src/main.py`: Orquestador global. Debe llamar al scraper, jalar las cuotas, pasar el filtro del predictor y escribir un reporte analítico exhaustivo en un archivo markdown llamado 'predictions_report.md'.\n\n"
        "REGLA DE FORMATO ABSOLUTA:\n"
        "Responde ÚNICAMENTE con un objeto JSON crudo y válido que contenga las rutas como llaves y el código como valores. "
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

    print("🧠 Conectando con Nemotron-3 Ultra 550B. Procesando análisis estructural del repositorio...")
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Este es el estado de mi repositorio actual:\n\n{json.dumps(repo_actual, indent=2)}"}
            ],
            temperature=0.1,  # Temperatura baja para asegurar consistencia en código de ingeniería
            max_tokens=4096
        )
        
        respuesta_cruda = completion.choices[0].message.content.strip()
        
        # Limpieza de salvaguarda por si se incluyen caracteres markdown de manera involuntaria
        if respuesta_cruda.startswith("```json"):
            respuesta_cruda = respuesta_cruda.split("```json")[1].split("```")[0].strip()
        elif respuesta_cruda.startswith("```"):
            respuesta_cruda = respuesta_cruda.split("```")[1].split("```")[0].strip()

        archivos_mejorados = json.loads(respuesta_cruda)
        
        print("💾 Análisis completado. Inyectando código optimizado en el entorno local...")
        for ruta, contenido in archivos_mejorados.items():
            directorio = os.path.dirname(ruta)
            if directorio:
                os.makedirs(directorio, exist_ok=True)
                
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)
            print(f"✅ Archivo configurado de forma autónoma: {ruta}")
            
        print("🚀 Proceso terminado con éxito. Todos los componentes han sido actualizados.")

    except json.JSONDecodeError:
        print("❌ Error: Nemotron no devolvió un JSON limpio. Respuesta obtenida:")
        print(respuesta_cruda)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error crítico en el canal de comunicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
