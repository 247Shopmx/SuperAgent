import os
import json
from openai import OpenAI

# Inicializar cliente de NVIDIA NIM para Nemotron-3 Ultra 550B
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# Definir qué archivos queremos que el modelo analice y mejore obligatoriamente
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
            contexto_repo[ruta] = "# Archivo nuevo o vacío por implementar"
    return contexto_repo

def main():
    print("📦 Leyendo el estado actual del repositorio...")
    repo_actual = leer_todo_el_repositorio()
    
    prompt_sistema = (
        "Eres Nemotron-3 Ultra (550B), un Ingeniero de Software Principal de nivel mundial. "
        "Tu tarea es recibir el código de un bot de predicciones deportivas de valor (que consume ESPN y The Odds API) "
        "y reescribir TODOS los archivos para optimizarlos al máximo nivel de producción.\n\n"
        "Debes responder EXCLUSIVAMENTE con un objeto JSON válido que contenga la estructura de los archivos mejorados. "
        "No incluyas texto explicativo, ni bloques de código formateados con ```json. Solo el JSON puro.\n\n"
        "El formato del JSON debe ser exactamente este:\n"
        "{\n"
        '  "requirements.txt": "contenido_del_archivo...",\n'
        '  "src/scraper_espn.py": "contenido_del_archivo...",\n'
        '  "src/odds_client.py": "contenido_del_archivo...",\n'
        '  "src/predictor_agent.py": "contenido_del_archivo...",\n'
        '  "src/main.py": "contenido_del_archivo..."\n'
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
            temperature=0.1,  # Ultra preciso, sin variaciones creativas extrañas
            max_tokens=4096
        )
        
        respuesta_cruda = completion.choices[0].message.content.strip()
        
        # Limpieza de seguridad por si el modelo incluye Markdown de JSON
        if respuesta_cruda.startswith("```json"):
            respuesta_cruda = respuesta_cruda.split("```json")[1].split("```")[0].strip()
        elif respuesta_cruda.startswith("```"):
            respuesta_cruda = respuesta_cruda.split("```")[1].split("```")[0].strip()

        archivos_mejorados = json.loads(respuesta_cruda)
        
        print("💾 Nemotron ha devuelto el código optimizado. Reescribiendo archivos locales...")
        for ruta, contenido in archivos_mejorados.items():
            # Crear directorios si no existen
            os.makedirs(os.path.dirname(ruta), exist_ok=True) if os.path.dirname(ruta) else None
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)
            print(f"✅ {ruta} actualizado de forma autónoma.")
            
        print("🚀 Refactorización completa terminada con éxito.")

    except Exception as e:
        print(f"❌ Error crítico durante el escaneo y actualización: {e}")
        exit(1)

if __name__ == "__main__":
    main()
