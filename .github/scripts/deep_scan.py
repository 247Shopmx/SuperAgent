import os
import json
import sys
import requests
from openai import OpenAI

# URL Base oficial de integración de NVIDIA NIM
BASE_URL = "https://integrate.api.nvidia.com/v1"

def obtener_modelo_nemotron_activo(api_key):
    """
    Punto 5 del Checklist: Consulta el endpoint /models para listar los 
    identificadores activos y evitar el error 404 de nombre de modelo.
    """
    print("🔍 Listando modelos disponibles en el endpoint de NVIDIA...")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=10)
        if response.status_code == 200:
            modelos = response.json().get("data", [])
            # Buscamos cualquier modelo que contenga 'nemotron' en su ID
            ids_nemotron = [m.get("id") for m in modelos if "nemotron" in m.get("id", "").lower()]
            
            if ids_nemotron:
                print(f"✅ Modelos Nemotron encontrados en tu tier: {ids_nemotron}")
                # Preferimos la versión ultra si está en la lista, si no, tomamos el primero disponible
                for model_id in ids_nemotron:
                    if "ultra" in model_id:
                        return model_id
                return ids_nemotron[0]
            else:
                print("⚠️ No se encontraron modelos con el nombre 'nemotron' asignados a tu cuenta.")
                if modelos:
                    print(f"Modelos alternativos disponibles: {[m.get('id') for m in modelos[:3]]}")
        else:
            print(f"❌ Error al consultar /models (Código {response.status_code}): {response.text}")
    except Exception as e:
        print(f"⚠️ No se pudo auto-detectar el modelo mediante la API: {e}")
    
    # Modelo por defecto si el fetch falla (Ajustado según catálogo de producción)
    return "nvidia/nemotron-3-ultra"

# Inicializar cliente de NVIDIA NIM
client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ.get("NVIDIA_API_KEY")
)

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
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: Falta la variable de entorno NVIDIA_API_KEY en los Secrets de GitHub.")
        sys.exit(1)

    # Paso de auto-detección para corregir la causa #2 del error 404
    model_name = obtener_modelo_nemotron_activo(api_key)

    print(f"📦 Mapeando el repositorio actual (247Shopmx/SuperAgent)...")
    repo_actual = leer_todo_el_repositorio()
    
    prompt_sistema = (
        "Eres un Ingeniero de Software experto en Algoritmos Cuantitativos y APIs de NVIDIA.\n"
        "Tu tarea es generar el código completo de un bot de Value Betting en fútbol que haga scraping de ESPN y consuma The Odds API.\n\n"
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

    print(f"🧠 Enviando solicitud a NVIDIA NIM utilizando el modelo: {model_name}...")
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Estructura del repositorio actual:\n\n{json.dumps(repo_actual, indent=2)}"}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        respuesta_cruda = completion.choices[0].message.content.strip()
        
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
        print(f"❌ Error crítico en la llamada de inferencia a {model_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
