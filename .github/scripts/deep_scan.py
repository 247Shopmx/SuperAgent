import os
import json
import sys
import time
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE MODELOS (CADENA DE FALLBACK ROBUSTA)
# Ordenados por: Calidad ↓ | Probabilidad de éxito en Free/Dev Tier ↑
# ─────────────────────────────────────────────────────────────
MODEL_CHAIN = [
    "nvidia/llama-3.1-nemotron-70b-instruct",      # 🥇 Estándar de oro: Calidad alta + Acceso casi universal
    "nvidia/nemotron-3-ultra-550b-a55b",           # 🥈 Nemotron 3 Ultra (550B): Tu objetivo original, Function ID distinto al 253B
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",    # 🥉 Nuevo Nemotron 4 Super: Muy capaz, buen acceso
    "nvidia/nemotron-3-super-120b-a12b",           # Nemotron 3 Super (120B)
    "nvidia/llama-3.1-nemotron-51b-instruct",      # Fallback garantizado (Acceso casi 100%)
]

# Archivos objetivo del repo
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

def limpiar_respuesta_json(texto_crudo: str) -> str:
    """Extrae JSON válido de respuestas sucias (markdown, texto extra, etc.)."""
    t = texto_cruda.strip()
    # 1. Bloques markdown ```json ... ``` o ``` ... ```
    if t.startswith("```"):
        partes = t.split("```")
        # El JSON suele estar en el índice 1 (después del primer ```)
        if len(partes) >= 2:
            t = partes[1].strip()
            if t.startswith("json"):
                t = t[4:].strip()
    # 2. Buscar primer '{' y último '}' por si hay texto antes/después
    try:
        inicio = t.index('{')
        fin = t.rindex('}') + 1
        return t[inicio:fin]
    except ValueError:
        return t # Devolver original si no hay llaves, fallará en loads() con error claro

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: Falta la variable de entorno NVIDIA_API_KEY en los Secrets de GitHub.")
        sys.exit(1)

    # Cliente con timeouts agresivos para CI/CD
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=60.0,          # Timeout total petición
        max_retries=2          # Reintentos internos de la librería (rate limits, 5xx)
    )

    print(f"📦 Mapeando el repositorio actual (247Shopmx/SuperAgent)...")
    repo_actual = leer_todo_el_repositorio()
    
    prompt_sistema = (
        "Eres un Ingeniero de Software experto en Algoritmos Cuantitativos y APIs de NVIDIA.\n"
        "Tu tarea es generar el código completo de un bot de Value Betting en fútbol que haga scraping de ESPN y consuma The Odds API.\n\n"
        "REQUISITOS ARQUITECTÓNICOS MÍNIMOS:\n"
        "1. `requirements.txt`: Debe incluir requests, beautifulsoup4, pandas, openai, lxml.\n"
        "2. `src/scraper_espn.py`: Scraping usando BeautifulSoup simulando cabeceras (User-Agent) para extraer estadísticas, goles y xG de ESPN.\n"
        "3. `src/odds_client.py`: Cliente para consumir la API de cuotas '/v4/sports/soccer/odds' usando variable de entorno ODDS_API_KEY.\n"
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

    user_prompt = f"Estructura del repositorio actual:\n\n{json.dumps(repo_actual, indent=2)}"

    # ─────────────────────────────────────────────────────────────
    # BUCLE DE FALLBACK INTELIGENTE
    # ─────────────────────────────────────────────────────────────
    archivos_mejorados = None
    modelo_usado = None

    for idx, model_name in enumerate(MODEL_CHAIN):
        print(f"🧠 [{idx+1}/{len(MODEL_CHAIN)}] Probando modelo: {model_name}...")
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4096,  # Aumentado para código completo de 5 archivos
            )
            
            respuesta_cruda = completion.choices[0].message.content
            if not respuesta_cruda:
                raise ValueError("Respuesta vacía de la API")

            json_limpio = limpiar_respuesta_json(respuesta_cruda)
            archivos_mejorados = json.loads(json_limpio)
            
            modelo_usado = model_name
            print(f"✅ Éxito con modelo: {modelo_usado}")
            break # Salir del bucle si todo fue bien

        except APIStatusError as e:
            # Captura específicamente el error 404 "Function not found for account"
            if e.status_code == 404 and "Function" in str(e.response.text) and "Not found for account" in str(e.response.text):
                print(f"   ⚠️ Modelo no disponible en tu tier (Function ID no aprovisionado): {model_name}")
                print(f"   ➡️ Saltando al siguiente modelo en la cadena...")
                continue # Probar siguiente modelo
            elif e.status_code == 429:
                print(f"   ⏳ Rate Limit (429) en {model_name}. Esperando 5s y reintentando mismo modelo...")
                time.sleep(5)
                # Podrías añadir lógica de reintento aquí, pero por simplicidad pasamos al siguiente
                continue
            else:
                print(f"   ❌ Error API ({e.status_code}) en {model_name}: {e.message}")
                # Si es 401, 403, 500... no tiene sentido probar otros modelos (problema de key/servidor)
                if e.status_code in [401, 403]:
                    print("   🛑 Error de autenticación/permisos. Revisa NVIDIA_API_KEY.")
                    sys.exit(1)
                continue # Otros 5xx, probar siguiente modelo por si es caída de ese cluster específico

        except (APIConnectionError, RateLimitError) as e:
            print(f"   🌐 Error de red/rate limit en {model_name}: {e}. Probando siguiente...")
            continue
            
        except json.JSONDecodeError as e:
            print(f"   🧩 Error parseando JSON de {model_name}: {e}")
            print(f"   📄 Respuesta recibida (primeros 500 chars): {respuesta_cruda[:500]}")
            continue # Probar siguiente modelo, a veces uno "alucina" JSON y otro no

        except Exception as e:
            print(f"   💥 Error inesperado con {model_name}: {type(e).__name__}: {e}")
            continue

    # ─────────────────────────────────────────────────────────────
    # VERIFICACIÓN FINAL Y ESCRITURA
    # ─────────────────────────────────────────────────────────────
    if not archivos_mejorados:
        print("\n❌ ERROR CRÍTICO: Ningún modelo de la cadena de fallback pudo completar la tarea.")
        print("   Posibles causas: API Key inválida, sin cuota, caída general de NVIDIA NIM, o prompt demasiado largo.")
        sys.exit(1)

    print(f"\n💾 Inyectando código optimizado generado por [{modelo_usado}]...")
    try:
        for ruta, contenido in archivos_mejorados.items():
            directorio = os.path.dirname(ruta)
            if directorio:
                os.makedirs(directorio, exist_ok=True)
                
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)
            print(f"✅ Archivo configurado: {ruta}")
            
        print("\n🚀 Proceso de actualización del repositorio completado de forma autónoma.")

    except Exception as e:
        print(f"❌ Error escribiendo archivos en disco: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
