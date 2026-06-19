# 🏆 SuperAgent - Sistema de Predicción de Apuestas Deportivas

Un **sistema avanzado de inteligencia artificial** para predecir resultados de partidos de fútbol, calcular probabilidades y gestionar apuestas deportivas con control de riesgo.

---

## 🌟 Características Principales

| Módulo | Descripción |
|--------|-------------|
| **📡 Obtención de Datos** | Scraping de [ESPN](https://www.espn.com) para partidos y API de [The Odds API](https://the-odds-api.com/) para cuotas en tiempo real. |
| **🔧 Procesamiento ETL** | Limpieza, transformación y fusión de datos de partidos y cuotas. |
| **🤖 Modelos de Predicción** | Combina **Poisson** (goles esperados), **XGBoost** (resultado) y **LSTM** (tendencias temporales). |
| **💰 Gestión de Banca** | Control de riesgo con **fórmula de Kelly**, seguimiento de apuestas y stop-loss. |
| **🖥️ Interfaces** | **CLI** (para scripts) y **API REST** (para integración con otras aplicaciones). |

---

## 🛠 Instalación

### 1. Requisitos Previos
- Python **3.8+**
- `pip` (gestor de paquetes de Python)
- Clave API de [The Odds API](https://the-odds-api.com/) (gratis para pruebas)

### 2. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/superagent.git
cd superagent
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Datos del día actual
python main.py fetch

# Datos para una fecha específica (formato YYYYMMDD)
python main.py fetch --date 20231015
# Entrenar con datos por defecto (data/merged.csv)
python main.py train

# Entrenar con un archivo personalizado
python main.py train --data data/historical_matches.csv
