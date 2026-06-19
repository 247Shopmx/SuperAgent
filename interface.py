from flask import Flask, request, jsonify, Blueprint
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import os
import pandas as pd

# Importar clases de otros módulos
from data_fetcher import ESPNScraper, OddsAPIClient
from data_processor import DataProcessor
from models import ModelEnsemble, PoissonModel, XGBoostModel, LSTMModel
from bankroll_manager import BankrollManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Inicializar componentes (se cargarán en el primer request o al iniciar)
scraper = None
odds_client = None
data_processor = None
model_ensemble = None
bankroll_manager = None

def initialize_components():
    """Inicializa todos los componentes del SuperAgent."""
    global scraper, odds_client, data_processor, model_ensemble, bankroll_manager

    if scraper is None:
        scraper = ESPNScraper()
    if odds_client is None:
        odds_client = OddsAPIClient()
    if data_processor is None:
        data_processor = DataProcessor()
    if bankroll_manager is None:
        bankroll_manager = BankrollManager(initial_bankroll=1000.0, risk_per_bet=0.01)

    # Inicializar modelos (si no están cargados)
    if model_ensemble is None:
        poisson_model = PoissonModel()
        xgboost_model = XGBoostModel()
        lstm_model = LSTMModel(input_shape=(5, 10))
        model_ensemble = ModelEnsemble(
            poisson_model=poisson_model,
            xgboost_model=xgboost_model,
            lstm_model=lstm_model
        )

# Blueprint para la API de predicciones
predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predict', methods=['GET'])
def predict_match():
    """Endpoint para obtener predicciones de un partido con manejo robusto de errores."""
    initialize_components()

    home_team = request.args.get('home_team')
    away_team = request.args.get('away_team')

    if not home_team or not away_team:
        return jsonify({
            'error': 'Parámetros requeridos',
            'required': ['home_team', 'away_team']
        }), 400

    try:
        # Validar que los modelos estén cargados
        if model_ensemble is None:
            return jsonify({
                'error': 'Modelos no disponibles',
                'solution': 'Cargar modelos primero usando /api/models/load'
            }), 503

        # Obtener datos del partido
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': 0,
            'away_goals': 0,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'league': 'Example League'
        }

        # Obtener cuotas
        odds_data = odds_client.get_odds_for_match(home_team, away_team)
        odds = {}
        if odds_data:
            for bookmaker in odds_data.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            outcome_name = outcome.get('name', '').lower()
                            if home_team.lower() in outcome_name:
                                odds['home'] = float(outcome.get('price', 2.0))
                            elif away_team.lower() in outcome_name:
                                odds['away'] = float(outcome.get('price', 2.0))
                            elif 'draw' in outcome_name:
                                odds['draw'] = float(outcome.get('price', 3.0))

        # Valores por defecto si no hay cuotas
        odds.setdefault('home', 2.0)
        odds.setdefault('away', 2.0)
        odds.setdefault('draw', 3.0)

        # Preparar características
        features = pd.DataFrame([{
            'home_attack_strength': 1.0,
            'home_defense_strength': 1.0,
            'away_attack_strength': 1.0,
            'away_defense_strength': 1.0,
            'home_form': 0.5,
            'away_form': 0.5,
            'implied_prob_home': 1 / odds['home'],
            'implied_prob_away': 1 / odds['away'],
            'implied_prob_draw': 1 / odds['draw']
        }])

        # Predecir con el ensemble
        proba = model_ensemble.predict_proba(features)
        home_goals, away_goals = model_ensemble.get_expected_goals(features)

        # Formatear respuesta (proba orden: [away_win, draw, home_win])
        prediction = {
            'match': f"{home_team} vs {away_team}",
            'timestamp': datetime.now().isoformat(),
            'probabilities': {
                'home_win': float(proba[0][2]),
                'draw': float(proba[0][1]),
                'away_win': float(proba[0][0])
            },
            'expected_goals': {
                'home': float(home_goals[0]),
                'away': float(away_goals[0])
            },
            'odds': odds,
            'recommended_bet': None
        }

        # Determinar apuesta recomendada
        prob_map = {
            'home': prediction['probabilities']['home_win'],
            'draw': prediction['probabilities']['draw'],
            'away': prediction['probabilities']['away_win']
        }
        
        best_outcome = max(prob_map, key=prob_map.get)
        max_prob = prob_map[best_outcome]
        implied_prob = 1 / odds[best_outcome]

        if max_prob > implied_prob * 1.05:  # 5% margen de seguridad
            stake = bankroll_manager.calculate_stake(
                odds[best_outcome],
                max_prob,
                method='kelly'
            )
            prediction['recommended_bet'] = {
                'outcome': best_outcome,
                'odds': odds[best_outcome],
                'probability': max_prob,
                'implied_probability': implied_prob,
                'edge': max_prob - implied_prob,
                'stake': float(stake),
                'potential_profit': float(stake * (odds[best_outcome] - 1))
            }

        return jsonify(prediction)

    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        return jsonify({
            'error': 'Error de validación',
            'message': str(e)
        }), 400
    except KeyError as e:
        logger.error(f"Falta campo en datos: {e}")
        return jsonify({
            'error': 'Campo faltante',
            'field': str(e)
        }), 422
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        return jsonify({
            'error': 'Error interno del servidor',
            'message': 'Por favor intente de nuevo más tarde'
        }), 500

@predictions_bp.route('/predictions', methods=['GET'])
def get_all_predictions():
    """Endpoint para obtener predicciones de todos los partidos disponibles."""
    initialize_components()

    try:
        # Obtener partidos del día
        matches = scraper.fetch_scoreboard()
        if not matches:
            return jsonify({'error': 'No matches found'}), 404

        # Obtener cuotas
        odds = odds_client.get_odds()

        # Procesar datos
        matches_df = data_processor.clean_match_data(matches)
        odds_df = data_processor.clean_odds_data(odds)
        merged_df = data_processor.merge_match_and_odds(matches_df, odds_df)

        if merged_df.empty:
            return jsonify({'error': 'No data available for predictions'}), 404

        # Calcular estadísticas de equipos
        team_stats_df = data_processor.calculate_team_stats(merged_df, window=5)
        features = data_processor.prepare_features(merged_df, team_stats_df)

        if features.empty:
            return jsonify({'error': 'No features available for predictions'}), 404

        # Predecir con el ensemble
        probas = model_ensemble.predict_proba(features)
        home_goals, away_goals = model_ensemble.get_expected_goals(features)

        # Formatear respuesta
        predictions = []
        for i, row in merged_df.iterrows():
            if i >= len(probas):
                break
                
            proba = probas[i]
            prediction = {
                'match': f"{row['home_team']} vs {row['away_team']}",
                'league': row.get('league', 'Unknown'),
                'date': str(row.get('date', 'Unknown')),
                'probabilities': {
                    'home_win': float(proba[2]),
                    'draw': float(proba[1]),
                    'away_win': float(proba[0])
                },
                'expected_goals': {
                    'home': float(home_goals[i]) if i < len(home_goals) else 0.0,
                    'away': float(away_goals[i]) if i < len(away_goals) else 0.0
                },
                'odds': {
                    'home': float(row.get('h2h_home', 0)) if not pd.isna(row.get('h2h_home')) else 0.0,
                    'draw': float(row.get('h2h_draw', 0)) if not pd.isna(row.get('h2h_draw')) else 0.0,
                    'away': float(row.get('h2h_away', 0)) if not pd.isna(row.get('h2h_away')) else 0.0
                }
            }
            predictions.append(prediction)

        return jsonify({'predictions': predictions, 'count': len(predictions)})

    except Exception as e:
        logger.error(f"Error in get_all_predictions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Blueprint para la API de bankroll
bankroll_bp = Blueprint('bankroll', __name__)

@bankroll_bp.route('/status', methods=['GET'])
def get_bankroll_status():
    """Endpoint para obtener el estado del bankroll."""
    initialize_components()
    try:
        status = bankroll_manager.get_bankroll_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in get_bankroll_status: {e}")
        return jsonify({'error': str(e)}), 500

@bankroll_bp.route('/history', methods=['GET'])
def get_bet_history():
    """Endpoint para obtener el historial de apuestas."""
    initialize_components()
    try:
        history = bankroll_manager.get_bet_history()
        return jsonify(history.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error in get_bet_history: {e}")
        return jsonify({'error': str(e)}), 500

@bankroll_bp.route('/place_bet', methods=['POST'])
def place_bet():
    """Endpoint para colocar una apuesta."""
    initialize_components()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['match', 'prediction', 'odds', 'probability']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        result = bankroll_manager.place_bet(
            match=data['match'],
            prediction=data['prediction'],
            odds=float(data['odds']),
            probability=float(data['probability']),
            stake_method=data.get('stake_method', 'kelly')
        )

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in place_bet: {e}")
        return jsonify({'error': str(e)}), 500

@bankroll_bp.route('/settle_bet', methods=['POST'])
def settle_bet():
    """Endpoint para resolver una apuesta."""
    initialize_components()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['match', 'actual_result']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        success = bankroll_manager.settle_bet(
            match=data['match'],
            actual_result=data['actual_result'],
            bet_index=data.get('bet_index')
        )

        if success:
            return jsonify({'status': 'success', 'message': 'Bet settled'})
        else:
            return jsonify({'error': 'Failed to settle bet'}), 400
    except Exception as e:
        logger.error(f"Error in settle_bet: {e}")
        return jsonify({'error': str(e)}), 500

# Blueprint para la API de modelos
models_bp = Blueprint('models', __name__)

@models_bp.route('/train', methods=['POST'])
def train_models():
    """Endpoint para entrenar los modelos con datos históricos."""
    initialize_components()
    try:
        data_path = request.args.get('data_path', 'data/merged.csv')
        
        # Cargar datos
        if not os.path.exists(data_path):
            return jsonify({'error': f'Data file not found: {data_path}'}), 404
            
        df = data_processor.load_from_csv(data_path)
        if df.empty:
            return jsonify({'error': 'No data in file'}), 400

        # Calcular estadísticas
        team_stats_df = data_processor.calculate_team_stats(df, window=5)
        features_df = data_processor.prepare_features(df, team_stats_df)

        if features_df.empty:
            return jsonify({'error': 'No features available for training'}), 400

        # Preparar datos
        required_cols = ['result', 'home_goals', 'away_goals']
        if not all(col in features_df.columns for col in required_cols):
            return jsonify({'error': f'Missing required columns: {required_cols}'}), 400

        X = features_df.drop(required_cols, axis=1, errors='ignore')
        y = features_df['result']
        home_goals = features_df['home_goals']
        away_goals = features_df['away_goals']

        # Entrenar modelos
        model_ensemble.poisson_model.fit(X, home_goals, away_goals)
        xgb_metrics = model_ensemble.xgboost_model.fit(X, y)
        lstm_metrics = model_ensemble.lstm_model.fit(X, y, window_size=5, epochs=10)

        return jsonify({
            'status': 'success',
            'message': 'Models trained',
            'metrics': {
                'xgboost': xgb_metrics,
                'lstm': lstm_metrics
            }
        })

    except Exception as e:
        logger.error(f"Error in train_models: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@models_bp.route('/save', methods=['POST'])
def save_models():
    """Endpoint para guardar los modelos entrenados."""
    initialize_components()
    try:
        directory = request.args.get('directory', 'models')
        success = model_ensemble.save(directory)
        if success:
            return jsonify({'status': 'success', 'message': f'Models saved to {directory}'})
        else:
            return jsonify({'error': 'Failed to save models'}), 500
    except Exception as e:
        logger.error(f"Error in save_models: {e}")
        return jsonify({'error': str(e)}), 500

@models_bp.route('/load', methods=['POST'])
def load_models():
    """Endpoint para cargar modelos entrenados."""
    initialize_components()
    try:
        directory = request.args.get('directory', 'models')
        if not os.path.exists(directory):
            return jsonify({'error': f'Directory not found: {directory}'}), 404
            
        success = model_ensemble.load(directory)
        if success:
            return jsonify({'status': 'success', 'message': f'Models loaded from {directory}'})
        else:
            return jsonify({'error': 'Failed to load models'}), 500
    except Exception as e:
        logger.error(f"Error in load_models: {e}")
        return jsonify({'error': str(e)}), 500

# Registrar blueprints
app.register_blueprint(predictions_bp, url_prefix='/api/predictions')
app.register_blueprint(bankroll_bp, url_prefix='/api/bankroll')
app.register_blueprint(models_bp, url_prefix='/api/models')

@app.route('/')
def index():
    """Endpoint raíz con información de la API."""
    return jsonify({
        'name': 'SuperAgent Football Betting API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'predictions': {
                'predict': '/api/predictions/predict?home_team=TeamA&away_team=TeamB',
                'all_predictions': '/api/predictions/predictions'
            },
            'bankroll': {
                'status': '/api/bankroll/status',
                'history': '/api/bankroll/history',
                'place_bet': '/api/bankroll/place_bet (POST)',
                'settle_bet': '/api/bankroll/settle_bet (POST)'
            },
            'models': {
                'train': '/api/models/train?data_path=data/merged.csv (POST)',
                'save': '/api/models/save?directory=models (POST)',
                'load': '/api/models/load?directory=models (POST)'
            }
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
