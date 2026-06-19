from flask import Flask, request, jsonify, Blueprint
from typing import Dict, List, Optional
import logging
from datetime import datetime
import os

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
        lstm_model = LSTMModel(input_shape=(5, 10))  # Ajustar según características
        model_ensemble = ModelEnsemble(
            poisson_model=poisson_model,
            xgboost_model=xgboost_model,
            lstm_model=lstm_model
        )

# Blueprint para la API de predicciones
predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predict', methods=['GET'])
def predict_match():
    """Endpoint para obtener predicciones de un partido."""
    initialize_components()

    home_team = request.args.get('home_team')
    away_team = request.args.get('away_team')

    if not home_team or not away_team:
        return jsonify({'error': 'home_team and away_team parameters are required'}), 400

    try:
        # Obtener datos del partido (simplificado para el ejemplo)
        # En una implementación real, buscarías en la base de datos o API
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': 0,
            'away_goals': 0,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'league': 'Example League'
        }

        # Obtener cuotas (simplificado)
        odds_data = odds_client.get_odds_for_match(home_team, away_team)
        odds = {}
        if odds_data:
            for bookmaker in odds_data.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            odds[outcome.get('name')] = outcome.get('price')

        # Preparar características (simplificado)
        # En una implementación real, usarías datos históricos y estadísticas
        features = pd.DataFrame([{
            'home_attack_strength': 1.0,
            'home_defense_strength': 1.0,
            'away_attack_strength': 1.0,
            'away_defense_strength': 1.0,
            'home_form': 0.5,
            'away_form': 0.5,
            'implied_prob_home': 1 / odds.get('home', 2.0),
            'implied_prob_away': 1 / odds.get('away', 2.0),
            'implied_prob_draw': 1 / odds.get('draw', 3.0)
        }])

        # Predecir con el ensemble
        proba = model_ensemble.predict_proba(features)
        home_goals, away_goals = model_ensemble.get_expected_goals(features)

        # Formatear respuesta
        prediction = {
            'match': f"{home_team} vs {away_team}",
            'timestamp': datetime.now().isoformat(),
            'probabilities': {
                'home_win': float(proba[0][2]),  # Índice 2 = home win (1)
                'draw': float(proba[0][1]),     # Índice 1 = draw (0)
                'away_win': float(proba[0][0])   # Índice 0 = away win (-1)
            },
            'expected_goals': {
                'home': float(home_goals[0]),
                'away': float(away_goals[0])
            },
            'odds': odds,
            'recommended_bet': None
        }

        # Determinar apuesta recomendada (si hay ventaja)
        max_prob = max(prediction['probabilities'].values())
        best_outcome = max(prediction['probabilities'], key=prediction['probabilities'].get)
        implied_prob = 1 / odds.get(best_outcome, 2.0)

        if max_prob > implied_prob:
            stake = bankroll_manager.calculate_stake(
                odds.get(best_outcome, 2.0),
                max_prob,
                method='kelly'
            )
            prediction['recommended_bet'] = {
                'outcome': best_outcome,
                'odds': odds.get(best_outcome, 2.0),
                'probability': max_prob,
                'implied_probability': implied_prob,
                'stake': float(stake),
                'potential_profit': float(stake * (odds.get(best_outcome, 2.0) - 1))
            }

        return jsonify(prediction)

    except Exception as e:
        logger.error(f"Error in predict_match: {e}")
        return jsonify({'error': str(e)}), 500

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

        # Preparar características (simplificado)
        # En una implementación real, usarías datos históricos
        features = data_processor.prepare_features(merged_df, pd.DataFrame())

        if features.empty:
            return jsonify({'error': 'No features available for predictions'}), 404

        # Predecir con el ensemble
        probas = model_ensemble.predict_proba(features)
        home_goals, away_goals = model_ensemble.get_expected_goals(features)

        # Formatear respuesta
        predictions = []
        for i, row in merged_df.iterrows():
            proba = probas[i]
            prediction = {
                'match': f"{row['home_team']} vs {row['away_team']}",
                'league': row.get('league', 'Unknown'),
                'date': row.get('date', 'Unknown'),
                'probabilities': {
                    'home_win': float(proba[2]),
                    'draw': float(proba[1]),
                    'away_win': float(proba[0])
                },
                'expected_goals': {
                    'home': float(home_goals[i]),
                    'away': float(away_goals[i])
                },
                'odds': {
                    'home': float(row.get('h2h_home', 0)),
                    'draw': float(row.get('h2h_draw', 0)),
                    'away': float(row.get('h2h_away', 0))
                }
            }
            predictions.append(prediction)

        return jsonify({'predictions': predictions})

    except Exception as e:
        logger.error(f"Error in get_all_predictions: {e}")
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
            odds=data['odds'],
            probability=data['probability'],
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
        # Obtener datos históricos (simplificado)
        # En una implementación real, cargarías datos de un archivo o base de datos
        historical_matches = scraper.fetch_historical_matches("Barcelona", "2023")
        if not historical_matches:
            return jsonify({'error': 'No historical data available'}), 404

        # Procesar datos
        matches_df = data_processor.clean_match_data(historical_matches)
        team_stats_df = data_processor.calculate_team_stats(matches_df)
        features_df = data_processor.prepare_features(matches_df, team_stats_df)

        if features_df.empty:
            return jsonify({'error': 'No features available for training'}), 404

        # Entrenar modelos
        X = features_df.drop(['result'], axis=1, errors='ignore')
        y = features_df['result']

        # Entrenar Poisson
        home_goals = features_df['home_goals']
        away_goals = features_df['away_goals']
        model_ensemble.poisson_model.fit(X, home_goals, away_goals)

        # Entrenar XGBoost
        model_ensemble.xgboost_model.fit(X, y)

        # Entrenar LSTM (necesita secuencias)
        # Simplificado: usar solo los últimos 5 partidos
        model_ensemble.lstm_model.fit(X, y, window_size=5, epochs=10)

        return jsonify({'status': 'success', 'message': 'Models trained'})

    except Exception as e:
        logger.error(f"Error in train_models: {e}")
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
                'train': '/api/models/train (POST)',
                'save': '/api/models/save?directory=models (POST)',
                'load': '/api/models/load?directory=models (POST)'
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
