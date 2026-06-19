#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('superagent.log')
    ]
)
logger = logging.getLogger(__name__)

# Importar clases
from data_fetcher import ESPNScraper, OddsAPIClient
from data_processor import DataProcessor
from models import ModelEnsemble, PoissonModel, XGBoostModel, LSTMModel
from bankroll_manager import BankrollManager

class SuperAgentCLI:
    """Interfaz de línea de comandos para el SuperAgent."""

    def __init__(self):
        self.scraper = ESPNScraper()
        self.odds_client = OddsAPIClient()
        self.data_processor = DataProcessor()
        self.bankroll_manager = BankrollManager(initial_bankroll=1000.0, risk_per_bet=0.01)

        # Inicializar modelos
        self.poisson_model = PoissonModel()
        self.xgboost_model = XGBoostModel()
        self.lstm_model = LSTMModel(input_shape=(5, 10))
        self.model_ensemble = ModelEnsemble(
            poisson_model=self.poisson_model,
            xgboost_model=self.xgboost_model,
            lstm_model=self.lstm_model
        )

    def fetch_data(self, date: Optional[str] = None) -> bool:
        """Obtiene datos de partidos y cuotas desde ESPN y Odds API."""
        try:
            logger.info("🔄 Fetching match data from ESPN...")
            matches = self.scraper.fetch_scoreboard(date)
            if not matches:
                logger.warning("⚠️ No matches found for the specified date.")
                return False

            logger.info("🔄 Fetching odds from Odds API...")
            odds = self.odds_client.get_odds()
            if not odds:
                logger.warning("⚠️ No odds data available.")
                return False

            # Procesar datos
            matches_df = self.data_processor.clean_match_data(matches)
            odds_df = self.data_processor.clean_odds_data(odds)
            merged_df = self.data_processor.merge_match_and_odds(matches_df, odds_df)

            if merged_df.empty:
                logger.warning("⚠️ No merged data available (no matching matches and odds).")
                return False

            # Guardar datos en directorio 'data/'
            os.makedirs('data', exist_ok=True)
            self.data_processor.save_to_csv(matches_df, 'data/matches.csv')
            self.data_processor.save_to_csv(odds_df, 'data/odds.csv')
            self.data_processor.save_to_csv(merged_df, 'data/merged.csv')

            logger.info(f"✅ Data saved successfully. Matches: {len(matches_df)}, Odds: {len(odds_df)}")
            return True
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return False

    def train_models(self, data_path: str = 'data/merged.csv') -> bool:
        """Entrena los modelos con datos históricos."""
        try:
            logger.info("📊 Loading training data...")
            df = self.data_processor.load_from_csv(data_path)
            if df.empty:
                logger.error("❌ No training data available. Run 'fetch' first.")
                return False

            # Calcular estadísticas de equipos (últimos 5 partidos)
            team_stats_df = self.data_processor.calculate_team_stats(df, window=5)
            if team_stats_df.empty:
                logger.warning("⚠️ No team statistics calculated (insufficient data).")

            # Preparar características para los modelos
            features_df = self.data_processor.prepare_features(df, team_stats_df)
            if features_df.empty:
                logger.error("❌ No features available for training.")
                return False

            # Extraer variables objetivo
            X = features_df.drop(['result', 'home_goals', 'away_goals'], axis=1, errors='ignore')
            y = features_df['result']
            home_goals = features_df['home_goals']
            away_goals = features_df['away_goals']

            # Entrenar modelos
            logger.info("🎯 Training Poisson model (expected goals)...")
            self.poisson_model.fit(X, home_goals, away_goals)

            logger.info("🎯 Training XGBoost model (match outcome)...")
            self.xgboost_model.fit(X, y)

            logger.info("🎯 Training LSTM model (temporal trends)...")
            self.lstm_model.fit(X, y, window_size=5, epochs=20)

            logger.info("✅ All models trained successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            return False

    def predict_match(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Predice el resultado de un partido y recomienda apuestas con valor."""
        try:
            # Obtener cuotas para el partido
            odds_data = self.odds_client.get_odds_for_match(home_team, away_team)
            odds = {}
            if odds_data:
                for bookmaker in odds_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                odds[outcome.get('name')] = float(outcome.get('price'))

            # Crear características (en producción, usarías datos históricos reales)
            # Aquí usamos valores por defecto para demostración
            features = {
                'home_attack_strength': 1.2,    # Fuerza de ataque del local
                'home_defense_strength': 0.8,   # Fuerza de defensa del local
                'away_attack_strength': 1.0,    # Fuerza de ataque del visitante
                'away_defense_strength': 1.0,   # Fuerza de defensa del visitante
                'home_form': 0.6,               # Forma reciente del local (0-1)
                'away_form': 0.5,               # Forma reciente del visitante (0-1)
                'implied_prob_home': 1 / odds.get('home', 2.5),
                'implied_prob_away': 1 / odds.get('away', 2.5),
                'implied_prob_draw': 1 / odds.get('draw', 3.0)
            }
            features_df = pd.DataFrame([features])

            # Predecir con el ensemble
            proba = self.model_ensemble.predict_proba(features_df)
            home_goals, away_goals = self.model_ensemble.get_expected_goals(features_df)

            # Formatear probabilidades (orden: away_win, draw, home_win)
            probabilities = {
                'home_win': float(proba[0][2]),  # Índice 2: home win (1)
                'draw': float(proba[0][1]),      # Índice 1: draw (0)
                'away_win': float(proba[0][0])   # Índice 0: away win (-1)
            }

            # Determinar apuesta recomendada (si hay valor)
            recommended_bet = None
            max_prob = max(probabilities.values())
            best_outcome = max(probabilities, key=probabilities.get)
            implied_prob = 1 / odds.get(best_outcome, 2.0)

            if max_prob > implied_prob:
                stake = self.bankroll_manager.calculate_stake(
                    odds.get(best_outcome, 2.0),
                    max_prob,
                    method='kelly'
                )
                recommended_bet = {
                    'outcome': best_outcome,
                    'odds': odds.get(best_outcome, 2.0),
                    'probability': max_prob,
                    'implied_probability': implied_prob,
                    'stake': float(stake),
                    'potential_profit': float(stake * (odds.get(best_outcome, 2.0) - 1))
                }

            prediction = {
                'match': f"{home_team} vs {away_team}",
                'timestamp': datetime.now().isoformat(),
                'probabilities': probabilities,
                'expected_goals': {
                    'home': float(home_goals[0]),
                    'away': float(away_goals[0])
                },
                'odds': odds,
                'recommended_bet': recommended_bet
            }
            return prediction
        except Exception as e:
            logger.error(f"❌ Error predicting match: {e}")
            return None

    def place_bet(self, match: str, prediction: str, odds: float, probability: float) -> Dict:
        """Coloca una apuesta basada en una predicción."""
        try:
            result = self.bankroll_manager.place_bet(
                match=match,
                prediction=prediction,
                odds=odds,
                probability=probability,
                stake_method='kelly'
            )
            logger.info(f"💰 Bet placed: {match} | Prediction: {prediction} | Stake: ${result['stake']:.2f}")
            return result
        except Exception as e:
            logger.error(f"❌ Error placing bet: {e}")
            return {'error': str(e)}

    def settle_bet(self, match: str, actual_result: str) -> bool:
        """Resuelve una apuesta después del partido."""
        try:
            success = self.bankroll_manager.settle_bet(
                match=match,
                actual_result=actual_result
            )
            if success:
                logger.info(f"✅ Bet settled: {match} | Result: {actual_result}")
            else:
                logger.warning(f"⚠️ Failed to settle bet for {match}")
            return success
        except Exception as e:
            logger.error(f"❌ Error settling bet: {e}")
            return False

    def show_bankroll(self) -> Dict:
        """Muestra el estado actual del bankroll."""
        return self.bankroll_manager.get_bankroll_status()

    def show_history(self) -> pd.DataFrame:
        """Muestra el historial de apuestas."""
        return self.bankroll_manager.get_bet_history()

    def save_models(self, directory: str = 'models') -> bool:
        """Guarda los modelos entrenados en disco."""
        try:
            os.makedirs(directory, exist_ok=True)
            success = self.model_ensemble.save(directory)
            if success:
                logger.info(f"💾 Models saved to {directory}")
            return success
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
            return False

    def load_models(self, directory: str = 'models') -> bool:
        """Carga modelos entrenados desde disco."""
        try:
            success = self.model_ensemble.load(directory)
            if success:
                logger.info(f"📂 Models loaded from {directory}")
            return success
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False

def main():
    """Función principal para la CLI."""
    parser = argparse.ArgumentParser(
        description='SuperAgent - Football Betting Prediction System',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- Comando: fetch ---
    fetch_parser = subparsers.add_parser('fetch', help='Fetch match data and odds from ESPN and Odds API')
    fetch_parser.add_argument(
        '--date',
        type=str,
        help='Date in YYYYMMDD format (default: today). Example: --date 20231015'
    )

    # --- Comando: train ---
    train_parser = subparsers.add_parser('train', help='Train prediction models with historical data')
    train_parser.add_argument(
        '--data',
        type=str,
        default='data/merged.csv',
        help='Path to training data CSV (default: data/merged.csv)'
    )

    # --- Comando: predict ---
    predict_parser = subparsers.add_parser('predict', help='Predict match outcome and recommend bets')
    predict_parser.add_argument('--home', type=str, required=True, help='Home team name (e.g., "Barcelona")')
    predict_parser.add_argument('--away', type=str, required=True, help='Away team name (e.g., "Real Madrid")')

    # --- Comando: place-bet ---
    place_bet_parser = subparsers.add_parser('place-bet', help='Place a bet based on a prediction')
    place_bet_parser.add_argument('--match', type=str, required=True, help='Match name (e.g., "Barcelona vs Real Madrid")')
    place_bet_parser.add_argument('--prediction', type=str, required=True, choices=['home', 'draw', 'away'], help='Predicted outcome')
    place_bet_parser.add_argument('--odds', type=float, required=True, help='Decimal odds (e.g., 2.5)')
    place_bet_parser.add_argument('--probability', type=float, required=True, help='Estimated probability (0-1, e.g., 0.6)')

    # --- Comando: settle-bet ---
    settle_bet_parser = subparsers.add_parser('settle-bet', help='Settle a bet after the match')
    settle_bet_parser.add_argument('--match', type=str, required=True, help='Match name')
    settle_bet_parser.add_argument('--result', type=str, required=True, choices=['home', 'draw', 'away'], help='Actual match result')

    # --- Comando: bankroll ---
    bankroll_parser = subparsers.add_parser('bankroll', help='Show current bankroll status')

    # --- Comando: history ---
    history_parser = subparsers.add_parser('history', help='Show bet history')

    # --- Comando: save-models ---
    save_models_parser = subparsers.add_parser('save-models', help='Save trained models to disk')
    save_models_parser.add_argument('--dir', type=str, default='models', help='Directory to save models (default: models/)')

    # --- Comando: load-models ---
    load_models_parser = subparsers.add_parser('load-models', help='Load trained models from disk')
    load_models_parser.add_argument('--dir', type=str, default='models', help='Directory to load models from (default: models/)')

    # --- Parse arguments ---
    args = parser.parse_args()
    agent = SuperAgentCLI()

    if not args.command:
        parser.print_help()
        return

    # --- Ejecutar comandos ---
    if args.command == 'fetch':
        agent.fetch_data(args.date)

    elif args.command == 'train':
        if agent.train_models(args.data):
            print("\n✅ Models trained successfully! You can now make predictions.")
        else:
            print("\n❌ Failed to train models. Check logs for details.")

    elif args.command == 'predict':
        prediction = agent.predict_match(args.home, args.away)
        if prediction:
            print("\n" + "="*60)
            print(f"🔮 PREDICTION: {prediction['match'].upper()}")
            print("="*60)

            print("\n📊 Expected Goals:")
            print(f"   Home: {prediction['expected_goals']['home']:.2f}")
            print(f"   Away: {prediction['expected_goals']['away']:.2f}")

            print("\n🎲 Probabilities:")
            for outcome, prob in prediction['probabilities'].items():
                bar = "█" * int(prob * 20)
                print(f"   {outcome.replace('_', ' ').title():<10} {prob*100:>5.1f}%  {bar}")

            print("\n💰 Current Odds:")
            for outcome, odd in prediction['odds'].items():
                print(f"   {outcome:<10} {odd:>5.2f}")

            if prediction['recommended_bet']:
                bet = prediction['recommended_bet']
                print("\n🎯 RECOMMENDED BET (VALUE FOUND!):")
                print(f"   Outcome:       {bet['outcome']}")
                print(f"   Odds:          {bet['odds']:.2f}")
                print(f"   Your Prob:     {bet['probability']*100:.1f}%")
                print(f"   Implied Prob:  {bet['implied_probability']*100:.1f}%")
                print(f"   Edge:          +{(bet['probability'] - bet['implied_probability'])*100:.1f}%")
                print(f"   Stake:         ${bet['stake']:.2f}")
                print(f"   Potential:     ${bet['potential_profit']:.2f}")
            else:
                print("\n⚠️ No value bet found (no edge over bookmaker odds).")

    elif args.command == 'place-bet':
        result = agent.place_bet(
            match=args.match,
            prediction=args.prediction,
            odds=args.odds,
            probability=args.probability
        )
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n" + "="*60)
            print("💰 BET PLACED SUCCESSFULLY")
            print("="*60)
            print(f"   Match:         {args.match}")
            print(f"   Prediction:    {args.prediction}")
            print(f"   Odds:          {args.odds:.2f}")
            print(f"   Stake:         ${result['stake']:.2f}")
            print(f"   Potential:     ${result['potential_profit']:.2f}")
            print(f"   New Bankroll:  ${result['new_bankroll']:.2f}")

    elif args.command == 'settle-bet':
        success = agent.settle_bet(
            match=args.match,
            actual_result=args.result
        )
        if success:
            print(f"\n✅ Bet settled for {args.match} with result: {args.result}")
        else:
            print(f"\n❌ Failed to settle bet for {args.match}")

    elif args.command == 'bankroll':
        status = agent.show_bankroll()
        print("\n" + "="*60)
        print("💵 BANKROLL STATUS")
        print("="*60)
        print(f"   Initial Bankroll:    ${status['initial_bankroll']:>10.2f}")
        print(f"   Current Bankroll:    ${status['current_bankroll']:>10.2f}")
        print(f"   Total Profit:        ${status['total_profit']:>10.2f} ({status['return_on_investment']:>6.2f}%)")
        print(f"   Total Bets:          {status['total_bets']:>10}")
        print(f"   Winning Bets:        {status['winning_bets']:>10}")
        print(f"   Losing Bets:         {status['losing_bets']:>10}")
        print(f"   Win Rate:            {status['win_rate']*100:>10.1f}%")
        print(f"   Risk per Bet:        {status['risk_per_bet']*100:>10.1f}%")

    elif args.command == 'history':
        history = agent.show_history()
        if history.empty:
            print("\n⚠️ No bet history available.")
        else:
            print("\n" + "="*60)
            print("📜 BET HISTORY")
            print("="*60)
            # Formatear el DataFrame para mejor visualización
            history_display = history.copy()
            history_display['timestamp'] = pd.to_datetime(history_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            history_display['profit'] = history_display['profit'].apply(lambda x: f"${x:.2f}")
            history_display['stake'] = history_display['stake'].apply(lambda x: f"${x:.2f}")
            history_display['potential_profit'] = history_display['potential_profit'].apply(lambda x: f"${x:.2f}")
            history_display['bankroll_after'] = history_display['bankroll_after'].apply(lambda x: f"${x:.2f}")
            print(history_display.to_string(index=False))

    elif args.command == 'save-models':
        if agent.save_models(args.dir):
            print(f"\n✅ Models saved to {args.dir}/")
        else:
            print(f"\n❌ Failed to save models to {args.dir}/")

    elif args.command == 'load-models':
        if agent.load_models(args.dir):
            print(f"\n✅ Models loaded from {args.dir}/")
        else:
            print(f"\n❌ Failed to load models from {args.dir}/")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
