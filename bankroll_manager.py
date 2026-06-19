import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import pickle

class BankrollManager:
    """Gestión de banca, apuestas y control de riesgo."""

    def __init__(self, initial_bankroll: float = 1000.0, risk_per_bet: float = 0.01):
        """
        Inicializa el gestor de banca.

        Args:
            initial_bankroll: Capital inicial.
            risk_per_bet: Porcentaje del bankroll a arriesgar por apuesta (ej: 0.01 = 1%).
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.risk_per_bet = risk_per_bet
        self.bet_history = pd.DataFrame(columns=[
            'timestamp', 'match', 'prediction', 'odds', 'stake',
            'potential_profit', 'result', 'profit', 'bankroll_after'
        ])
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def calculate_stake(
        self,
        odds: float,
        probability: float,
        method: str = 'kelly'
    ) -> float:
        """
        Calcula el monto a apostar usando diferentes métodos con verificaciones de seguridad.

        Args:
            odds: Cuota decimal (ej: 2.0).
            probability: Probabilidad estimada de ganar (0-1).
            method: Método para calcular el stake ('kelly', 'fixed', 'proportional').

        Returns:
            float: Monto a apostar.
        """
        if odds <= 1.0 or probability <= 0 or probability >= 1:
            self.logger.warning("Parámetros inválidos para cálculo de stake")
            return 0.0
        
        if method == 'kelly':
            b = odds - 1  # Beneficio por unidad apostada
            p = probability
            q = 1 - p
            
            if b <= 0:
                return 0.0
                
            # Fórmula corregida: f* = (bp - q) / b
            f_star = (b * p - q) / b
            
            # Limitar a riesgo máximo y mínimo
            max_fraction = min(self.risk_per_bet, 0.05)  # Máximo 5% por apuesta
            f_star = max(0, min(f_star, max_fraction))
            
            return f_star * self.current_bankroll

        elif method == 'fixed':
            return self.risk_per_bet * self.current_bankroll

        elif method == 'proportional':
            # Ajustar stake según la ventaja (odds * probability - 1)
            edge = odds * probability - 1
            if edge <= 0:
                return 0.0
            return min(edge * self.current_bankroll, self.risk_per_bet * self.current_bankroll)

        else:
            raise ValueError(f"Unknown stake method: {method}")

    def place_bet(
        self,
        match: str,
        prediction: str,
        odds: float,
        probability: float,
        stake_method: str = 'kelly'
    ) -> Dict[str, float]:
        """
        Registra una apuesta y actualiza el bankroll.

        Args:
            match: Nombre del partido (ej: "Barcelona vs Real Madrid").
            prediction: Resultado predicho ('home', 'draw', 'away').
            odds: Cuota decimal.
            probability: Probabilidad estimada.
            stake_method: Método para calcular el stake.

        Returns:
            Dict[str, float]: Información de la apuesta (stake, potential_profit, etc.).
        """
        try:
            stake = self.calculate_stake(odds, probability, stake_method)
            if stake <= 0:
                self.logger.warning(f"No bet placed for {match} (stake <= 0)")
                return {'stake': 0, 'potential_profit': 0}

            potential_profit = stake * (odds - 1)

            # Registrar apuesta
            bet_record = {
                'timestamp': datetime.now(),
                'match': match,
                'prediction': prediction,
                'odds': odds,
                'stake': stake,
                'potential_profit': potential_profit,
                'result': None,  # Se actualizará después del partido
                'profit': 0.0,
                'bankroll_after': self.current_bankroll - stake
            }

            self.bet_history = pd.concat([
                self.bet_history,
                pd.DataFrame([bet_record])
            ], ignore_index=True)

            # Actualizar bankroll
            self.current_bankroll -= stake

            self.logger.info(
                f"Bet placed: {match} | Prediction: {prediction} | "
                f"Stake: {stake:.2f} | Odds: {odds:.2f} | "
                f"Bankroll: {self.current_bankroll:.2f}"
            )

            return {
                'stake': stake,
                'potential_profit': potential_profit,
                'new_bankroll': self.current_bankroll
            }

        except Exception as e:
            self.logger.error(f"Error placing bet: {e}")
            return {'stake': 0, 'potential_profit': 0}

    def settle_bet(
        self,
        match: str,
        actual_result: str,
        bet_index: Optional[int] = None
    ) -> bool:
        """
        Resuelve una apuesta y actualiza el bankroll.

        Args:
            match: Nombre del partido.
            actual_result: Resultado real ('home', 'draw', 'away').
            bet_index: Índice de la apuesta en bet_history (opcional).

        Returns:
            bool: True si la apuesta se resolvió correctamente.
        """
        try:
            if bet_index is not None:
                bet = self.bet_history.iloc[bet_index]
            else:
                # Buscar la apuesta más reciente para este partido
                bet_matches = self.bet_history[self.bet_history['match'] == match]
                if bet_matches.empty:
                    self.logger.warning(f"No bet found for match: {match}")
                    return False
                bet = bet_matches.iloc[-1]

            prediction = bet['prediction']
            odds = bet['odds']
            stake = bet['stake']

            # Determinar si la apuesta ganó
            if prediction == actual_result:
                profit = stake * (odds - 1)
                result = 'win'
            else:
                profit = -stake
                result = 'loss'

            # Actualizar bankroll
            self.current_bankroll += profit

            # Actualizar registro de la apuesta
            self.bet_history.at[bet.name, 'result'] = result
            self.bet_history.at[bet.name, 'profit'] = profit
            self.bet_history.at[bet.name, 'bankroll_after'] = self.current_bankroll

            self.logger.info(
                f"Bet settled: {match} | Prediction: {prediction} | "
                f"Actual: {actual_result} | Profit: {profit:.2f} | "
                f"Bankroll: {self.current_bankroll:.2f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error settling bet: {e}")
            return False

    def get_bankroll_status(self) -> Dict[str, float]:
        """Obtiene el estado actual del bankroll."""
        total_bets = len(self.bet_history)
        winning_bets = len(self.bet_history[self.bet_history['result'] == 'win'])
        losing_bets = len(self.bet_history[self.bet_history['result'] == 'loss'])
        total_profit = self.bet_history['profit'].sum()
        win_rate = winning_bets / total_bets if total_bets > 0 else 0.0

        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'total_profit': total_profit,
            'return_on_investment': (total_profit / self.initial_bankroll) * 100,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'losing_bets': losing_bets,
            'win_rate': win_rate,
            'risk_per_bet': self.risk_per_bet
        }

    def get_bet_history(self) -> pd.DataFrame:
        """Obtiene el historial de apuestas."""
        return self.bet_history.copy()

    def set_risk_per_bet(self, risk: float) -> None:
        """Actualiza el porcentaje de riesgo por apuesta."""
        if 0 < risk <= 0.1:  # Máximo 10% por apuesta
            self.risk_per_bet = risk
            self.logger.info(f"Risk per bet updated to {risk*100:.1f}%")
        else:
            self.logger.warning(f"Invalid risk value: {risk}. Must be between 0 and 0.1.")

    def apply_stop_loss(self, stop_loss_pct: float = 0.2) -> bool:
        """
        Aplica un stop loss si el bankroll cae por debajo de un umbral.

        Args:
            stop_loss_pct: Porcentaje de pérdida máximo (ej: 0.2 = 20%).

        Returns:
            bool: True si se activó el stop loss.
        """
        loss_pct = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        if loss_pct >= stop_loss_pct:
            self.logger.warning(
                f"STOP LOSS TRIGGERED! Loss: {loss_pct*100:.1f}% | "
                f"Bankroll: {self.current_bankroll:.2f}"
            )
            return True
        return False

    def save_state(self, filepath: str) -> bool:
        """Guarda el estado del bankroll de forma segura."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            state = {
                'initial_bankroll': self.initial_bankroll,
                'current_bankroll': self.current_bankroll,
                'risk_per_bet': self.risk_per_bet,
                'bet_history': self.bet_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            self.logger.info(f"Bankroll state saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving bankroll state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """Carga el estado del bankroll de forma segura."""
        try:
            # Verificar integridad del archivo
            with open(filepath, 'rb') as f:
                data = f.read()
                if len(data) < 10:
                    return False
            
            # Cargar de forma segura
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Validar estructura de datos
            required_keys = ['initial_bankroll', 'current_bankroll', 'risk_per_bet']
            if not all(key in state for key in required_keys):
                self.logger.error("Archivo de estado corrupto")
                return False
                
            self.initial_bankroll = state['initial_bankroll']
            self.current_bankroll = state['current_bankroll']
            self.risk_per_bet = state['risk_per_bet']
            if 'bet_history' in state:
                self.bet_history = state['bet_history']
                
            self.logger.info(f"Estado cargado desde {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error cargando estado: {e}")
            return False
