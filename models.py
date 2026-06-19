import numpy as np
import pandas as pd
from scipy.stats import poisson, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Dict, List, Optional
import logging
import joblib
import os

class PoissonModel:
    """Modelo de Poisson para predecir goles esperados en un partido."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.home_goals_model = None
        self.away_goals_model = None
        self.scaler = StandardScaler()

    def fit(
        self,
        X: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
        features: List[str] = None
    ) -> None:
        """
        Entrena el modelo de Poisson.

        Args:
            X: DataFrame con características (ej: attack_strength, defense_strength).
            y_home: Serie con goles del equipo local.
            y_away: Serie con goles del equipo visitante.
            features: Lista de características a usar (si None, usa todas).
        """
        try:
            if features:
                X = X[features]

            # Escalar características
            X_scaled = self.scaler.fit_transform(X)

            # Ajustar modelos de Poisson para goles local y visitante
            # Usamos regresión lineal para predecir lambda (parámetro de Poisson)
            from sklearn.linear_model import PoissonRegressor

            self.home_goals_model = PoissonRegressor()
            self.home_goals_model.fit(X_scaled, y_home)

            self.away_goals_model = PoissonRegressor()
            self.away_goals_model.fit(X_scaled, y_away)

            self.logger.info("Poisson model trained successfully")

        except Exception as e:
            self.logger.error(f"Error training Poisson model: {e}")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice goles esperados para local y visitante.

        Args:
            X: DataFrame con características para predecir.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (goles_local_esperados, goles_visitante_esperados)
        """
        if self.home_goals_model is None or self.away_goals_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        try:
            X_scaled = self.scaler.transform(X)
            home_lambda = self.home_goals_model.predict(X_scaled)
            away_lambda = self.away_goals_model.predict(X_scaled)
            return home_lambda, away_lambda
        except Exception as e:
            self.logger.error(f"Error predicting with Poisson model: {e}")
            return np.zeros(len(X)), np.zeros(len(X))

    def predict_proba(
        self,
        X: pd.DataFrame,
        max_goals: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predice probabilidades de resultado (1, 0, -1) usando Poisson.

        Args:
            X: DataFrame con características.
            max_goals: Máximo número de goles a considerar.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (prob_home_win, prob_draw, prob_away_win)
        """
        home_lambda, away_lambda = self.predict(X)
        prob_home_win = np.zeros(len(X))
        prob_draw = np.zeros(len(X))
        prob_away_win = np.zeros(len(X))

        for i, (h_lambda, a_lambda) in enumerate(zip(home_lambda, away_lambda)):
            # Calcular probabilidades para todos los resultados posibles
            home_goals_proba = poisson.pmf(range(max_goals), h_lambda)
            away_goals_proba = poisson.pmf(range(max_goals), a_lambda)

            # Probabilidad de que el local gane (home_goals > away_goals)
            for home in range(max_goals):
                for away in range(max_goals):
                    if home > away:
                        prob_home_win[i] += home_goals_proba[home] * away_goals_proba[away]
                    elif home == away:
                        prob_draw[i] += home_goals_proba[home] * away_goals_proba[away]
                    else:
                        prob_away_win[i] += home_goals_proba[home] * away_goals_proba[away]

        return prob_home_win, prob_draw, prob_away_win

    def save(self, filepath: str) -> bool:
        """Guarda el modelo y el scaler en un archivo."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump({
                'home_model': self.home_goals_model,
                'away_model': self.away_goals_model,
                'scaler': self.scaler
            }, filepath)
            self.logger.info(f"Poisson model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving Poisson model: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Carga el modelo y el scaler desde un archivo."""
        try:
            data = joblib.load(filepath)
            self.home_goals_model = data['home_model']
            self.away_goals_model = data['away_model']
            self.scaler = data['scaler']
            self.logger.info(f"Poisson model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading Poisson model: {e}")
            return False

class XGBoostModel:
    """Modelo XGBoost para predecir el resultado del partido (1, 0, -1)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.classes_ = np.array([-1, 0, 1])  # Orden: away win, draw, home win

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str] = None,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Entrena el modelo XGBoost.

        Args:
            X: DataFrame con características.
            y: Serie con resultados (-1, 0, 1).
            features: Lista de características a usar.
            test_size: Proporción de datos para validación.

        Returns:
            Dict[str, float]: Métricas de evaluación (accuracy, log_loss).
        """
        try:
            if features:
                X = X[features]

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Entrenar modelo
            self.model.fit(X_train_scaled, y_train)

            # Evaluar
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)

            self.logger.info(
                f"XGBoost model trained. Accuracy: {accuracy:.4f}, Log Loss: {loss:.4f}"
            )

            return {'accuracy': accuracy, 'log_loss': loss}

        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {e}")
            return {'accuracy': 0.0, 'log_loss': float('inf')}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilidades para cada clase (-1, 0, 1).

        Args:
            X: DataFrame con características.

        Returns:
            np.ndarray: Matriz de probabilidades (n_samples, 3).
        """
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            self.logger.error(f"Error predicting with XGBoost model: {e}")
            return np.zeros((len(X), 3))

    def save(self, filepath: str) -> bool:
        """Guarda el modelo y el scaler en un archivo."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'classes': self.classes_
            }, filepath)
            self.logger.info(f"XGBoost model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Carga el modelo y el scaler desde un archivo."""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.classes_ = data['classes']
            self.logger.info(f"XGBoost model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {e}")
            return False

class LSTMModel:
    """Modelo LSTM para predecir tendencias temporales en resultados."""

    def __init__(self, input_shape: Tuple[int, int] = (5, 10)):
        self.logger = logging.getLogger(__name__)
        self.model = self._build_model(input_shape)
        self.scaler = StandardScaler()
        self.input_shape = input_shape

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Construye la arquitectura del modelo LSTM."""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 clases: -1, 0, 1
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara secuencias para el modelo LSTM.

        Args:
            X: DataFrame con características.
            y: Serie con resultados (-1, 0, 1).
            window_size: Tamaño de la ventana temporal.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_sequences, y_sequences)
        """
        X_scaled = self.scaler.fit_transform(X)
        X_sequences = []
        y_sequences = []

        for i in range(len(X_scaled) - window_size):
            X_sequences.append(X_scaled[i:i+window_size])
            y_sequences.append(y.iloc[i+window_size])

        return np.array(X_sequences), np.array(y_sequences)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str] = None,
        window_size: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Entrena el modelo LSTM.

        Args:
            X: DataFrame con características.
            y: Serie con resultados (-1, 0, 1).
            features: Lista de características a usar.
            window_size: Tamaño de la ventana temporal.
            epochs: Número de épocas.
            batch_size: Tamaño del batch.
            validation_split: Proporción de datos para validación.

        Returns:
            Dict[str, float]: Métricas de evaluación.
        """
        try:
            if features:
                X = X[features]

            # Preparar secuencias
            X_seq, y_seq = self.prepare_sequences(X, y, window_size)

            # Entrenar modelo
            history = self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0
            )

            # Obtener métricas
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]

            self.logger.info(
                f"LSTM model trained. Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

            return {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }

        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            return {
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'train_accuracy': 0.0,
                'val_accuracy': 0.0
            }

    def predict_proba(self, X: pd.DataFrame, window_size: int = 5) -> np.ndarray:
        """
        Predice probabilidades para cada clase (-1, 0, 1).

        Args:
            X: DataFrame con características.
            window_size: Tamaño de la ventana temporal.

        Returns:
            np.ndarray: Matriz de probabilidades (n_samples, 3).
        """
        try:
            X_scaled = self.scaler.transform(X)
            X_seq = []

            # Crear secuencias para predicción
            for i in range(len(X_scaled) - window_size + 1):
                X_seq.append(X_scaled[i:i+window_size])

            if not X_seq:
                return np.zeros((0, 3))

            X_seq = np.array(X_seq)
            return self.model.predict(X_seq)
        except Exception as e:
            self.logger.error(f"Error predicting with LSTM model: {e}")
            return np.zeros((len(X), 3))

    def save(self, filepath: str) -> bool:
        """Guarda el modelo y el scaler en un archivo."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath + '_model.h5')
            joblib.dump({
                'scaler': self.scaler,
                'input_shape': self.input_shape
            }, filepath + '_scaler.pkl')
            self.logger.info(f"LSTM model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Carga el modelo y el scaler desde un archivo."""
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(filepath + '_model.h5')
            data = joblib.load(filepath + '_scaler.pkl')
            self.scaler = data['scaler']
            self.input_shape = data['input_shape']
            self.logger.info(f"LSTM model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
            return False

class ModelEnsemble:
    """Combina predicciones de Poisson, XGBoost y LSTM."""

    def __init__(
        self,
        poisson_model: PoissonModel,
        xgboost_model: XGBoostModel,
        lstm_model: LSTMModel,
        weights: Dict[str, float] = None
    ):
        """
        Inicializa el ensemble con los modelos y pesos.

        Args:
            poisson_model: Modelo de Poisson.
            xgboost_model: Modelo XGBoost.
            lstm_model: Modelo LSTM.
            weights: Pesos para cada modelo (ej: {'poisson': 0.4, 'xgboost': 0.4, 'lstm': 0.2}).
        """
        self.poisson_model = poisson_model
        self.xgboost_model = xgboost_model
        self.lstm_model = lstm_model
        self.weights = weights or {'poisson': 0.3, 'xgboost': 0.4, 'lstm': 0.3}
        self.logger = logging.getLogger(__name__)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilidades combinadas de los tres modelos.

        Args:
            X: DataFrame con características.

        Returns:
            np.ndarray: Matriz de probabilidades (n_samples, 3) para [-1, 0, 1].
        """
        try:
            # Obtener predicciones de cada modelo
            poisson_proba = self.poisson_model.predict_proba(X)
            xgboost_proba = self.xgboost_model.predict_proba(X)

            # Para LSTM, necesitamos secuencias. Si X tiene solo una fila, no podemos usar LSTM.
            if len(X) >= self.lstm_model.input_shape[0]:
                lstm_proba = self.lstm_model.predict_proba(X)
                # Ajustar forma si es necesario
                if len(lstm_proba.shape) == 3:
                    lstm_proba = lstm_proba[-1]  # Tomar la última predicción
            else:
                lstm_proba = np.ones((len(X), 3)) / 3  # Probabilidades uniformes

            # Combinar predicciones con pesos
            combined_proba = (
                self.weights['poisson'] * np.column_stack(poisson_proba) +
                self.weights['xgboost'] * xgboost_proba +
                self.weights['lstm'] * lstm_proba
            )

            # Normalizar para que sumen 1
            combined_proba = combined_proba / combined_proba.sum(axis=1, keepdims=True)

            return combined_proba

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return np.ones((len(X), 3)) / 3  # Probabilidades uniformes

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice la clase más probable (-1, 0, 1)."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) - 1  # Convertir [0,1,2] a [-1,0,1]

    def get_expected_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Obtiene goles esperados del modelo Poisson."""
        return self.poisson_model.predict(X)

    def save(self, directory: str) -> bool:
        """Guarda todos los modelos en un directorio."""
        try:
            os.makedirs(directory, exist_ok=True)
            self.poisson_model.save(os.path.join(directory, 'poisson_model.pkl'))
            self.xgboost_model.save(os.path.join(directory, 'xgboost_model.pkl'))
            self.lstm_model.save(os.path.join(directory, 'lstm_model'))
            joblib.dump(self.weights, os.path.join(directory, 'weights.pkl'))
            self.logger.info(f"Ensemble models saved to {directory}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving ensemble models: {e}")
            return False

    def load(self, directory: str) -> bool:
        """Carga todos los modelos desde un directorio."""
        try:
            self.poisson_model.load(os.path.join(directory, 'poisson_model.pkl'))
            self.xgboost_model.load(os.path.join(directory, 'xgboost_model.pkl'))
            self.lstm_model.load(os.path.join(directory, 'lstm_model'))
            self.weights = joblib.load(os.path.join(directory, 'weights.pkl'))
            self.logger.info(f"Ensemble models loaded from {directory}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ensemble models: {e}")
            return False
