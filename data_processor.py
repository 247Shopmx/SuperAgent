import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime
import os

class DataProcessor:
    """Clase para procesar y limpiar datos de partidos y cuotas (ETL)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def clean_match_data(self, matches: List[Dict]) -> pd.DataFrame:
        """Limpia y transforma datos de partidos crudos desde ESPN."""
        if not matches:
            self.logger.warning("No matches data provided")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(matches)

            # Convertir fecha a datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                df['date'] = df['date'].fillna(pd.to_datetime('today'))

            # Procesar scores (ej: "2-1" -> home_goals=2, away_goals=1)
            def extract_goals(score_str: str) -> Tuple[float, float]:
                if pd.isna(score_str) or not str(score_str).strip():
                    return (0.0, 0.0)
                if "-" in str(score_str):
                    parts = str(score_str).split("-")
                    home = float(parts[0]) if parts[0].replace('.', '').isdigit() else 0.0
                    away = float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '').isdigit() else 0.0
                    return (home, away)
                return (float(score_str) if str(score_str).isdigit() else 0.0, 0.0)

            if 'home_score' in df.columns:
                df[['home_goals', 'temp_away']] = df['home_score'].apply(
                    lambda x: pd.Series(extract_goals(x))
                )
            else:
                df['home_goals'] = 0.0
                df['temp_away'] = 0.0

            if 'away_score' in df.columns:
                df[['temp_home', 'away_goals']] = df['away_score'].apply(
                    lambda x: pd.Series(extract_goals(x))
                )
            else:
                df['away_goals'] = 0.0

            # Usar el valor más confiable para away_goals
            df['away_goals'] = np.where(
                df['away_goals'] > 0,
                df['away_goals'],
                df['temp_away']
            )
            df.drop(['temp_away', 'temp_home'], axis=1, inplace=True, errors='ignore')

            # Calcular goles totales y resultado
            df['total_goals'] = df['home_goals'] + df['away_goals']
            df['result'] = np.where(
                df['home_goals'] > df['away_goals'], 1,
                np.where(df['home_goals'] < df['away_goals'], -1, 0)
            )

            # Procesar estado del partido
            if 'status' in df.columns:
                df['is_finished'] = df['status'].str.contains('Final|FT|Completed', case=False, na=False)
                df['is_live'] = df['status'].str.contains('Live|In Progress|In-Play', case=False, na=False)
            else:
                df['is_finished'] = False
                df['is_live'] = False

            # Rellenar valores nulos
            df.fillna({
                'home_goals': 0, 'away_goals': 0, 'total_goals': 0,
                'result': 0, 'league': 'Unknown', 'status': 'Scheduled', 'time': 'TBD'
            }, inplace=True)

            # Convertir tipos de datos
            df = df.astype({
                'home_goals': 'float32', 'away_goals': 'float32',
                'total_goals': 'float32', 'result': 'int8'
            })

            self.logger.info(f"Cleaned match data. Shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error cleaning match data: {e}")
            return pd.DataFrame()

    def clean_odds_data(self, odds: List[Dict]) -> pd.DataFrame:
        """Limpia y transforma datos de cuotas crudos desde Odds API."""
        if not odds:
            self.logger.warning("No odds data provided")
            return pd.DataFrame()

        try:
            processed_odds = []
            for game in odds:
                game_id = game.get('id')
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                commence_time = game.get('commence_time')

                for bookmaker in game.get('bookmakers', []):
                    bookmaker_name = bookmaker.get('key')
                    bookmaker_title = bookmaker.get('title')

                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')
                        for outcome in market.get('outcomes', []):
                            outcome_name = outcome.get('name')
                            price = outcome.get('price')
                            point = outcome.get('point')

                            processed_odds.append({
                                'game_id': game_id,
                                'home_team': home_team,
                                'away_team': away_team,
                                'commence_time': commence_time,
                                'bookmaker': bookmaker_name,
                                'bookmaker_title': bookmaker_title,
                                'market': market_key,
                                'outcome': outcome_name,
                                'odds': float(price) if price else np.nan,
                                'point': float(point) if point else np.nan
                            })

            df = pd.DataFrame(processed_odds)

            # Procesar commence_time a datetime
            if 'commence_time' in df.columns:
                df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')

            # Filtrar mercados relevantes (h2h, spreads, totals)
            relevant_markets = ['h2h', 'spreads', 'totals']
            df = df[df['market'].isin(relevant_markets)]

            # Pivotear para tener cuotas por mercado y resultado
            pivot_df = df.pivot_table(
                index=['game_id', 'home_team', 'away_team', 'commence_time'],
                columns=['market', 'outcome'],
                values='odds',
                aggfunc='first'
            ).reset_index()

            # Aplanar columnas multiíndice
            pivot_df.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0]
                for col in pivot_df.columns
            ]

            self.logger.info(f"Cleaned odds data. Shape: {pivot_df.shape}")
            return pivot_df

        except Exception as e:
            self.logger.error(f"Error cleaning odds data: {e}")
            return pd.DataFrame()

    def merge_match_and_odds(
        self,
        matches_df: pd.DataFrame,
        odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Fusiona datos de partidos y cuotas en un solo DataFrame."""
        if matches_df.empty or odds_df.empty:
            self.logger.warning("Empty DataFrame provided for merging")
            return pd.DataFrame()

        try:
            # Crear clave de fusión (home_team + away_team)
            matches_df['match_key'] = (
                matches_df['home_team'].str.lower() + "_" +
                matches_df['away_team'].str.lower()
            )
            odds_df['match_key'] = (
                odds_df['home_team'].str.lower() + "_" +
                odds_df['away_team'].str.lower()
            )

            # Fusionar por match_key y fecha aproximada
            merged_df = pd.merge(
                matches_df,
                odds_df,
                on='match_key',
                how='left',
                suffixes=('_match', '_odds')
            )

            # Eliminar duplicados
            merged_df.drop_duplicates(
                subset=['home_team', 'away_team', 'date', 'commence_time'],
                keep='first',
                inplace=True
            )

            self.logger.info(f"Merged data. Shape: {merged_df.shape}")
            return merged_df

        except Exception as e:
            self.logger.error(f"Error merging match and odds data: {e}")
            return pd.DataFrame()

    def calculate_team_stats(
        self,
        historical_matches: pd.DataFrame,
        window: int = 5
    ) -> pd.DataFrame:
        """Calcula estadísticas de equipos (promedios móviles de goles, etc.)."""
        if historical_matches.empty:
            return pd.DataFrame()

        try:
            # Crear DataFrame para estadísticas de equipos
            team_stats = []

            # Obtener equipos únicos
            all_teams = pd.concat([
                historical_matches['home_team'],
                historical_matches['away_team']
            ]).unique()

            for team in all_teams:
                # Filtrar partidos donde el equipo jugó (como local o visitante)
                team_matches = historical_matches[
                    (historical_matches['home_team'] == team) |
                    (historical_matches['away_team'] == team)
                ].copy()

                if len(team_matches) < window:
                    continue

                # Ordenar por fecha
                team_matches = team_matches.sort_values('date')

                # Calcular goles como local
                home_matches = team_matches[team_matches['home_team'] == team]
                if not home_matches.empty:
                    home_goals_for = home_matches['home_goals'].rolling(window=window).mean().iloc[-1]
                    home_goals_against = home_matches['away_goals'].rolling(window=window).mean().iloc[-1]
                else:
                    home_goals_for = np.nan
                    home_goals_against = np.nan

                # Calcular goles como visitante
                away_matches = team_matches[team_matches['away_team'] == team]
                if not away_matches.empty:
                    away_goals_for = away_matches['away_goals'].rolling(window=window).mean().iloc[-1]
                    away_goals_against = away_matches['home_goals'].rolling(window=window).mean().iloc[-1]
                else:
                    away_goals_for = np.nan
                    away_goals_against = np.nan

                # Calcular estadísticas generales
                total_goals_for = team_matches.apply(
                    lambda row: row['home_goals'] if row['home_team'] == team else row['away_goals'],
                    axis=1
                ).rolling(window=window).mean().iloc[-1]

                total_goals_against = team_matches.apply(
                    lambda row: row['away_goals'] if row['home_team'] == team else row['home_goals'],
                    axis=1
                ).rolling(window=window).mean().iloc[-1]

                # Calcular forma (últimos 5 resultados)
                results = team_matches.apply(
                    lambda row: 1 if (row['home_team'] == team and row['result'] == 1) or
                               (row['away_team'] == team and row['result'] == -1) else
                               (-1 if (row['home_team'] == team and row['result'] == -1) or
                                      (row['away_team'] == team and row['result'] == 1) else 0),
                    axis=1
                ).tail(window).tolist()

                form = sum(results) / len(results) if results else np.nan

                team_stats.append({
                    'team': team,
                    'home_goals_for_avg': home_goals_for,
                    'home_goals_against_avg': home_goals_against,
                    'away_goals_for_avg': away_goals_for,
                    'away_goals_against_avg': away_goals_against,
                    'total_goals_for_avg': total_goals_for,
                    'total_goals_against_avg': total_goals_against,
                    'form': form,
                    'matches_considered': len(team_matches)
                })

            stats_df = pd.DataFrame(team_stats)
            self.logger.info(f"Calculated team stats for {len(stats_df)} teams")
            return stats_df

        except Exception as e:
            self.logger.error(f"Error calculating team stats: {e}")
            return pd.DataFrame()

    def prepare_features(
        self,
        merged_df: pd.DataFrame,
        team_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepara características para los modelos de predicción."""
        if merged_df.empty or team_stats_df.empty:
            return pd.DataFrame()

        try:
            # Fusionar con estadísticas de equipos
            df = pd.merge(
                merged_df,
                team_stats_df,
                left_on='home_team',
                right_on='team',
                how='left',
                suffixes=('', '_home')
            )
            df = pd.merge(
                df,
                team_stats_df,
                left_on='away_team',
                right_on='team',
                how='left',
                suffixes=('', '_away')
            )

            # Calcular características derivadas
            df['home_attack_strength'] = df['home_goals_for_avg'] / df['home_goals_for_avg'].mean()
            df['home_defense_strength'] = df['home_goals_against_avg'] / df['home_goals_against_avg'].mean()
            df['away_attack_strength'] = df['away_goals_for_avg'] / df['away_goals_for_avg'].mean()
            df['away_defense_strength'] = df['away_goals_against_avg'] / df['away_goals_against_avg'].mean()

            # Calcular probabilidades implícitas de las cuotas (si existen)
            if 'h2h_home' in df.columns:
                df['implied_prob_home'] = 1 / df['h2h_home']
                df['implied_prob_away'] = 1 / df['h2h_away']
                df['implied_prob_draw'] = 1 / df['h2h_draw']

            # Seleccionar características relevantes
            features = [
                'home_attack_strength', 'home_defense_strength',
                'away_attack_strength', 'away_defense_strength',
                'home_form', 'away_form',
                'implied_prob_home', 'implied_prob_away', 'implied_prob_draw'
            ]

            # Asegurarse de que todas las características existan
            for feature in features:
                if feature not in df.columns:
                    df[feature] = np.nan

            # Eliminar columnas innecesarias
            df.drop([
                'team_home', 'team_away', 'match_key',
                'home_goals_for_avg', 'home_goals_against_avg',
                'away_goals_for_avg', 'away_goals_against_avg',
                'total_goals_for_avg_home', 'total_goals_against_avg_home',
                'total_goals_for_avg_away', 'total_goals_against_avg_away',
                'form_home', 'form_away', 'matches_considered_home', 'matches_considered_away'
            ], axis=1, inplace=True, errors='ignore')

            self.logger.info(f"Prepared features. Shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """Guarda un DataFrame en un archivo CSV."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved data to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            return False

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Carga un DataFrame desde un archivo CSV."""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded data from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading from CSV: {e}")
            return pd.DataFrame()
