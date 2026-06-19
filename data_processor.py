import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class DataProcessor:
    """Clase para procesar y limpiar datos de partidos y cuotas (ETL)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def clean_match_data(self, matches: List[Dict]) -> pd.DataFrame:
        """Limpia y transforma datos de partidos crudos."""
        if not matches:
            self.logger.warning("No matches data provided")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(matches)

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                df['date'] = df['date'].fillna(pd.to_datetime('today'))

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

            df['away_goals'] = np.where(df['away_goals'] > 0, df['away_goals'], df['temp_away'])
            df.drop(['temp_away', 'temp_home'], axis=1, inplace=True, errors='ignore')

            # Generación segura de variables métricas
            df['total_goals'] = df['home_goals'] + df['away_goals']
            df['result'] = np.where(
                df['home_goals'] > df['away_goals'], 1,
                np.where(df['home_goals'] < df['away_goals'], -1, 0)
            )

            if 'status' in df.columns:
                df['is_finished'] = df['status'].str.contains('Final|FT|Completed', case=False, na=False)
                df['is_live'] = df['status'].str.contains('Live|In Progress|In-Play', case=False, na=False)
            else:
                df['is_finished'] = False
                df['is_live'] = False

            df.fillna({
                'home_goals': 0, 'away_goals': 0, 'total_goals': 0,
                'result': 0, 'league': 'Unknown', 'status': 'Scheduled', 'time': 'TBD'
            }, inplace=True)

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
        """Limpia y transforma datos de cuotas crudos."""
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
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')
                        for outcome in market.get('outcomes', []):
                            processed_odds.append({
                                'game_id': game_id,
                                'home_team': home_team,
                                'away_team': away_team,
                                'commence_time': commence_time,
                                'bookmaker': bookmaker_name,
                                'market': market_key,
                                'outcome': outcome.get('name'),
                                'odds': float(outcome.get('price')) if outcome.get('price') else np.nan
                            })

            df = pd.DataFrame(processed_odds)
            if 'commence_time' in df.columns:
                df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')

            df = df[df['market'].isin(['h2h', 'spreads', 'totals'])]

            pivot_df = df.pivot_table(
                index=['game_id', 'home_team', 'away_team', 'commence_time'],
                columns=['market', 'outcome'],
                values='odds',
                aggfunc='first'
            ).reset_index()

            pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_df.columns]
            self.logger.info(f"Cleaned odds data. Shape: {pivot_df.shape}")
            return pivot_df

        except Exception as e:
            self.logger.error(f"Error cleaning odds data: {e}")
            return pd.DataFrame()

    def merge_match_and_odds(self, matches_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Fusiona datos de partidos y cuotas en un solo DataFrame de manera segura."""
        if matches_df.empty:
            self.logger.warning("Matches DataFrame is empty. Cannot merge.")
            return pd.DataFrame()
        if odds_df.empty:
            self.logger.warning("Odds DataFrame is empty. Returning matches only.")
            return matches_df

        try:
            matches_df = matches_df.copy()
            odds_df = odds_df.copy()

            matches_df['match_key'] = (
                matches_df['home_team'].astype(str).str.lower().str.strip() + "_" +
                matches_df['away_team'].astype(str).str.lower().str.strip()
            )
            odds_df['match_key'] = (
                odds_df['home_team'].astype(str).str.lower().str.strip() + "_" +
                odds_df['away_team'].astype(str).str.lower().str.strip()
            )

            # Dropear columnas redundantes de cuotas para no ensuciar el merge
            odds_clean = odds_df.drop(['home_team', 'away_team'], axis=1, errors='ignore')

            merged_df = pd.merge(matches_df, odds_clean, on='match_key', how='left')

            if 'date' in merged_df.columns:
                merged_df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first', inplace=True)
            else:
                merged_df.drop_duplicates(subset=['home_team', 'away_team'], keep='first', inplace=True)

            merged_df.drop(['match_key'], axis=1, inplace=True, errors='ignore')
            
            self.logger.info(f"Merged data successfully. Shape: {merged_df.shape}")
            return merged_df

        except Exception as e:
            self.logger.error(f"Error merging match and odds data: {e}")
            return matches_df

    def calculate_team_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calcula estadísticas históricas por equipo de forma segura.
        
        Args:
            df: DataFrame con datos históricos de partidos
            window: Ventana de partidos recientes a considerar
            
        Returns:
            DataFrame con estadísticas por equipo
        """
        if df.empty or 'result' not in df.columns or 'home_team' not in df.columns:
            self.logger.error("DataFrame inválido o faltan columnas requeridas")
            return pd.DataFrame()
        
        try:
            stats_list = []
            
            for team in pd.concat([df['home_team'], df['away_team']]).unique():
                team_matches = df[
                    (df['home_team'] == team) | (df['away_team'] == team)
                ].tail(window)
                
                if len(team_matches) < 2:
                    continue
                    
                home_matches = team_matches[team_matches['home_team'] == team]
                away_matches = team_matches[team_matches['away_team'] == team]
                
                stats = {
                    'team': team,
                    'matches_played': len(team_matches),
                    'home_attack_strength': home_matches['home_goals'].mean() if not home_matches.empty else 0,
                    'home_defense_strength': home_matches['away_goals'].mean() if not home_matches.empty else 0,
                    'away_attack_strength': away_matches['away_goals'].mean() if not away_matches.empty else 0,
                    'away_defense_strength': away_matches['home_goals'].mean() if not away_matches.empty else 0,
                    'home_form': (home_matches['result'] == 1).mean() if not home_matches.empty else 0.5,
                    'away_form': (away_matches['result'] == -1).mean() if not away_matches.empty else 0.5,
                }
                stats_list.append(stats)
            
            stats_df = pd.DataFrame(stats_list)
            self.logger.info(f"Estadísticas calculadas para {len(stats_df)} equipos")
            return stats_df
            
        except Exception as e:
            self.logger.error(f"Error calculando estadísticas: {e}")
            return pd.DataFrame()

    def prepare_features(self, matches_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara características para el entrenamiento de modelos.
        
        Args:
            matches_df: DataFrame con datos de partidos
            team_stats_df: DataFrame con estadísticas de equipos
            
        Returns:
            DataFrame con características preparadas
        """
        if matches_df.empty:
            self.logger.warning("No match data to prepare features")
            return pd.DataFrame()
        
        try:
            features_df = matches_df.copy()
            
            # Unir estadísticas de equipos
            if not team_stats_df.empty:
                # Estadísticas del equipo local
                features_df = features_df.merge(
                    team_stats_df.add_prefix('home_'),
                    left_on='home_team',
                    right_on='home_team',
                    how='left'
                )
                
                # Estadísticas del equipo visitante
                features_df = features_df.merge(
                    team_stats_df.add_prefix('away_'),
                    left_on='away_team',
                    right_on='away_team',
                    how='left'
                )
            
            # Rellenar valores nulos con defaults
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
            
            self.logger.info(f"Features prepared. Shape: {features_df.shape}")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """Guarda DataFrame en archivo CSV."""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            return False

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Carga DataFrame desde archivo CSV."""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Data loaded from {filepath}. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading from CSV: {e}")
            return pd.DataFrame()
