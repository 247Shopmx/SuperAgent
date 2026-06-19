import os
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
import time
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class ESPNScraper:
    """
    Clase para obtener datos históricos y en vivo de partidos de fútbol desde ESPN.
    Implementa scraping de la página de resultados y partidos.
    """

    def __init__(self):
        """Inicializa el scraper con headers y configuración básica."""
        self.base_url = "https://www.espn.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Configurar handler para logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Obtiene el objeto BeautifulSoup para una URL dada.

        Args:
            url (str): URL a la que se hará el request.

        Returns:
            Optional[BeautifulSoup]: Objeto BeautifulSoup o None si falla.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            self.logger.error(f"Error fetching URL {url}: {e}")
            return None

    def fetch_scoreboard(self, date: Optional[str] = None) -> List[Dict]:
        """
        Obtiene los partidos del día (o fecha específica) desde ESPN.

        Args:
            date (str, optional): Fecha en formato YYYYMMDD. Si es None, usa la fecha actual.

        Returns:
            List[Dict]: Lista de diccionarios con información de los partidos.
        """
        if date:
            url = f"{self.base_url}/soccer/scoreboard/_/date/{date}"
        else:
            url = f"{self.base_url}/soccer/scoreboard"

        soup = self._get_soup(url)
        if not soup:
            return []

        matches = []
        # Selector para los contenedores de partidos en ESPN
        match_containers = soup.select('div[class*="MatchupRow"]')

        for container in match_containers:
            try:
                # Extraer información básica del partido
                home_team = container.select_one('div[class*="home"] a').text.strip()
                away_team = container.select_one('div[class*="away"] a').text.strip()

                # Extraer scores (pueden no estar disponibles para partidos futuros)
                home_score = container.select_one('div[class*="home"] .score')
                home_score = home_score.text.strip() if home_score else "0"
                away_score = container.select_one('div[class*="away"] .score')
                away_score = away_score.text.strip() if away_score else "0"

                # Extraer estado del partido (en vivo, finalizado, etc.)
                status = container.select_one('div[class*="status"]')
                status = status.text.strip() if status else "Scheduled"

                # Extraer liga
                league = container.select_one('div[class*="competition"] a')
                league = league.text.strip() if league else "Unknown"

                # Extraer hora
                time_element = container.select_one('div[class*="time"]')
                match_time = time_element.text.strip() if time_element else "TBD"

                matches.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': status,
                    'league': league,
                    'time': match_time,
                    'date': date if date else time.strftime("%Y%m%d"),
                    'source': 'ESPN'
                })
            except Exception as e:
                self.logger.warning(f"Error parsing match container: {e}")
                continue

        self.logger.info(f"Fetched {len(matches)} matches from ESPN")
        return matches

    def fetch_team_stats(self, team_name: str, season: str = "2023") -> List[Dict]:
        """
        Obtiene estadísticas de un equipo para una temporada específica.

        Args:
            team_name (str): Nombre del equipo.
            season (str): Temporada (ej: "2023").

        Returns:
            List[Dict]: Lista de estadísticas del equipo.
        """
        # Formatear el nombre del equipo para la URL
        team_slug = team_name.lower().replace(" ", "-").replace(".", "")
        url = f"{self.base_url}/soccer/team/stats/_/id/{team_slug}/season/{season}"

        soup = self._get_soup(url)
        if not soup:
            return []

        stats = []

        try:
            # Extraer estadísticas generales (esto puede variar según la estructura de ESPN)
            stat_sections = soup.select('div[class*="StatSection"]')

            for section in stat_sections:
                section_title = section.select_one('h2')
                if not section_title:
                    continue

                title = section_title.text.strip()
                rows = section.select('tr')

                for row in rows:
                    cells = row.select('td')
                    if len(cells) >= 2:
                        stat_name = cells[0].text.strip()
                        stat_value = cells[1].text.strip()
                        stats.append({
                            'team': team_name,
                            'season': season,
                            'category': title,
                            'stat_name': stat_name,
                            'stat_value': stat_value
                        })

        except Exception as e:
            self.logger.error(f"Error parsing team stats for {team_name}: {e}")

        return stats

    def fetch_historical_matches(self, team_name: str, season: str = "2023") -> List[Dict]:
        """
        Obtiene partidos históricos de un equipo en una temporada.

        Args:
            team_name (str): Nombre del equipo.
            season (str): Temporada (ej: "2023").

        Returns:
            List[Dict]: Lista de partidos históricos.
        """
        team_slug = team_name.lower().replace(" ", "-").replace(".", "")
        url = f"{self.base_url}/soccer/team/schedule/_/id/{team_slug}/season/{season}"

        soup = self._get_soup(url)
        if not soup:
            return []

        matches = []

        try:
            # Buscar la tabla de partidos
            table = soup.select_one('table[class*="schedule"]')
            if not table:
                self.logger.warning(f"No schedule table found for {team_name}")
                return matches

            rows = table.select('tr[class*="row"]')

            for row in rows:
                try:
                    date = row.select_one('td[class*="date"]')
                    date = date.text.strip() if date else "Unknown"

                    opponent = row.select_one('td[class*="opponent"] a')
                    opponent = opponent.text.strip() if opponent else "Unknown"

                    result = row.select_one('td[class*="result"]')
                    result = result.text.strip() if result else "Unknown"

                    home_away = row.select_one('td[class*="home-away"]')
                    home_away = home_away.text.strip() if home_away else "Unknown"

                    score = row.select_one('td[class*="score"]')
                    score = score.text.strip() if score else "0-0"

                    venue = row.select_one('td[class*="venue"]')
                    venue = venue.text.strip() if venue else "Unknown"

                    matches.append({
                        'team': team_name,
                        'opponent': opponent,
                        'date': date,
                        'result': result,
                        'home_away': home_away,
                        'score': score,
                        'venue': venue,
                        'season': season
                    })
                except Exception as e:
                    self.logger.warning(f"Error parsing historical match row: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error fetching historical matches for {team_name}: {e}")

        return matches

class OddsAPIClient:
    """
    Clase para interactuar con la API de Odds API.
    Obtiene cuotas en tiempo real para partidos de fútbol.
    """

    def __init__(self):
        """Inicializa el cliente con la clave API desde variables de entorno."""
        self.api_key = os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY environment variable not set")

        self.base_url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def get_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso"
    ) -> List[Dict]:
        """
        Obtiene cuotas para partidos de fútbol.

        Args:
            regions (str): Regiones para las cuotas (ej: "us", "eu", "uk").
            markets (str): Mercados de apuestas (ej: "h2h", "spreads", "totals").
            date_format (str): Formato de fecha ("iso" o "unix").

        Returns:
            List[Dict]: Lista de diccionarios con información de cuotas.
        """
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': markets,
            'dateFormat': date_format
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get('success') is not True:
                self.logger.error(f"Odds API error: {data.get('message', 'Unknown error')}")
                return []

            return data.get('data', [])
        except requests.RequestException as e:
            self.logger.error(f"Error fetching odds from API: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing odds API response: {e}")
            return []

    def get_odds_for_match(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Obtiene cuotas específicas para un partido entre dos equipos.

        Args:
            home_team (str): Nombre del equipo local.
            away_team (str): Nombre del equipo visitante.

        Returns:
            Optional[Dict]: Diccionario con cuotas para el partido o None si no se encuentra.
        """
        odds = self.get_odds()
        for game in odds:
            if (game.get('home_team') == home_team and
                game.get('away_team') == away_team):
                return game
        return None

    def get_bookmaker_odds(self, game_id: str, bookmaker: str = "bet365") -> Optional[Dict]:
        """
        Obtiene cuotas de un bookmaker específico para un partido.

        Args:
            game_id (str): ID del partido.
            bookmaker (str): Nombre del bookmaker (ej: "bet365", "pinnacle").

        Returns:
            Optional[Dict]: Diccionario con cuotas del bookmaker o None.
        """
        odds = self.get_odds()
        for game in odds:
            if game.get('id') == game_id:
                for bm in game.get('bookmakers', []):
                    if bm.get('key') == bookmaker:
                        return bm
        return None
