from datetime import datetime
import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class PriceData:
    date: str
    price: float

class HistoricalPriceService:
    def __init__(self):
        self.search_base_url = "https://api.heifereum.com/api/cryptocurrency/search"
        self.density_base_url = "https://api.heifereum.com/api/density-flow"
        self.start_date = datetime(2024, 12, 1).isoformat()

    def search_coin(self, query: str) -> Optional[Dict[str, str]]:
        """Search for a coin and return its ID and symbol"""
        try:
            response = requests.get(f"{self.search_base_url}?query={query}")
            data = response.json()

            if not data.get('coins') or len(data['coins']) == 0:
                return None

            coin = data['coins'][0]
            return {
                'id': coin['id'],
                'symbol': coin['symbol']
            }
        except Exception as e:
            print(f"Error searching coin: {str(e)}")
            return None

    def get_price_data(self, coin_id: str) -> List[PriceData]:
        """Get historical price data for a coin"""
        try:
            response = requests.get(f"{self.density_base_url}/{coin_id}")
            data = response.json()

            if not data.get('data'):
                return []

            return [
                PriceData(date=item['date'], price=item['price'])
                for item in data['data']
                if item['date'] >= self.start_date
            ]
        except Exception as e:
            print(f"Error getting price data: {str(e)}")
            return []

    def get_historical_prices(self, coin: str) -> Dict[str, Union[str, List[PriceData]]]:
        """Main method to get historical prices for a coin"""
        try:
            # Search for coin
            coin_info = self.search_coin(coin)
            if not coin_info:
                raise ValueError(f"No coin found for query: {coin}")

            # Get price data
            prices = self.get_price_data(coin_info['id'])
            if not prices:
                raise ValueError(f"No price data found for coin: {coin}")

            return {
                "symbol": coin_info['symbol'],
                "prices": prices
            }
        except Exception as e:
            raise Exception(f"Error getting historical prices: {str(e)}")
