from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime
from pydantic import BaseModel, Field

app = FastAPI(
    title="Combined Market Analysis API",
    description="Comprehensive cryptocurrency market analysis and metrics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SourceModel(BaseModel):
    channel: str
    processed_at: str
    published_at: str
    video_title: str
    video_url: str

class ReasonModel(BaseModel):
    coin: str
    reason: str
    sentiment: str
    source: SourceModel
    
class PredictionModel(BaseModel):
    direction: str
    reasoning: List[str]

    
class MarketAnalysisResponse(BaseModel):

    
    confidence: int
    lastUpdated: str
    market: str
    id: str
    symbol: str
    image: str = Field(description="URL to the coin's thumbnail image")
    prediction: PredictionModel
    current_price: float
    price_change_24h: float
    sentiment: str
    sentiment_score: int
    status: str
    target_price: float
    timeline: str
    
    
class MarketDataProcessor:
    def __init__(self):
        self.predictions_url = "https://api.example.com/predictions"  # Replace with actual URL
        self.metrics_url = "https://api.example.com/metrics"  # Replace with actual URL
        self.client = httpx.AsyncClient(timeout=30.0)

    async def process_combined_data(self, prediction_data: Dict[str, Any], metrics_data: Dict[str, Any]) -> MarketAnalysisResponse:
        return MarketAnalysisResponse(
            # Market Analysis Data
            date=prediction_data['data']['date'],
            prediction=prediction_data['data'],
            
            # Price Metrics
            bearish=metrics_data['bearish'],
            bullish=metrics_data['bullish'],
            coin=metrics_data['coin'],
            currentPrice=metrics_data['currentPrice'],
            price24h=metrics_data['price24h'],
            price7d=metrics_data['price7d'],
            price30d=metrics_data['price30d'],
            dominance=metrics_data['dominance'],
            bullPercentage=metrics_data['bullPercentage'],
            bearPercentage=metrics_data['bearPercentage'],
            
        )


class MarketDataProcessor:
    def __init__(self):
        self.moometrics_url = "https://api.moometrics.io/news/market-analysis/get"
        self.heifereum_base_url = "https://api.heifereum.com/api"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_market_analysis(self) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get(self.moometrics_url)
            response.raise_for_status()
            return response.json().get('data', [])
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Error fetching market analysis: {str(e)}")

    async def search_cryptocurrency(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            response = await self.client.get(
                f"{self.heifereum_base_url}/cryptocurrency/search",
                params={"query": symbol}
            )
            response.raise_for_status()
            data = response.json().get('coins', [])
            return data[0] if data else None
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Error searching cryptocurrency: {str(e)}")

    async def get_crypto_info(self, coin_id: str) -> Dict[str, Any]:
        try:
            response = await self.client.get(
                f"{self.heifereum_base_url}/cryptocurrency/info",
                params={"id": coin_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Error fetching crypto info: {str(e)}")

    async def process_combined_data(self, analysis: Dict[str, Any]) -> MarketAnalysisResponse:
        coin_details = await self.search_cryptocurrency(analysis['symbol'])
        if not coin_details:
            raise HTTPException(status_code=404, detail=f"Coin not found: {analysis['symbol']}")

        coin_info = await self.get_crypto_info(coin_details['id'])
        market_data = coin_info.get('market_data', {})

        return MarketAnalysisResponse(
            confidence=analysis['confidence'],
            lastUpdated=analysis['lastUpdated'],
            market=analysis['market'],
            id=coin_details['id'],
            symbol=analysis['symbol'],
            prediction=analysis['prediction'],
            current_price=str(market_data.get('current_price', {}).get('usd', '0')),
            price_change_24h=str(market_data.get('price_change_percentage_24h', '0')),
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['sentimentScore'],
            status=analysis['status'],
            target_price=convert_target_price(analysis['targetPrice']),
            timeline=analysis['timeline'],
            image=coin_info.get('image', {}).get('thumb', '')
        )

processor = MarketDataProcessor()

@app.get("/api/v1/combined-analysis", response_model=List[MarketAnalysisResponse])
async def get_combined_analysis():
    """
    Fetch comprehensive market analysis including both analysis and price metrics
    """
    try:
        analyses = await processor.get_market_analysis()
        return [await processor.process_combined_data(analysis) for analysis in analyses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/combined-analysis/{symbol}", response_model=MarketAnalysisResponse)
async def get_single_combined_analysis(symbol: str):
    """
    Fetch comprehensive market analysis for a specific cryptocurrency
    """
    try:
        analyses = await processor.get_market_analysis()
        analysis = next((a for a in analyses if a['symbol'].upper() == symbol.upper()), None)
        
        if not analysis:
            raise HTTPException(status_code=404, detail=f"No analysis found for {symbol}")
            
        return await processor.process_combined_data(analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
 
    
def convert_target_price(target_price: str) -> float:
    return float(target_price.replace('$', '').replace(',', ''))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)