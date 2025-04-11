# Cryptocurrency Analysis API

A Flask-based API that uses Google's Gemini to analyze cryptocurrency information across multiple categories.

## Features

- Analyze cryptocurrencies across 12 different categories
- Individual API endpoints for each category
- Parallel processing with multiple API keys for faster analysis
- JSON API responses for easy frontend integration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python multi_api_crypto_analyzer.py
```

## API Endpoints

The API will be available at `http://localhost:5000` with the following endpoints:

- `GET /api/analyze/<coin_symbol>` - Full analysis of all categories
- `GET /api/analyze/<coin_symbol>/feature_releases_or_updates` - Feature Releases or Updates analysis
- `GET /api/analyze/<coin_symbol>/whitepaper_updates` - Whitepaper Updates analysis
- `GET /api/analyze/<coin_symbol>/testnet_or_mainnet_milestones` - Testnet or Mainnet Milestones analysis
- `GET /api/analyze/<coin_symbol>/platform_integration` - Platform Integration analysis
- `GET /api/analyze/<coin_symbol>/team_changes` - Team Changes analysis
- `GET /api/analyze/<coin_symbol>/new_partnerships` - New Partnerships analysis
- `GET /api/analyze/<coin_symbol>/on-chain_activity` - On-chain Activity analysis
- `GET /api/analyze/<coin_symbol>/active_addresses_growth` - Active Addresses Growth analysis
- `GET /api/analyze/<coin_symbol>/real_world_adoption` - Real World Adoption analysis
- `GET /api/analyze/<coin_symbol>/developer_activity` - Developer Activity analysis
- `GET /api/analyze/<coin_symbol>/community_sentiment` - Community Sentiment analysis
- `GET /api/analyze/<coin_symbol>/liquidity_changes` - Liquidity Changes analysis

## Example Usage

```javascript
// Frontend JavaScript example
async function analyzeBitcoin() {
  // Get one specific category
  const response = await fetch('http://localhost:5000/api/analyze/BTC/developer_activity');
  const data = await response.json();
  console.log(data);
  
  // Or get all categories
  const fullResponse = await fetch('http://localhost:5000/api/analyze/BTC');
  const fullData = await fullResponse.json();
  console.log(fullData);
}
```

## Response Format

Individual category endpoint response:
```json
{
  "category": "Developer Activity",
  "content": "Detailed information about the developer activity..."
}
```

Full analysis endpoint response:
```json
{
  "Feature Releases or Updates": "Content about feature releases...",
  "Whitepaper Updates": "Content about whitepaper updates...",
  "Testnet or Mainnet Milestones": "Content about milestones...",
  ...
}
``` 