from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import time
import traceback

class CryptoSearcher:
    """An enhanced class-based implementation of cryptocurrency information search with extended categories"""
    
    # API keys to use (rotating for load balancing)
    API_KEYS = [
        "AIzaSyAOogW_ZTPgDniIc0ecGSQk_4L9U_y7dno",
        "AIzaSyCw6ZIGyFjmPu5GiRgtYAFyfkvFmVFl5V4",
        "AIzaSyD0gGlc_8RmRDg2Cx8Ab9qBFMthjZCTrX4",
        "AIzaSyB-T6qexYo6sx2rV8wM9S4a6y5A1x8M360",
        "AIzaSyAnMabdfFsx32AzWN06oWtuad2qA7IMRJI",
        "AIzaSyAJkVH1OkkhIJIvkQ4_zj7MvbwOgcvJifA",
        "AIzaSyBsOiaKxYX4xy0C9PEZjgMHzn7aTIuHy9s",
        "AIzaSyA3sdIgptQbJaGN9l6WWFd35362tOFbrOQ",
        "AIzaSyCCBYJ6gIPYr_UX2QGs2eMWhA3BEJDE-N8",
        "AIzaSyD9xQAeuftfV9uRfnIUBxYrL3KQoyYe-uI",
        "AIzaSyAQJ3CwosJnp7x6XxhSZeORy5siwHs90yo",
        "AIzaSyCf9XCnLbAqnhhEj_jZB7H4OWkJKb44CCo",
        "AIzaSyDVxwX8TwdFQQaYQlAXpwhZt8WngYexsV4",
        "AIzaSyCcRfl83Qcih9-JfKQ0vLcDPe_vBLpMjBk",
        "AIzaSyBxG1dN2TqtmeH2OoCmJpugmLjyfAgjia0",
        "AIzaSyDg6vphZ4rGu87f4orzk1B56wY0es5DQes",
        "AIzaSyCbOrS-AndGjOfp7W63eKmUxrpOd-yeNes",
        "AIzaSyAsGm3XOeuBU1KHOLfTF0Jkku9UPLiuF08",
        "AIzaSyCDY-OsHBmeG2lSzvXDhctJxk8vt6NETlg",
        "AIzaSyADDrCwK4NC3W6oPbRP4-3Eyj2xiIhTSTk",
        "AIzaSyBHqCtWOCaI-cnPXqu0G59EC9a35mz8HXw",
        "AIzaSyBxeQKYExn_2Mu2wkg9ExfQr_yn7RiJ6Ow",
        "AIzaSyDL5ml4tRGn2c5uYhhiKo4SrC748mqYzXo",
        "AIzaSyACT6pP6BUL2sC8VCGrKP072JDN0lGSmLk",
        "AIzaSyBeTu1X-ajnmD-QsyZQ6WX79zKJN6i3An4",
        "AIzaSyAU_WsfM-IkYFmoJv63o3jvSwfIITTsxvk",
        "AIzaSyC3xj-AqJAmprINTAIHx_IBQXDx_QuhwRI",
        "AIzaSyBZ58pXnjvKqZwon_QhWhPvrnEzvr3ZDjU",
        "AIzaSyDNWfFmywWgydEI0NxL9xbCTjdlnYlOoKE",
        "AIzaSyAl4wyZrygCVOD6Y8bE2QCcS5KhY-djCYM",
        "AIzaSyAh-i4Yl3L0MeVn6WHSnrUKFdsreFK0x7I",

    ]
    
    # Gemini model ID
    MODEL_ID = "gemini-2.0-flash"
    
    def __init__(self, api_key_index=0):
        """Initialize with an API key index"""
        self.api_key_index = api_key_index % len(self.API_KEYS)
        self.client = genai.Client(api_key=self.API_KEYS[self.api_key_index])
    
    def _send_search_request(self, query, max_retries=3):
        """Send a search request to Gemini API"""
        print(f"Searching for information: {query[:100]}...")
        
        # Create the Google Search tool
        google_search_tool = Tool(
            google_search = GoogleSearch()
        )
        
        retries = 0
        while retries < max_retries:
            try:
                # Generate a response with search capability
                response = self.client.models.generate_content(
                    model=self.MODEL_ID,
                    contents=query,
                    config=GenerateContentConfig(
                        tools=[google_search_tool],
                        response_modalities=["TEXT"],
                        temperature=0.7,
                        max_output_tokens=8096,
                    )
                )
                
                # Extract text from response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        if hasattr(content, 'parts') and content.parts:
                            all_text = []
                            for part in content.parts:
                                if hasattr(part, 'text'):
                                    all_text.append(part.text)
                            
                            result_text = "\n".join(all_text).strip()
                            if result_text:
                                return result_text
                
                # If we get here, we didn't extract text successfully
                print(f"No usable content found, retrying ({retries+1}/{max_retries})")
                retries += 1
                time.sleep(2)  # Short delay before retry
                
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                traceback.print_exc()
                retries += 1
                time.sleep(5)  # Longer delay on exception
        
        return "Error: Could not retrieve information after multiple attempts."

    # Original category search functions
    def search_feature_releases(self, coin_symbol):
        """Search for feature releases or updates for a cryptocurrency"""
        prompt = f"Provide detailed information about recent feature releases or updates for {coin_symbol} cryptocurrency. Focus on specific technical improvements, new features, protocol changes, or roadmap updates. Be factual and detailed, citing specific sources."
        return self._send_search_request(prompt)
    
    def search_whitepaper_updates(self, coin_symbol):
        """Search for whitepaper updates for a cryptocurrency"""
        prompt = f"Find recent whitepaper updates or technical documentation changes for {coin_symbol} cryptocurrency. Focus on any modifications to the project's core design, tokenomics, consensus mechanisms, or technical architecture. Be specific and detailed."
        return self._send_search_request(prompt)
    
    def search_testnet_mainnet_milestones(self, coin_symbol):
        """Search for testnet or mainnet milestones for a cryptocurrency"""
        prompt = f"Provide detailed information about recent testnet or mainnet milestones for {coin_symbol} cryptocurrency. Include information about network upgrades, hard forks, testing phases, or mainnet launches. Be factual and cite specific dates and achievements."
        return self._send_search_request(prompt)
    
    def search_platform_integration(self, coin_symbol):
        """Search for platform integrations for a cryptocurrency"""
        prompt = f"Find recent platform integrations for {coin_symbol} cryptocurrency. Focus on which services, exchanges, wallets, DeFi protocols, or other platforms have added support for this cryptocurrency. Be factual and detailed."
        return self._send_search_request(prompt)
    
    def search_team_changes(self, coin_symbol):
        """Search for team changes for a cryptocurrency"""
        prompt = f"Provide detailed information about recent team changes within the {coin_symbol} cryptocurrency project. Include information about key hires, departures, role changes, or organizational restructuring that might impact the project's development or direction."
        return self._send_search_request(prompt)
    
    def search_new_partnerships(self, coin_symbol):
        """Search for new partnerships for a cryptocurrency"""
        prompt = f"Find recent partnership announcements for {coin_symbol} cryptocurrency. Focus on collaborations with companies, institutions, other blockchain projects, or government entities. Include details about the nature of these partnerships and their potential impact."
        return self._send_search_request(prompt)
    
    def search_onchain_activity(self, coin_symbol):
        """Search for on-chain activity for a cryptocurrency"""
        prompt = f"Provide detailed analysis of recent on-chain activity for {coin_symbol} cryptocurrency. Include data about transaction volumes, active addresses, network usage patterns, gas/fee trends, and any notable on-chain events or anomalies."
        return self._send_search_request(prompt)
    
    def search_active_addresses_growth(self, coin_symbol):
        """Search for active addresses growth for a cryptocurrency"""
        prompt = f"Find recent data about active addresses growth for {coin_symbol} cryptocurrency. Focus on trends in daily/monthly active addresses, new address creation, and user adoption metrics. Include specific numbers and growth percentages when available."
        return self._send_search_request(prompt)
    
    def search_real_world_adoption(self, coin_symbol):
        """Search for real world adoption for a cryptocurrency"""
        prompt = f"Provide detailed information about real-world adoption of {coin_symbol} cryptocurrency. Focus on merchant adoption, payment processing integration, institutional usage, or adoption by specific industries or countries. Be factual and cite specific examples."
        return self._send_search_request(prompt)
    
    def search_developer_activity(self, coin_symbol):
        """Search for developer activity for a cryptocurrency"""
        prompt = f"Find recent data about developer activity for {coin_symbol} cryptocurrency. Include information about GitHub commits, pull requests, developer count, code audits, bounty programs, and general development progress. Be specific about technical achievements."
        return self._send_search_request(prompt)
    
    def search_community_sentiment(self, coin_symbol):
        """Search for community sentiment for a cryptocurrency"""
        prompt = f"Provide analysis of recent community sentiment for {coin_symbol} cryptocurrency. Include information from social media trends, community discussions, sentiment analysis, and general market perception. Focus on factual data rather than speculation."
        return self._send_search_request(prompt)
    
    def search_liquidity_changes(self, coin_symbol):
        """Search for liquidity changes for a cryptocurrency"""
        prompt = f"Find recent data about liquidity changes for {coin_symbol} cryptocurrency. Focus on trading volume trends, market depth, liquidity pool sizes, exchange listings/delistings, and any factors affecting market liquidity. Include specific metrics when available."
        return self._send_search_request(prompt)
    
    # New category search functions
    def search_market_cap(self, coin_symbol):
        """Search for market cap information for a cryptocurrency"""
        prompt = f"Provide detailed information about the current market capitalization of {coin_symbol} cryptocurrency. Include historical market cap trends, rank among other cryptocurrencies, and factors influencing recent market cap changes. Include specific numbers and percentages when available."
        return self._send_search_request(prompt)
    
    def search_circulating_vs_total_supply(self, coin_symbol):
        """Search for circulating supply vs total supply information for a cryptocurrency"""
        prompt = f"Find detailed information about the circulating supply versus total supply of {coin_symbol} cryptocurrency. Include current numbers, release schedules, token unlocks, and how the relationship between circulating and total supply affects tokenomics. Be specific with numbers and percentages."
        return self._send_search_request(prompt)
    
    def search_tokenomics(self, coin_symbol):
        """Search for tokenomics information for a cryptocurrency"""
        prompt = f"Provide a comprehensive analysis of the tokenomics for {coin_symbol} cryptocurrency. Include token distribution, allocation to different stakeholders (team, investors, community, etc.), vesting schedules, token utility, and economic model. Be detailed and factual."
        return self._send_search_request(prompt)
    
    def search_inflation_emission_rate(self, coin_symbol):
        """Search for inflation/emission rate information for a cryptocurrency"""
        prompt = f"Find detailed information about the inflation or emission rate of {coin_symbol} cryptocurrency. Include current emission schedule, changes to monetary policy, inflation rate compared to other assets, and how emission affects token value. Be specific with numbers and time frames."
        return self._send_search_request(prompt)
    
    def search_token_utility(self, coin_symbol):
        """Search for token utility information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of the utility of {coin_symbol} token. Focus on use cases within its ecosystem, staking benefits, governance rights, fee mechanisms, and any other ways the token provides value to holders. Include specific examples of token utility in action."
        return self._send_search_request(prompt)
    
    def search_network_activity_usage(self, coin_symbol):
        """Search for network activity and usage information for a cryptocurrency"""
        prompt = f"Find detailed information about network activity and usage metrics for {coin_symbol} cryptocurrency. Include data on daily transactions, unique users, dApp usage if applicable, network load, and growth trends. Be specific with numbers and comparisons to previous periods."
        return self._send_search_request(prompt)
    
    def search_transaction_fees_revenue(self, coin_symbol):
        """Search for transaction fees and revenue information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of transaction fees and revenue for the {coin_symbol} network. Include data on fee structures, average transaction costs, total network revenue, fee distribution mechanism, and how fees compare to competitor networks. Be specific with numbers and trends."
        return self._send_search_request(prompt)
    
    def search_governance_decentralization(self, coin_symbol):
        """Search for governance and decentralization information for a cryptocurrency"""
        prompt = f"Find detailed information about the governance model and level of decentralization of {coin_symbol} cryptocurrency. Include data on governance token distribution, voting mechanisms, recent governance proposals, validator/node distribution, and measures of decentralization. Be factual and specific."
        return self._send_search_request(prompt)
    
    def search_exchange_listings(self, coin_symbol):
        """Search for exchange listings information for a cryptocurrency"""
        prompt = f"Provide detailed information about exchange listings for {coin_symbol} cryptocurrency. Include major exchanges where it's available, recent new listings, delisting events, trading pairs offered, and liquidity across different exchanges. Be comprehensive and specific."
        return self._send_search_request(prompt)
    
    def search_regulatory_status(self, coin_symbol):
        """Search for regulatory status information for a cryptocurrency"""
        prompt = f"Find detailed information about the regulatory status of {coin_symbol} cryptocurrency across different jurisdictions. Include regulatory classifications, compliance efforts, legal challenges, and how regulatory developments might affect the project. Be factual and jurisdiction-specific."
        return self._send_search_request(prompt)
    
    def search_security_audits(self, coin_symbol):
        """Search for security audits information for a cryptocurrency"""
        prompt = f"Provide detailed information about security audits conducted for {coin_symbol} cryptocurrency. Include names of auditing firms, dates of audits, key findings, vulnerabilities identified and fixed, and overall security posture. Be specific about technical security aspects."
        return self._send_search_request(prompt)
    
    def search_competitor_analysis(self, coin_symbol):
        """Search for competitor analysis information for a cryptocurrency"""
        prompt = f"Find detailed competitive analysis of {coin_symbol} cryptocurrency compared to its main competitors. Include feature comparisons, technical advantages/disadvantages, market positioning, adoption metrics, and how it differentiates itself in the market. Be objective and fact-based."
        return self._send_search_request(prompt)
    
    def search_volatility_price_swings(self, coin_symbol):
        """Search for volatility and price swings information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of volatility and significant price swings for {coin_symbol} cryptocurrency. Include volatility metrics compared to other assets, recent major price movements, triggers for volatility, and historical volatility patterns. Be specific with percentage changes and time frames."
        return self._send_search_request(prompt)
    
    def search_unusual_trading_patterns(self, coin_symbol):
        """Search for unusual trading patterns information for a cryptocurrency"""
        prompt = f"Find detailed information about any unusual trading patterns observed for {coin_symbol} cryptocurrency. Include anomalous volume spikes, price manipulations, wash trading indicators, or other suspicious market activities. Focus on factual data and avoid speculation."
        return self._send_search_request(prompt)
    
    def search_divergence_market_trends(self, coin_symbol):
        """Search for divergence from market trends information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of how {coin_symbol} cryptocurrency has diverged from broader market trends. Include correlation with Bitcoin, other major cryptos, traditional markets, and instances of decoupling. Explain potential reasons for divergence with specific examples and time periods."
        return self._send_search_request(prompt)
    
    def search_catalysts(self, coin_symbol):
        """Search for catalysts (news/events) information for a cryptocurrency"""
        prompt = f"Find detailed information about upcoming catalysts (news or events) that could impact {coin_symbol} cryptocurrency. Include scheduled protocol upgrades, token unlocks, conference appearances, product launches, or other significant events. Be specific about dates and potential impact."
        return self._send_search_request(prompt)
    
    def search_technical_signals(self, coin_symbol):
        """Search for technical signals information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of technical signals for {coin_symbol} cryptocurrency. Include key technical indicators (moving averages, RSI, MACD, etc.), chart patterns, support/resistance levels, and what these signals might suggest about future price action. Be specific about current technical positioning."
        return self._send_search_request(prompt)
    
    def search_comparative_strength_weakness(self, coin_symbol):
        """Search for comparative strength/weakness information for a cryptocurrency"""
        prompt = f"Find detailed information about the comparative strength or weakness of {coin_symbol} cryptocurrency relative to the broader market or its sector. Include relative strength indicators, outperformance/underperformance metrics, and factors contributing to its relative position. Use specific metrics and time frames."
        return self._send_search_request(prompt)
    
    def search_trading_volume_anomalies(self, coin_symbol):
        """Search for trading volume anomalies information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of trading volume anomalies for {coin_symbol} cryptocurrency. Include unusual volume spikes, volume divergence from price action, changes in volume distribution across exchanges, and potential factors behind abnormal trading activity. Be specific with data and dates."
        return self._send_search_request(prompt)
    
    def search_exchange_inflows_outflows(self, coin_symbol):
        """Search for exchange inflows/outflows information for a cryptocurrency"""
        prompt = f"Find detailed information about exchange inflows and outflows for {coin_symbol} cryptocurrency. Include data on tokens moving to/from exchanges, whale wallet movements, accumulation/distribution patterns, and what these flows might indicate about market sentiment. Be specific with numbers and trends."
        return self._send_search_request(prompt)
    
    def search_open_interest_futures(self, coin_symbol):
        """Search for open interest and futures activity information for a cryptocurrency"""
        prompt = f"Provide detailed analysis of open interest and futures market activity for {coin_symbol} cryptocurrency. Include data on open interest changes, funding rates, long/short ratios, liquidation events, and derivatives market positioning. Be specific with numbers and exchange-level data when available."
        return self._send_search_request(prompt)
    
    def analyze_cryptocurrency_full(self, coin_symbol):
        """Analyze all categories for a cryptocurrency"""
        results = {}
        
        # Use a different API key for each category to avoid rate limiting
        categories = [
            # Original categories
            ("Feature Releases or Updates", self.search_feature_releases),
            ("Whitepaper Updates", self.search_whitepaper_updates),
            ("Testnet or Mainnet Milestones", self.search_testnet_mainnet_milestones),
            ("Platform Integration", self.search_platform_integration),
            ("Team Changes", self.search_team_changes),
            ("New Partnerships", self.search_new_partnerships),
            ("On-chain Activity", self.search_onchain_activity),
            ("Active Addresses Growth", self.search_active_addresses_growth),
            ("Real World Adoption", self.search_real_world_adoption),
            ("Developer Activity", self.search_developer_activity),
            ("Community Sentiment", self.search_community_sentiment),
            ("Liquidity Changes", self.search_liquidity_changes),
            
            # New categories
            ("Market Cap", self.search_market_cap),
            ("Circulating Supply vs Total Supply", self.search_circulating_vs_total_supply),
            ("Tokenomics", self.search_tokenomics),
            ("Inflation/Emission Rate", self.search_inflation_emission_rate),
            ("Token Utility", self.search_token_utility),
            ("Network Activity & Usage", self.search_network_activity_usage),
            ("Transaction Fees and Revenue", self.search_transaction_fees_revenue),
            ("Governance and Decentralization", self.search_governance_decentralization),
            ("Exchange Listings", self.search_exchange_listings),
            ("Regulatory Status", self.search_regulatory_status),
            ("Security Audits", self.search_security_audits),
            ("Competitor Analysis", self.search_competitor_analysis),
            ("Volatility Spikes / Price Swings", self.search_volatility_price_swings),
            ("Unusual Trading Patterns", self.search_unusual_trading_patterns),
            ("Divergence from Market Trends", self.search_divergence_market_trends),
            ("Catalysts (News/Events)", self.search_catalysts),
            ("Technical Signals", self.search_technical_signals),
            ("Comparative Strength/Weakness", self.search_comparative_strength_weakness),
            ("Trading Volume Anomalies", self.search_trading_volume_anomalies),
            ("Exchange Inflows/Outflows", self.search_exchange_inflows_outflows),
            ("Open Interest & Futures Activity", self.search_open_interest_futures)
        ]
        
        for i, (category, search_func) in enumerate(categories):
            # Rotate API keys
            searcher = CryptoSearcher(api_key_index=i)
            print(f"Processing category: {category}")
            result = search_func(searcher, coin_symbol)
            results[category] = result
            print(f"✓ Completed: {category}")
            time.sleep(0.5)  # Small delay between requests
        
        return results

    def save_analysis_to_file(self, coin_symbol, results):
        """Save analysis results to a file"""
        filename = f"{coin_symbol}_analysis.txt"
        
        with open(filename, "w") as f:
            f.write(f"=== CRYPTOCURRENCY ANALYSIS FOR {coin_symbol} ===\n\n")
            
            # Write results in order
            for category in [
                # Original categories
                "Feature Releases or Updates",
                "Whitepaper Updates",
                "Testnet or Mainnet Milestones",
                "Platform Integration",
                "Team Changes",
                "New Partnerships",
                "On-chain Activity",
                "Active Addresses Growth",
                "Real World Adoption",
                "Developer Activity",
                "Community Sentiment",
                "Liquidity Changes",
                
                # New categories
                "Market Cap",
                "Circulating Supply vs Total Supply",
                "Tokenomics",
                "Inflation/Emission Rate",
                "Token Utility",
                "Network Activity & Usage",
                "Transaction Fees and Revenue",
                "Governance and Decentralization",
                "Exchange Listings",
                "Regulatory Status",
                "Security Audits",
                "Competitor Analysis",
                "Volatility Spikes / Price Swings",
                "Unusual Trading Patterns",
                "Divergence from Market Trends",
                "Catalysts (News/Events)",
                "Technical Signals",
                "Comparative Strength/Weakness",
                "Trading Volume Anomalies",
                "Exchange Inflows/Outflows",
                "Open Interest & Futures Activity"
            ]:
                if category in results:
                    f.write(f"=== {category} ===\n")
                    f.write(results[category])
                    f.write("\n\n")
        
        print(f"Analysis complete. Results saved to {filename}")
        return filename


# Exported individual functions for direct import - original functions
def search_feature_releases(coin_symbol, api_key_index=0):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_feature_releases(coin_symbol)

def search_whitepaper_updates(coin_symbol, api_key_index=1):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_whitepaper_updates(coin_symbol)

def search_testnet_mainnet_milestones(coin_symbol, api_key_index=2):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_testnet_mainnet_milestones(coin_symbol)

def search_platform_integration(coin_symbol, api_key_index=3):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_platform_integration(coin_symbol)

def search_team_changes(coin_symbol, api_key_index=4):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_team_changes(coin_symbol)

def search_new_partnerships(coin_symbol, api_key_index=5):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_new_partnerships(coin_symbol)

def search_onchain_activity(coin_symbol, api_key_index=6):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_onchain_activity(coin_symbol)

def search_active_addresses_growth(coin_symbol, api_key_index=7):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_active_addresses_growth(coin_symbol)

def search_real_world_adoption(coin_symbol, api_key_index=8):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_real_world_adoption(coin_symbol)

def search_developer_activity(coin_symbol, api_key_index=9):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_developer_activity(coin_symbol)

def search_community_sentiment(coin_symbol, api_key_index=10):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_community_sentiment(coin_symbol)

def search_liquidity_changes(coin_symbol, api_key_index=11):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_liquidity_changes(coin_symbol)

# Exported individual functions for direct import - new functions
def search_market_cap(coin_symbol, api_key_index=12):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_market_cap(coin_symbol)

def search_circulating_vs_total_supply(coin_symbol, api_key_index=13):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_circulating_vs_total_supply(coin_symbol)

def search_tokenomics(coin_symbol, api_key_index=14):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_tokenomics(coin_symbol)

def search_inflation_emission_rate(coin_symbol, api_key_index=15):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_inflation_emission_rate(coin_symbol)

def search_token_utility(coin_symbol, api_key_index=16):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_token_utility(coin_symbol)

def search_network_activity_usage(coin_symbol, api_key_index=17):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_network_activity_usage(coin_symbol)

def search_transaction_fees_revenue(coin_symbol, api_key_index=18):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_transaction_fees_revenue(coin_symbol)

def search_governance_decentralization(coin_symbol, api_key_index=19):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_governance_decentralization(coin_symbol)

def search_exchange_listings(coin_symbol, api_key_index=20):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_exchange_listings(coin_symbol)

def search_regulatory_status(coin_symbol, api_key_index=21):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_regulatory_status(coin_symbol)

def search_security_audits(coin_symbol, api_key_index=22):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_security_audits(coin_symbol)

def search_competitor_analysis(coin_symbol, api_key_index=23):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_competitor_analysis(coin_symbol)

def search_volatility_price_swings(coin_symbol, api_key_index=24):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_volatility_price_swings(coin_symbol)

def search_unusual_trading_patterns(coin_symbol, api_key_index=25):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_unusual_trading_patterns(coin_symbol)

def search_divergence_market_trends(coin_symbol, api_key_index=26):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_divergence_market_trends(coin_symbol)

def search_catalysts(coin_symbol, api_key_index=27):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_catalysts(coin_symbol)

def search_technical_signals(coin_symbol, api_key_index=0):  # Reset index counter as needed
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_technical_signals(coin_symbol)

def search_comparative_strength_weakness(coin_symbol, api_key_index=1):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_comparative_strength_weakness(coin_symbol)

def search_trading_volume_anomalies(coin_symbol, api_key_index=2):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_trading_volume_anomalies(coin_symbol)

def search_exchange_inflows_outflows(coin_symbol, api_key_index=3):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_exchange_inflows_outflows(coin_symbol)

def search_open_interest_futures(coin_symbol, api_key_index=4):
    searcher = CryptoSearcher(api_key_index)
    return searcher.search_open_interest_futures(coin_symbol)

def analyze_cryptocurrency_full(coin_symbol):
    searcher = CryptoSearcher()
    return searcher.analyze_cryptocurrency_full(coin_symbol)

def save_analysis_to_file(coin_symbol, results):
    searcher = CryptoSearcher()
    return searcher.save_analysis_to_file(coin_symbol, results)


if __name__ == "__main__":
    # Example of direct usage
    print("Enhanced Crypto Searcher")
    
    # Get coin symbol from user
    coin_symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").upper()
    
    # Choose analysis type
    print("\nChoose analysis type:")
    print("1. Full analysis (all categories)")
    print("2. Single category analysis")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Full analysis
        results = analyze_cryptocurrency_full(coin_symbol)
        save_analysis_to_file(coin_symbol, results)
    elif choice == "2":
        # Single category analysis
        categories = [
            # Original categories
            "Feature Releases or Updates",
            "Whitepaper Updates",
            "Testnet or Mainnet Milestones",
            "Platform Integration",
            "Team Changes",
            "New Partnerships",
            "On-chain Activity",
            "Active Addresses Growth",
            "Real World Adoption",
            "Developer Activity",
            "Community Sentiment",
            "Liquidity Changes",
            
            # New categories
            "Market Cap",
            "Circulating Supply vs Total Supply",
            "Tokenomics",
            "Inflation/Emission Rate",
            "Token Utility",
            "Network Activity & Usage",
            "Transaction Fees and Revenue",
            "Governance and Decentralization",
            "Exchange Listings",
            "Regulatory Status",
            "Security Audits",
            "Competitor Analysis",
            "Volatility Spikes / Price Swings",
            "Unusual Trading Patterns",
            "Divergence from Market Trends",
            "Catalysts (News/Events)",
            "Technical Signals",
            "Comparative Strength/Weakness",
            "Trading Volume Anomalies",
            "Exchange Inflows/Outflows",
            "Open Interest & Futures Activity"
        ]
        
        print("\nAvailable categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        cat_choice = int(input("\nSelect category number: ")) - 1
        
        if 0 <= cat_choice < len(categories):
            category = categories[cat_choice]
            searcher = CryptoSearcher(api_key_index=cat_choice)
            
            # Call the appropriate method based on the category
            if category == "Feature Releases or Updates":
                result = searcher.search_feature_releases(coin_symbol)
            elif category == "Whitepaper Updates":
                result = searcher.search_whitepaper_updates(coin_symbol)
            elif category == "Testnet or Mainnet Milestones":
                result = searcher.search_testnet_mainnet_milestones(coin_symbol)
            elif category == "Platform Integration":
                result = searcher.search_platform_integration(coin_symbol)
            elif category == "Team Changes":
                result = searcher.search_team_changes(coin_symbol)
            elif category == "New Partnerships":
                result = searcher.search_new_partnerships(coin_symbol)
            elif category == "On-chain Activity":
                result = searcher.search_onchain_activity(coin_symbol)
            elif category == "Active Addresses Growth":
                result = searcher.search_active_addresses_growth(coin_symbol)
            elif category == "Real World Adoption":
                result = searcher.search_real_world_adoption(coin_symbol)
            elif category == "Developer Activity":
                result = searcher.search_developer_activity(coin_symbol)
            elif category == "Community Sentiment":
                result = searcher.search_community_sentiment(coin_symbol)
            elif category == "Liquidity Changes":
                result = searcher.search_liquidity_changes(coin_symbol)
            # New category handlers
            elif category == "Market Cap":
                result = searcher.search_market_cap(coin_symbol)
            elif category == "Circulating Supply vs Total Supply":
                result = searcher.search_circulating_vs_total_supply(coin_symbol)
            elif category == "Tokenomics":
                result = searcher.search_tokenomics(coin_symbol)
            elif category == "Inflation/Emission Rate":
                result = searcher.search_inflation_emission_rate(coin_symbol)
            elif category == "Token Utility":
                result = searcher.search_token_utility(coin_symbol)
            elif category == "Network Activity & Usage":
                result = searcher.search_network_activity_usage(coin_symbol)
            elif category == "Transaction Fees and Revenue":
                result = searcher.search_transaction_fees_revenue(coin_symbol)
            elif category == "Governance and Decentralization":
                result = searcher.search_governance_decentralization(coin_symbol)
            elif category == "Exchange Listings":
                result = searcher.search_exchange_listings(coin_symbol)
            elif category == "Regulatory Status":
                result = searcher.search_regulatory_status(coin_symbol)
            elif category == "Security Audits":
                result = searcher.search_security_audits(coin_symbol)
            elif category == "Competitor Analysis":
                result = searcher.search_competitor_analysis(coin_symbol)
            elif category == "Volatility Spikes / Price Swings":
                result = searcher.search_volatility_price_swings(coin_symbol)
            elif category == "Unusual Trading Patterns":
                result = searcher.search_unusual_trading_patterns(coin_symbol)
            elif category == "Divergence from Market Trends":
                result = searcher.search_divergence_market_trends(coin_symbol)
            elif category == "Catalysts (News/Events)":
                result = searcher.search_catalysts(coin_symbol)
            elif category == "Technical Signals":
                result = searcher.search_technical_signals(coin_symbol)
            elif category == "Comparative Strength/Weakness":
                result = searcher.search_comparative_strength_weakness(coin_symbol)
            elif category == "Trading Volume Anomalies":
                result = searcher.search_trading_volume_anomalies(coin_symbol)
            elif category == "Exchange Inflows/Outflows":
                result = searcher.search_exchange_inflows_outflows(coin_symbol)
            elif category == "Open Interest & Futures Activity":
                result = searcher.search_open_interest_futures(coin_symbol)
            
            print(f"\n=== {category} ===")
            print(result)
        else:
            print("Invalid category selection.")
    else:
        print("Invalid choice.")