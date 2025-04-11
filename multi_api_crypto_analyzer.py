from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import threading
import time
import traceback
import queue
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

class CryptoAnalyzer:
    # API keys to use
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
        "AIzaSyDg6vphZ4rGu87f4orzk1B56wY0es5DQes"
    ]

    # Categories to analyze
    CATEGORIES = [
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
    ]

    # Gemini model ID
    MODEL_ID = "gemini-2.0-flash"

    # Thread local storage for clients
    thread_local = threading.local()

    def __init__(self):
        """Initialize the CryptoAnalyzer"""
        pass
    
    def get_gemini_client(self, api_key):
        """Get or create a Gemini client for the current thread"""
        if not hasattr(self.thread_local, "clients"):
            self.thread_local.clients = {}
        
        if api_key not in self.thread_local.clients:
            self.thread_local.clients[api_key] = genai.Client(api_key=api_key)
        
        return self.thread_local.clients[api_key]

    def search_with_gemini(self, api_key, query, category):
        """Use Gemini with Google Search integration to answer a query with specific API key"""
        
        print(f"Searching for {category} information...")
        
        # Get client from thread local storage
        client = self.get_gemini_client(api_key)
        
        # Create the Google Search tool
        google_search_tool = Tool(
            google_search = GoogleSearch()
        )
        
        try:
            # Generate a response with search capability
            response = client.models.generate_content(
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
                            return {
                                "category": category,
                                "content": result_text
                            }
            
            # If we get here, we didn't extract text successfully
            print(f"No usable content found for {category}")
            return {
                "category": category,
                "content": f"No information found for {category}."
            }
            
        except Exception as e:
            print(f"Error in API call for {category}: {str(e)}")
            traceback.print_exc()
            return {
                "category": category,
                "content": f"Error retrieving information for {category}: {str(e)}"
            }

    def worker_thread(self, api_key, coin_symbol, category):
        """Worker thread that analyzes one category using one API key"""
        
        # Simple prompt focused on a single category
        prompt = f"Go explore about latest about {coin_symbol}, regarding {category}. Never start with Here is the information about or anything like that. Just start with the information directly. Do not start with a introduction at all. Explain the information very detail and long, do not leave any information out. "
        
        # Call Gemini API for this category
        return self.search_with_gemini(api_key, prompt, category)

    def analyze_single_category(self, coin_symbol, category):
        """Analyze a single category for a cryptocurrency"""
        
        # Get an API key (simple round-robin)
        api_key = self.API_KEYS[self.CATEGORIES.index(category) % len(self.API_KEYS)]
        
        # Use the worker thread function directly
        return self.worker_thread(api_key, coin_symbol, category)

    def analyze_cryptocurrency(self, coin_symbol):
        """Analyze a cryptocurrency using multiple API keys in parallel threads"""
        
        results = {}
        
        # Use ThreadPoolExecutor for better thread management
        with ThreadPoolExecutor(max_workers=min(len(self.CATEGORIES), 8)) as executor:
            # Submit all tasks
            future_to_category = {}
            for i, category in enumerate(self.CATEGORIES):
                api_key = self.API_KEYS[i % len(self.API_KEYS)]
                future = executor.submit(self.worker_thread, api_key, coin_symbol, category)
                future_to_category[future] = category
                
                # Small delay to prevent API rate limiting
                time.sleep(0.2)
            
            # Process results as they complete
            for future in as_completed(future_to_category):
                result = future.result()
                category = result["category"]
                content = result["content"]
                
                # Store in results dictionary
                results[category] = content
                print(f"✓ Completed: {category}")
        
        return results

    def save_analysis_to_file(self, coin_symbol, results):
        """Save analysis results to a file"""
        
        # Create a buffer for the file output
        output_buffer = io.StringIO()
        output_buffer.write(f"=== CRYPTOCURRENCY ANALYSIS FOR {coin_symbol} ===\n\n")
        
        # Write results to buffer in order of CATEGORIES
        for category in self.CATEGORIES:
            if category in results:
                output_buffer.write(f"=== {category} ===\n")
                output_buffer.write(results[category])
                output_buffer.write("\n\n")
        
        # Get the output string from the buffer
        output_text = output_buffer.getvalue()
        
        # Write results to a file using direct buffer write
        filename = f"{coin_symbol}_analysis.txt"
        with open(filename, "w", buffering=1024*1024) as f:
            f.write(output_text)
        
        print(f"Analysis complete. Results saved to {filename}")
        return filename


if __name__ == "__main__":
    # Example of direct usage
    analyzer = CryptoAnalyzer()
    print("Crypto Analyzer")
    
    # Get coin symbol from user
    coin_symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").upper()
    
    # Choose analysis type
    print("\nChoose analysis type:")
    print("1. Full analysis (all categories)")
    print("2. Single category analysis")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Full analysis
        results = analyzer.analyze_cryptocurrency(coin_symbol)
        analyzer.save_analysis_to_file(coin_symbol, results)
    elif choice == "2":
        # Single category analysis
        print("\nAvailable categories:")
        for i, category in enumerate(analyzer.CATEGORIES, 1):
            print(f"{i}. {category}")
        
        cat_choice = int(input("\nSelect category number: ")) - 1
        if 0 <= cat_choice < len(analyzer.CATEGORIES):
            category = analyzer.CATEGORIES[cat_choice]
            result = analyzer.analyze_single_category(coin_symbol, category)
            print(f"\n=== {category} ===")
            print(result["content"])
        else:
            print("Invalid category selection.")
    else:
        print("Invalid choice.") 