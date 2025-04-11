from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBdMrLHWWswcUjJABblf2E1n2s1TZqYu9w")

# Initialize the client
client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash"

def search_with_gemini(query):
    """Use Gemini with Google Search integration to answer a query"""
    
    # Create the Google Search tool
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    
    try:
        # Generate a response with search capability
        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
                temperature=0.7,
                max_output_tokens=8192,
            )
        )
        
        # Extract and return the text from the response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts') and content.parts:
                    all_text = ""
                    for part in content.parts:
                        if hasattr(part, 'text'):
                            all_text += part.text + "\n"
                    return all_text.strip()
        
        return "No response generated."
        
    except Exception as e:
        return f"Error: {str(e)}"

def ask_gemini(topic):
    """Ask Gemini about any topic with web search capability"""
    
    prompt = f"""
    Please provide information about: {topic}
    
    Include:
    1. Key facts and important details
    2. Recent developments if applicable
    3. Relevant context
    
    Please cite your sources.
    """
    
    print(f"\nSearching for information about: {topic}")
    result = search_with_gemini(prompt)
    print("\n=== ANSWER ===")
    print(result)
    return result

if __name__ == "__main__":
    print("Gemini AI Web Search Tool")
    print("------------------------")
    
    while True:
        user_query = input("\nWhat would you like to learn about? (or type 'exit' to quit): ")
        
        if user_query.lower() in ["exit", "quit", "q"]:
            break
        
        ask_gemini(user_query) 