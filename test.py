from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import google.generativeai as palm  # Using palm as the alias since this version is based on PaLM API
from google.genai import types
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = Flask(__name__)
CORS(app)

# Configure Gemini clients
EMBEDDING_API_KEY = "AIzaSyAJkVH1OkkhIJIvkQ4_zj7MvbwOgcvJifA"
GEMINI_API_KEY = "AIzaSyAOogW_ZTPgDniIc0ecGSQk_4L9U_y7dno"
GEMINI_API_KEY_2 = "AIzaSyBxeQKYExn_2Mu2wkg9ExfQr_yn7RiJ6Ow"
GENERAL_API_KEY = "AIzaSyDNWfFmywWgydEI0NxL9xbCTjdlnYlOoKE"

# Initialize the API
palm.configure(api_key=EMBEDDING_API_KEY)

def clean_frontend_query(query: str) -> str:
    patterns = [
        r"^Are you asking about ",
        r"^Would you like to know ",
        r"^Do you want to learn ",
        r"^Are you interested in ",
    ]
    cleaned = query.lower()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("?", "").replace("the", "").replace("and", "")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_embeddings(text: str):
    try:
        result = palm.generate_embeddings(text=text)
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_similar_texts(query: str, top_k: int = 5):
    """Shared function to get similar texts"""
    try:
        with open("validated_content/embedded_toharoh_entries.json", 'r', encoding='utf-8') as f:
            toharoh_data = json.load(f)
            # Flatten the madhab-structured data
            embedded_entries = []
            for madhab, entries in toharoh_data.items():
                for entry in entries:
                    entry['madhab'] = madhab  # Add madhab info to entry
                    embedded_entries.append(entry)
    except FileNotFoundError:
        return None, "Database file not found"
    
    cleaned_query = clean_frontend_query(query)
    query_embedding = get_embeddings(cleaned_query)
    
    if not query_embedding:
        return None, "Failed to generate embeddings"
    
    results = []
    for entry in embedded_entries:
        similarity = cosine_similarity(query_embedding, entry['english_translation_embedding'])
        results.append({
            'madhab': entry['madhab'],
            'page': entry['page_number'],
            'similarity': similarity,
            'arabic': entry['arabic_text'],
            'english': entry['english_translation']
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k], cleaned_query

def generate_content(prompt: str, api_key: str):
    palm.configure(api_key=api_key)
    model = palm.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

def analyze_first_three(results):
    """Analyze definition, ruling, and evidence"""
    content_for_analysis = "Analyze these Islamic texts for definition, ruling, and evidence:\n\n"
    for idx, result in enumerate(results, 1):
        content_for_analysis += f"Text {idx}:\nOriginal:\n{result['arabic']}\n\nTranslation:\n{result['english']}\n\nPage: {result['page']}\n\n"

    prompt = f"""{content_for_analysis}
Please analyze these texts and provide a response in the following JSON format:
{{
    "definition": {{
        "explanation": "General explanation of the concept",
        "schools": [
            {{
                "name": "Hanafi",
                "definition": "How the Hanafi school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "Maliki",
                "definition": "How the Maliki school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "Shafi",
                "definition": "How the Shafi school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "Hanbali",
                "definition": "How the Hanbali school defines this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }},
            {{
                "name": "OTHER",
                "definition": "How other schools or scholars define this concept",
                "source": "Text number and page reference if available, otherwise 'No specific definition found'"
            }}
        ]
    }},
    "ruling": {{
        "explanation": "General explanation of the ruling",
        "schools": [
            {{
                "name": "Hanafi",
                "ruling": "The Hanafi ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "Maliki",
                "ruling": "The Maliki ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "Shafi",
                "ruling": "The Shafi ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "Hanbali",
                "ruling": "The Hanbali ruling on this matter",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }},
            {{
                "name": "OTHER",
                "ruling": "Rulings from other schools or scholars",
                "source": "Text number and page reference if available, otherwise 'No specific ruling found'"
            }}
        ]
    }},
    "evidence": {{
        "explanation": "Overview of the evidences presented",
        "schools": [
            {{
                "name": "Hanafi",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Hanafi school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
            {{
                "name": "Maliki",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Maliki school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
            {{
                "name": "Shafi",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Shafi school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
            {{
                "name": "Hanbali",
                "arabic_text": "Original Arabic/Urdu text supporting this evidence, strictly in arabic/urdu, which is evidence from Hanbali school if not available, return 'No specific evidence found' ",
                "translation": "English translation of the evidence text"
            }},
        ]
    }}
}}"""

    try:
        response_text = generate_content(prompt, GEMINI_API_KEY)
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            json_str = response_text.split('```')[1].strip()
        else:
            json_str = response_text.strip()
            
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Analysis failed", "message": str(e)}

def analyze_last_three(results):
    """Analyze reasoning, application, and examples"""
    content_for_analysis = "Analyze these Islamic texts for reasoning, practical applications, and examples:\n\n"
    for idx, result in enumerate(results, 1):
        content_for_analysis += f"Text {idx}:\nOriginal:\n{result['arabic']}\n\nTranslation:\n{result['english']}\n\nPage: {result['page']}\n\n"

    prompt = f"""{content_for_analysis}
Please analyze these texts and provide a response in the following JSON format:
{{
    "reasoning": {{
        "explanation": "Overview of the underlying wisdom and reasoning",
        "schools": [
            {{
                "name": "Hanafi",
                "reasoning": "The underlying wisdom according to Hanafi school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }},
            {{
                "name": "Maliki",
                "reasoning": "The underlying wisdom according to Maliki school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }},
            {{
                "name": "Shafi",
                "reasoning": "The underlying wisdom according to Shafi school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }},
            {{
                "name": "Hanbali",
                "reasoning": "The underlying wisdom according to Hanbali school",
                "arabic_text": "Original Arabic/Urdu text showing this reasoning",
                "translation": "English translation of the reasoning text"
            }}
        ]
    }},
    "application": {{
        "explanation": "Overview of practical applications",
        "schools": [
            {{
                "name": "Hanafi",
                "application": "Practical application according to Hanafi school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }},
            {{
                "name": "Maliki",
                "application": "Practical application according to Maliki school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }},
            {{
                "name": "Shafi",
                "application": "Practical application according to Shafi school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }},
            {{
                "name": "Hanbali",
                "application": "Practical application according to Hanbali school",
                "arabic_text": "Original Arabic/Urdu text about this application",
                "translation": "English translation of the application text"
            }}
        ]
    }},
    "examples": {{
        "explanation": "Overview of practical examples",
        "schools": [
            {{
                "name": "Hanafi",
                "example": "Examples from Hanafi school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }},
            {{
                "name": "Maliki",
                "example": "Examples from Maliki school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }},
            {{
                "name": "Shafi",
                "example": "Examples from Shafi school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }},
            {{
                "name": "Hanbali",
                "example": "Examples from Hanbali school",
                "arabic_text": "Original Arabic/Urdu text of the examples",
                "translation": "English translation of the examples"
            }}
        ]
    }}
}}"""

    try:
        response = palm.Client(api_key=GEMINI_API_KEY_2).models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        response_text = response.text
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            json_str = response_text.split('```')[1].strip()
        else:
            json_str = response_text.strip()
            
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Analysis failed", "message": str(e)}

@app.route('/api/search/first', methods=['POST'])
def handle_first_three():
    """Handle definition, ruling, and evidence"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results, cleaned_query = get_similar_texts(data['query'])
    if results is None:
        return jsonify({"error": "Search failed", "message": cleaned_query}), 500
    
    analysis = analyze_first_three(results)
    
    response = {
        "query": {
            "original": data['query'],
            "cleaned": cleaned_query
        },
        "similar_texts": [
            {
                "page": result['page'],
                "similarity": f"{result['similarity']:.2%}",
                "arabic": result['arabic'],
                "english": result['english']
            }
            for result in results
        ],
        "analysis": analysis
    }
    
    return jsonify(response)

@app.route('/api/search/last', methods=['POST'])
def handle_last_three():
    """Handle reasoning, application, and examples"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query parameter is required"}), 400
    
    results, cleaned_query = get_similar_texts(data['query'])
    if results is None:
        return jsonify({"error": "Search failed", "message": cleaned_query}), 500
    
    analysis = analyze_last_three(results)
    
    response = {
        "query": {
            "original": data['query'],
            "cleaned": cleaned_query
        },
        "similar_texts": [
            {
                "page": result['page'],
                "similarity": f"{result['similarity']:.2%}",
                "arabic": result['arabic'],
                "english": result['english']
            }
            for result in results
        ],
        "analysis": analysis
    }
    
    return jsonify(response)

@app.route('/api/query/general', methods=['POST'])
def handle_general_query():
    """Handle initial query by returning both query and extracted keywords"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        palm.configure(api_key=GENERAL_API_KEY)
        
        answer_prompt = f"""As an Islamic scholar, what you get is a formulated question. Just directly answer the question in general. Make sure at least there's some detailness".

This is the question user clicked that he wants to know about: {data['query']}

Return only the answer not another question, don't start the question with yes or no. Directly answer the question. Nothing else."""

        answer_response = generate_content(answer_prompt, GENERAL_API_KEY)

        keyword_prompt = f"""From this Islamic question: "{data['query']}"
Extract only the most important keywords related to Islamic terms, concepts, or practices. 
Return just 2-4 keywords separated by spaces, nothing else. For example: "wudu prayer fasting" """

        keyword_response = generate_content(keyword_prompt, GENERAL_API_KEY)
        
        return jsonify({
            "query": answer_response,
            "keyword": keyword_response
        })

    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        return jsonify({"error": error_msg}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)