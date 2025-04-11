from flask import jsonify, request
from multi_api_crypto_analyzer import CryptoAnalyzer

# Initialize the analyzer once
analyzer = CryptoAnalyzer()

def register_crypto_endpoints(app):
    """Register all crypto analyzer endpoints with the Flask app"""
    
    @app.route('/api/analyze/<coin_symbol>', methods=['GET'])
    def analyze_full(coin_symbol):
        """Full cryptocurrency analysis across all categories"""
        results = analyzer.analyze_cryptocurrency(coin_symbol)
        return jsonify(results)
    
    # Create individual category endpoints
    for category in analyzer.CATEGORIES:
        # Convert category to URL-friendly format
        endpoint = category.lower().replace(' ', '_')
        
        # Create a unique endpoint function for each category using a factory pattern
        def create_category_endpoint(cat):
            def endpoint_function(coin_symbol):
                result = analyzer.analyze_single_category(coin_symbol, cat)
                return jsonify(result)
            # Return the created function
            return endpoint_function
        
        # Create a unique function for this specific category
        category_func = create_category_endpoint(category)
        
        # Set a unique name for the function
        category_func.__name__ = f"analyze_{endpoint}"
        
        # Register the route with the app
        app.route(f'/api/analyze/<coin_symbol>/{endpoint}', methods=['GET'])(category_func) 