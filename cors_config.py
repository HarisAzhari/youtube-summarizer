from flask_cors import CORS

def configure_cors(app):
    """Configure CORS for the Flask application"""
    
    # Enable CORS for all routes
    CORS(app, resources={
        # Allow all routes
        r"/*": {
            # Allow requests from any origin
            "origins": "*",
            # Allow these methods
            "methods": ["GET", "OPTIONS"],
            # Allow these headers in requests
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    return app 