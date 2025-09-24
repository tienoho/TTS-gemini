"""
Flask TTS API Main Application Entry Point
"""

import os
import sys
sys.path.append('/app')
from app import create_app

# Create Flask application
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    config_name = os.getenv('FLASK_ENV', 'development')

    # Development server configuration
    if config_name == 'development':
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=True,
            threaded=True
        )
    else:
        # Production server should be run with a WSGI server like gunicorn
        print("For production, use a WSGI server like gunicorn:")
        print("gunicorn --bind 0.0.0.0:5000 app.main:app")