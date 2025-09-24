"""
Flask TTS API Application Factory
"""

import os
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .extensions import init_extensions, init_error_handlers, init_logging
from config import DevelopmentConfig, ProductionConfig, TestingConfig


def create_app(config_name: Optional[str] = None) -> Flask:
    """Create and configure Flask application.

    Args:
        config_name: Configuration name (development, production, testing)

    Returns:
        Configured Flask application instance
    """
    # Determine configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    # Configuration mapping
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }

    if config_name not in config_map:
        raise ValueError(f"Invalid configuration name: {config_name}")

    # Create Flask app
    app = Flask(__name__)

    # Load configuration
    config_class = config_map[config_name]
    app.config.from_object(config_class)

    # Initialize logging
    init_logging(app)

    # Initialize extensions
    init_extensions(app)

    # Initialize error handlers
    init_error_handlers(app)

    # Register blueprints
    from routes import auth_bp, tts_bp, bi_bp
    app.register_blueprint(auth_bp, url_prefix='/api/v1/auth')
    app.register_blueprint(tts_bp, url_prefix='/api/v1/tts')
    app.register_blueprint(bi_bp, url_prefix='/api/v1/bi')

    # Health check endpoint
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'environment': config_name
        })

    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API information."""
        return jsonify({
            'name': 'Flask TTS API',
            'version': '1.0.0',
            'description': 'Text-to-Speech API using Google Gemini',
            'docs': '/api/v1/health',
            'endpoints': {
                'auth': '/api/v1/auth',
                'tts': '/api/v1/tts',
                'bi': '/api/v1/bi',
                'health': '/api/v1/health'
            }
        })

    # Request logging middleware
    @app.before_request
    def log_request_info():
        """Log request information with sensitive data protection."""
        if app.config.get('LOG_LEVEL', 'INFO') == 'DEBUG':
            # Sanitize request information to prevent information disclosure
            sanitized_url = request.url
            sanitized_headers = dict(request.headers)

            # Remove sensitive headers
            sensitive_headers = ['authorization', 'cookie', 'x-api-key', 'x-auth-token']
            for header in sensitive_headers:
                if header in sanitized_headers:
                    sanitized_headers[header] = '[REDACTED]'

            app.logger.debug(f'{request.method} {sanitized_url} - {request.remote_addr}')

    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'; "
            "object-src 'none'"
        )
        response.headers['Content-Security-Policy'] = csp_policy

        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

        # Log response information
        if app.config.get('LOG_LEVEL', 'INFO') == 'DEBUG':
            app.logger.debug(f'Response: {response.status_code} - {response.content_length or 0} bytes')

        return response

    # Handle all HTTP exceptions
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle HTTP exceptions."""
        response = error.get_response()
        response.data = jsonify({
            'error': error.name,
            'message': error.description,
            'status_code': error.code
        }).data
        response.content_type = 'application/json'
        return response, error.code

    # Handle unhandled exceptions
    @app.errorhandler(Exception)
    def handle_unhandled_exception(error):
        """Handle unhandled exceptions."""
        app.logger.error(f'Unhandled exception: {str(error)}', exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500

    app.logger.info(f'Flask TTS API started in {config_name} mode')

    return app


# Create app instance for WSGI servers
app = create_app()