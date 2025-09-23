"""
Flask extensions initialization for TTS API
"""

from flask_caching import Cache
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_monitoringdashboard import MonitoringDashboard

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
cors = CORS()
cache = Cache()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize monitoring dashboard
monitoring_dashboard = MonitoringDashboard()


def init_extensions(app):
    """Initialize all Flask extensions.

    Args:
        app: Flask application instance
    """
    # Initialize SQLAlchemy
    db.init_app(app)

    # Initialize Flask-Migrate
    migrate.init_app(app, db)

    # Initialize JWT
    jwt.init_app(app)

    # Initialize CORS
    cors.init_app(app, resources={
        r"/api/*": {
            "origins": app.config.get('CORS_ORIGINS', ['*']),
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            "supports_credentials": True
        }
    })

    # Initialize Cache
    cache.init_app(app)

    # Initialize Rate Limiter
    limiter.init_app(app)

    # Initialize Monitoring Dashboard (only in development)
    if app.config.get('ENABLE_MONITORING', False) and not app.testing:
        monitoring_dashboard.init_app(app)


def init_error_handlers(app):
    """Initialize custom error handlers.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors."""
        return {
            'error': 'Bad Request',
            'message': str(error.description) if hasattr(error, 'description') else 'Invalid request',
            'status_code': 400
        }, 400

    @app.errorhandler(401)
    def unauthorized(error):
        """Handle unauthorized errors."""
        return {
            'error': 'Unauthorized',
            'message': 'Authentication required',
            'status_code': 401
        }, 401

    @app.errorhandler(403)
    def forbidden(error):
        """Handle forbidden errors."""
        return {
            'error': 'Forbidden',
            'message': 'Access denied',
            'status_code': 403
        }, 403

    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors."""
        return {
            'error': 'Not Found',
            'message': 'Resource not found',
            'status_code': 404
        }, 404

    @app.errorhandler(413)
    def request_too_large(error):
        """Handle request too large errors."""
        return {
            'error': 'Request Too Large',
            'message': 'File size exceeds maximum allowed limit',
            'status_code': 413
        }, 413

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limit exceeded errors."""
        return {
            'error': 'Too Many Requests',
            'message': 'Rate limit exceeded. Please try again later.',
            'status_code': 429
        }, 429

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors."""
        return {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }, 500

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        """Handle expired JWT tokens."""
        return {
            'error': 'Token Expired',
            'message': 'The token has expired',
            'status_code': 401
        }, 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        """Handle invalid JWT tokens."""
        return {
            'error': 'Invalid Token',
            'message': 'Token validation failed',
            'status_code': 401
        }, 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        """Handle missing JWT tokens."""
        return {
            'error': 'Authorization Required',
            'message': 'Request does not contain an access token',
            'status_code': 401
        }, 401

    @jwt.token_verification_failed_loader
    def token_verification_failed_callback(jwt_header, jwt_payload):
        """Handle JWT verification failures."""
        return {
            'error': 'Token Verification Failed',
            'message': 'Token verification failed',
            'status_code': 401
        }, 401


def init_logging(app):
    """Initialize logging configuration.

    Args:
        app: Flask application instance
    """
    import logging
    import os
    from logging.handlers import RotatingFileHandler

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(app.config.get('LOG_FILE', 'logs/app.log'))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    if not app.debug and not app.testing:
        # Production logging
        if app.config.get('LOG_FILE'):
            file_handler = RotatingFileHandler(
                app.config['LOG_FILE'],
                maxBytes=app.config.get('LOG_MAX_SIZE', 10485760),
                backupCount=app.config.get('LOG_BACKUP_COUNT', 5)
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(getattr(logging, app.config.get('LOG_LEVEL', 'INFO')))
            app.logger.addHandler(file_handler)
            app.logger.setLevel(getattr(logging, app.config.get('LOG_LEVEL', 'INFO')))

        # Disable default Flask logging to console in production
        app.logger.removeHandler(app.logger.handlers[0] if app.logger.handlers else None)

    # Always log to console in development
    if app.debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        app.logger.addHandler(console_handler)
        app.logger.setLevel(logging.DEBUG)