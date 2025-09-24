"""
Swagger/OpenAPI Configuration Module
Provides comprehensive API documentation setup with security definitions
"""

import os
from datetime import datetime
from flask import Blueprint, jsonify, render_template_string, request, current_app
from flask_swagger_ui import get_swaggerui_blueprint
# Note: Using Flask-Swagger-UI only, manual OpenAPI spec generation


def create_swagger_blueprint(app):
    """Create and configure Swagger UI blueprint.

    Args:
        app: Flask application instance

    Returns:
        Configured Swagger UI blueprint
    """

    # Swagger UI configuration
    SWAGGER_URL = '/api/v1/docs/ui'
    API_URL = '/api/v1/docs/swagger.json'

    swagger_ui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Flask TTS API",
            'docExpansion': 'list',
            'filter': True,
            'showExtensions': True,
            'showCommonExtensions': True,
            'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch'],
            'tagsSorter': 'alpha',
            'operationsSorter': 'alpha',
            'validatorUrl': None,
            'layout': 'StandaloneLayout',
            'tryItOutEnabled': True,
            'requestInterceptor': """
                (request) => {
                    // Add authentication headers if available
                    const token = localStorage.getItem('jwt_token');
                    const apiKey = localStorage.getItem('api_key');

                    if (token) {
                        request.headers.Authorization = `Bearer ${token}`;
                    }
                    if (apiKey) {
                        request.headers['X-API-Key'] = apiKey;
                    }

                    return request;
                }
            """,
            'responseInterceptor': """
                (response) => {
                    // Store authentication tokens from responses
                    const token = response.headers.get('Authorization');
                    const apiKey = response.headers.get('X-API-Key');

                    if (token) {
                        localStorage.setItem('jwt_token', token.replace('Bearer ', ''));
                    }
                    if (apiKey) {
                        localStorage.setItem('api_key', apiKey);
                    }

                    return response;
                }
            """
        }
    )

    return swagger_ui_blueprint


def generate_openapi_spec(app):
    """Generate OpenAPI 3.0 specification.

    Args:
        app: Flask application instance

    Returns:
        OpenAPI specification dictionary
    """

    # Get all registered routes
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            routes.append({
                'path': rule.rule,
                'methods': list(rule.methods),
                'endpoint': rule.endpoint
            })

    # Generate OpenAPI spec
    spec = {
        'openapi': '3.0.3',
        'info': {
            'title': 'Flask TTS API',
            'description': 'Text-to-Speech API using Google Gemini AI',
            'version': '1.0.0',
            'contact': {
                'name': 'API Support',
                'email': 'support@example.com'
            },
            'license': {
                'name': 'MIT',
                'url': 'https://opensource.org/licenses/MIT'
            }
        },
        'servers': [
            {
                'url': f"{app.config.get('API_BASE_URL', 'http://localhost:5000')}",
                'description': 'Development server'
            }
        ],
        'security': [
            {
                'JWT': []
            },
            {
                'ApiKeyAuth': []
            }
        ],
        'components': {
            'securitySchemes': {
                'JWT': {
                    'type': 'http',
                    'scheme': 'bearer',
                    'bearerFormat': 'JWT',
                    'description': 'JWT token for authentication'
                },
                'ApiKeyAuth': {
                    'type': 'apiKey',
                    'in': 'header',
                    'name': 'X-API-Key',
                    'description': 'API key for authentication'
                }
            },
            'schemas': {
                'Error': {
                    'type': 'object',
                    'properties': {
                        'error': {
                            'type': 'string',
                            'description': 'Error type'
                        },
                        'message': {
                            'type': 'string',
                            'description': 'Error message'
                        },
                        'status_code': {
                            'type': 'integer',
                            'description': 'HTTP status code'
                        }
                    }
                },
                'AudioRequest': {
                    'type': 'object',
                    'required': ['text'],
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Text to convert to speech'
                        },
                        'voice': {
                            'type': 'string',
                            'description': 'Voice configuration',
                            'default': 'default'
                        },
                        'speed': {
                            'type': 'number',
                            'description': 'Speech speed multiplier',
                            'minimum': 0.5,
                            'maximum': 2.0,
                            'default': 1.0
                        },
                        'language': {
                            'type': 'string',
                            'description': 'Language code',
                            'default': 'en'
                        }
                    }
                },
                'AudioResponse': {
                    'type': 'object',
                    'properties': {
                        'audio_url': {
                            'type': 'string',
                            'description': 'URL to generated audio file'
                        },
                        'duration': {
                            'type': 'number',
                            'description': 'Audio duration in seconds'
                        },
                        'format': {
                            'type': 'string',
                            'description': 'Audio format',
                            'enum': ['mp3', 'wav', 'ogg']
                        },
                        'text_length': {
                            'type': 'integer',
                            'description': 'Length of input text'
                        }
                    }
                },
                'HealthCheck': {
                    'type': 'object',
                    'properties': {
                        'status': {
                            'type': 'string',
                            'enum': ['healthy', 'unhealthy']
                        },
                        'timestamp': {
                            'type': 'string',
                            'format': 'date-time'
                        },
                        'version': {
                            'type': 'string'
                        },
                        'environment': {
                            'type': 'string'
                        }
                    }
                }
            }
        },
        'paths': {
            '/api/v1/health': {
                'get': {
                    'summary': 'Health Check',
                    'description': 'Check API health status',
                    'responses': {
                        '200': {
                            'description': 'API is healthy',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/HealthCheck'
                                    }
                                }
                            }
                        }
                    }
                }
            },
            '/api/v1/tts/generate': {
                'post': {
                    'summary': 'Generate Audio',
                    'description': 'Convert text to speech',
                    'security': [
                        {'JWT': []},
                        {'ApiKeyAuth': []}
                    ],
                    'requestBody': {
                        'required': True,
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/AudioRequest'
                                }
                            }
                        }
                    },
                    'responses': {
                        '200': {
                            'description': 'Audio generated successfully',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/AudioResponse'
                                    }
                                }
                            }
                        },
                        '400': {
                            'description': 'Bad request',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/Error'
                                    }
                                }
                            }
                        },
                        '401': {
                            'description': 'Unauthorized',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/Error'
                                    }
                                }
                            }
                        },
                        '500': {
                            'description': 'Internal server error',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/Error'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        'tags': [
            {
                'name': 'health',
                'description': 'Health check operations'
            },
            {
                'name': 'tts',
                'description': 'Text-to-speech operations'
            },
            {
                'name': 'auth',
                'description': 'Authentication operations'
            },
            {
                'name': 'bi',
                'description': 'Business intelligence operations'
            }
        ]
    }

    return spec




def init_swagger(app):
    """Initialize Swagger documentation.

    Args:
        app: Flask application instance
    """

    # Create and register Swagger UI blueprint
    swagger_ui_blueprint = create_swagger_blueprint(app)
    app.register_blueprint(swagger_ui_blueprint, url_prefix='/api/v1')

    app.logger.info('Swagger documentation initialized successfully')