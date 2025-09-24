"""
Documentation Routes Module
Provides comprehensive API documentation routes and static file serving
"""

import os
from datetime import datetime
from flask import Blueprint, jsonify, render_template, send_from_directory, request, current_app
from flask_cors import cross_origin

# Create documentation blueprint
docs_bp = Blueprint('docs', __name__, url_prefix='/api/v1/docs')


def generate_openapi_spec():
    """Generate OpenAPI 3.0 specification dynamically.

    Returns:
        dict: OpenAPI specification
    """

    # Get all registered routes
    routes = []
    for rule in current_app.url_map.iter_rules():
        if rule.endpoint != 'static' and rule.rule.startswith('/api/'):
            routes.append({
                'path': rule.rule,
                'methods': list(rule.methods) if rule.methods else [],
                'endpoint': rule.endpoint
            })

    # Base URL configuration
    base_url = os.getenv('API_BASE_URL', request.host_url.rstrip('/'))

    spec = {
        'openapi': '3.0.3',
        'info': {
            'title': 'Flask TTS API',
            'description': '''
            # Flask TTS API Documentation

            A comprehensive Text-to-Speech API powered by Google Gemini AI.

            ## Features

            - **High-Quality Speech Synthesis**: Generate natural-sounding speech from text
            - **Multiple Voice Options**: Choose from various voice configurations
            - **Real-time Processing**: Fast audio generation with streaming support
            - **Batch Processing**: Handle multiple requests efficiently
            - **Plugin System**: Extensible architecture for custom enhancements
            - **Business Intelligence**: Advanced analytics and reporting
            - **Multi-tenancy**: Support for multiple organizations
            - **WebSocket Support**: Real-time audio streaming

            ## Authentication

            This API supports two authentication methods:

            1. **JWT Token**: Bearer token authentication
            2. **API Key**: Header-based authentication

            ## Getting Started

            1. Obtain authentication credentials
            2. Configure your client with the appropriate headers
            3. Start making requests to generate speech

            ## Support

            For support and questions, please contact the development team.
            ''',
            'version': '1.0.0',
            'contact': {
                'name': 'API Support Team',
                'email': 'support@example.com',
                'url': 'https://example.com/support'
            },
            'license': {
                'name': 'MIT License',
                'url': 'https://opensource.org/licenses/MIT'
            },
            'termsOfService': 'https://example.com/terms'
        },
        'servers': [
            {
                'url': base_url,
                'description': 'Current server'
            },
            {
                'url': 'https://api.example.com',
                'description': 'Production server'
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
                    'description': '''
                    JWT token authentication. Include the token in the Authorization header:
                    `Authorization: Bearer <your-jwt-token>`
                    '''
                },
                'ApiKeyAuth': {
                    'type': 'apiKey',
                    'in': 'header',
                    'name': 'X-API-Key',
                    'description': '''
                    API key authentication. Include the key in the X-API-Key header:
                    `X-API-Key: <your-api-key>`
                    '''
                }
            },
            'schemas': {
                'Error': {
                    'type': 'object',
                    'properties': {
                        'error': {
                            'type': 'string',
                            'description': 'Error type or code'
                        },
                        'message': {
                            'type': 'string',
                            'description': 'Human-readable error message'
                        },
                        'status_code': {
                            'type': 'integer',
                            'description': 'HTTP status code'
                        },
                        'timestamp': {
                            'type': 'string',
                            'format': 'date-time',
                            'description': 'Error timestamp'
                        },
                        'path': {
                            'type': 'string',
                            'description': 'Request path that caused the error'
                        }
                    }
                },
                'AudioRequest': {
                    'type': 'object',
                    'required': ['text'],
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Text content to convert to speech',
                            'example': 'Hello, this is a sample text to speech conversion.',
                            'minLength': 1,
                            'maxLength': 5000
                        },
                        'voice': {
                            'type': 'string',
                            'description': 'Voice configuration identifier',
                            'default': 'default',
                            'enum': ['default', 'female', 'male', 'neutral'],
                            'example': 'female'
                        },
                        'speed': {
                            'type': 'number',
                            'description': 'Speech speed multiplier',
                            'minimum': 0.5,
                            'maximum': 2.0,
                            'default': 1.0,
                            'example': 1.2
                        },
                        'language': {
                            'type': 'string',
                            'description': 'Language code (ISO 639-1)',
                            'default': 'en',
                            'example': 'en',
                            'enum': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh']
                        },
                        'format': {
                            'type': 'string',
                            'description': 'Output audio format',
                            'default': 'mp3',
                            'enum': ['mp3', 'wav', 'ogg', 'flac'],
                            'example': 'mp3'
                        },
                        'sample_rate': {
                            'type': 'integer',
                            'description': 'Audio sample rate in Hz',
                            'default': 22050,
                            'enum': [8000, 16000, 22050, 44100, 48000],
                            'example': 22050
                        }
                    }
                },
                'AudioResponse': {
                    'type': 'object',
                    'properties': {
                        'audio_url': {
                            'type': 'string',
                            'description': 'URL to the generated audio file',
                            'example': 'https://api.example.com/audio/12345.mp3'
                        },
                        'duration': {
                            'type': 'number',
                            'description': 'Audio duration in seconds',
                            'example': 3.45
                        },
                        'format': {
                            'type': 'string',
                            'description': 'Audio format',
                            'enum': ['mp3', 'wav', 'ogg', 'flac'],
                            'example': 'mp3'
                        },
                        'text_length': {
                            'type': 'integer',
                            'description': 'Length of input text in characters',
                            'example': 52
                        },
                        'processing_time': {
                            'type': 'number',
                            'description': 'Time taken to process the request in seconds',
                            'example': 1.23
                        },
                        'request_id': {
                            'type': 'string',
                            'description': 'Unique identifier for the request',
                            'example': 'req_12345'
                        }
                    }
                },
                'BatchAudioRequest': {
                    'type': 'object',
                    'required': ['requests'],
                    'properties': {
                        'requests': {
                            'type': 'array',
                            'items': {
                                '$ref': '#/components/schemas/AudioRequest'
                            },
                            'minItems': 1,
                            'maxItems': 100,
                            'description': 'Array of audio requests to process in batch'
                        },
                        'webhook_url': {
                            'type': 'string',
                            'description': 'Optional webhook URL for completion notification',
                            'example': 'https://example.com/webhook/batch-complete'
                        }
                    }
                },
                'HealthCheck': {
                    'type': 'object',
                    'properties': {
                        'status': {
                            'type': 'string',
                            'enum': ['healthy', 'degraded', 'unhealthy'],
                            'description': 'Overall system health status'
                        },
                        'timestamp': {
                            'type': 'string',
                            'format': 'date-time',
                            'description': 'Health check timestamp'
                        },
                        'version': {
                            'type': 'string',
                            'description': 'API version',
                            'example': '1.0.0'
                        },
                        'environment': {
                            'type': 'string',
                            'description': 'Deployment environment',
                            'enum': ['development', 'staging', 'production'],
                            'example': 'production'
                        },
                        'services': {
                            'type': 'object',
                            'description': 'Status of individual services',
                            'properties': {
                                'database': {'type': 'string', 'enum': ['up', 'down']},
                                'redis': {'type': 'string', 'enum': ['up', 'down']},
                                'gemini_api': {'type': 'string', 'enum': ['up', 'down']},
                                'storage': {'type': 'string', 'enum': ['up', 'down']}
                            }
                        }
                    }
                }
            }
        },
        'paths': {
            '/api/v1/health': {
                'get': {
                    'summary': 'Health Check',
                    'description': 'Check the overall health status of the API and its dependencies',
                    'tags': ['System'],
                    'responses': {
                        '200': {
                            'description': 'API is healthy',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/HealthCheck'
                                    },
                                    'example': {
                                        'status': 'healthy',
                                        'timestamp': '2024-01-15T10:30:00Z',
                                        'version': '1.0.0',
                                        'environment': 'production',
                                        'services': {
                                            'database': 'up',
                                            'redis': 'up',
                                            'gemini_api': 'up',
                                            'storage': 'up'
                                        }
                                    }
                                }
                            }
                        },
                        '503': {
                            'description': 'Service unavailable',
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
            },
            '/api/v1/tts/generate': {
                'post': {
                    'summary': 'Generate Audio',
                    'description': 'Convert text to speech using Google Gemini AI',
                    'tags': ['TTS'],
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
                                },
                                'examples': {
                                    'basic': {
                                        'summary': 'Basic text-to-speech',
                                        'value': {
                                            'text': 'Hello, world!',
                                            'voice': 'default',
                                            'speed': 1.0
                                        }
                                    },
                                    'advanced': {
                                        'summary': 'Advanced configuration',
                                        'value': {
                                            'text': 'This is a longer text with custom voice settings.',
                                            'voice': 'female',
                                            'speed': 1.2,
                                            'language': 'en',
                                            'format': 'mp3',
                                            'sample_rate': 22050
                                        }
                                    }
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
                                    },
                                    'example': {
                                        'audio_url': 'https://api.example.com/audio/12345.mp3',
                                        'duration': 2.34,
                                        'format': 'mp3',
                                        'text_length': 13,
                                        'processing_time': 0.89,
                                        'request_id': 'req_12345'
                                    }
                                }
                            }
                        },
                        '400': {
                            'description': 'Bad request - invalid input parameters',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/Error'
                                    },
                                    'example': {
                                        'error': 'VALIDATION_ERROR',
                                        'message': 'Text is required and must be between 1 and 5000 characters',
                                        'status_code': 400
                                    }
                                }
                            }
                        },
                        '401': {
                            'description': 'Unauthorized - invalid or missing authentication',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/Error'
                                    },
                                    'example': {
                                        'error': 'UNAUTHORIZED',
                                        'message': 'Invalid JWT token or API key',
                                        'status_code': 401
                                    }
                                }
                            }
                        },
                        '429': {
                            'description': 'Rate limit exceeded',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/Error'
                                    },
                                    'example': {
                                        'error': 'RATE_LIMIT_EXCEEDED',
                                        'message': 'Too many requests. Please try again later.',
                                        'status_code': 429
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
                                    },
                                    'example': {
                                        'error': 'INTERNAL_ERROR',
                                        'message': 'An unexpected error occurred during processing',
                                        'status_code': 500
                                    }
                                }
                            }
                        }
                    }
                }
            },
            '/api/v1/tts/batch': {
                'post': {
                    'summary': 'Batch Audio Generation',
                    'description': 'Process multiple text-to-speech requests in a single batch',
                    'tags': ['TTS', 'Batch'],
                    'security': [
                        {'JWT': []},
                        {'ApiKeyAuth': []}
                    ],
                    'requestBody': {
                        'required': True,
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/BatchAudioRequest'
                                }
                            }
                        }
                    },
                    'responses': {
                        '202': {
                            'description': 'Batch request accepted for processing',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'batch_id': {'type': 'string'},
                                            'status': {'type': 'string'},
                                            'estimated_completion': {'type': 'string', 'format': 'date-time'}
                                        }
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
                'name': 'System',
                'description': 'System health and monitoring endpoints'
            },
            {
                'name': 'TTS',
                'description': 'Text-to-speech generation endpoints'
            },
            {
                'name': 'Batch',
                'description': 'Batch processing endpoints'
            },
            {
                'name': 'Authentication',
                'description': 'Authentication and authorization endpoints'
            },
            {
                'name': 'Business Intelligence',
                'description': 'Analytics and reporting endpoints'
            }
        ]
    }

    return spec


@docs_bp.route('/')
@cross_origin()
def documentation_home():
    """Serve the main documentation page."""
    return render_template('docs.html')


@docs_bp.route('/swagger.json')
@cross_origin()
def swagger_spec():
    """Serve OpenAPI specification in JSON format."""
    try:
        spec = generate_openapi_spec()
        return jsonify(spec)
    except Exception as e:
        current_app.logger.error(f"Error generating OpenAPI spec: {str(e)}")
        return jsonify({
            'error': 'INTERNAL_ERROR',
            'message': 'Failed to generate API specification',
            'status_code': 500
        }), 500


@docs_bp.route('/openapi.json')
@cross_origin()
def openapi_spec():
    """Serve OpenAPI specification (alias for swagger.json)."""
    return swagger_spec()


@docs_bp.route('/openapi.yaml')
@cross_origin()
def openapi_yaml():
    """Serve OpenAPI specification in YAML format."""
    try:
        import yaml
        spec = generate_openapi_spec()
        yaml_content = yaml.dump(spec, default_flow_style=False, allow_unicode=True)

        response = current_app.response_class(
            response=yaml_content,
            status=200,
            mimetype='application/x-yaml'
        )
        response.headers['Content-Disposition'] = 'inline; filename="openapi.yaml"'
        return response
    except ImportError:
        return jsonify({
            'error': 'NOT_IMPLEMENTED',
            'message': 'YAML support not available. Install PyYAML to use this endpoint.',
            'status_code': 501
        }), 501
    except Exception as e:
        current_app.logger.error(f"Error generating YAML spec: {str(e)}")
        return jsonify({
            'error': 'INTERNAL_ERROR',
            'message': 'Failed to generate YAML specification',
            'status_code': 500
        }), 500


@docs_bp.route('/health')
def docs_health():
    """Health check for documentation endpoints."""
    return jsonify({
        'status': 'healthy',
        'service': 'documentation',
        'timestamp': datetime.utcnow().isoformat(),
        'endpoints': [
            '/api/v1/docs/',
            '/api/v1/docs/swagger.json',
            '/api/v1/docs/openapi.json',
            '/api/v1/docs/openapi.yaml'
        ]
    })


@docs_bp.route('/ui')
@cross_origin()
def swagger_ui():
    """Serve Swagger UI interface."""
    try:
        # Try to serve Swagger UI from static files
        static_dir = os.path.join(current_app.root_path, 'static', 'docs')
        ui_path = os.path.join(static_dir, 'swagger_ui.html')

        if os.path.exists(ui_path):
            with open(ui_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Replace the API URL in the HTML
            content = content.replace('"/api/v1/docs/swagger.json"', '"/api/v1/docs/swagger.json"')
            return content
        else:
            # Fallback to inline Swagger UI
            return generate_swagger_ui_html()
    except Exception as e:
        current_app.logger.error(f"Error serving Swagger UI: {str(e)}")
        return generate_swagger_ui_html()


def generate_swagger_ui_html():
    """Generate inline Swagger UI HTML."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flask TTS API - Swagger UI</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui.css" />
        <style>
            html {{
                box-sizing: border-box;
                overflow: -moz-scrollbars-vertical;
                overflow-y: scroll;
            }}
            *, *:before, *:after {{
                box-sizing: inherit;
            }}
            body {{
                margin:0;
                background: #fafafa;
            }}
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui-standalone-preset.js"></script>
        <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: '/api/v1/docs/swagger.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                requestInterceptor: (request) => {{
                    const token = localStorage.getItem('jwt_token');
                    const apiKey = localStorage.getItem('api_key');

                    if (token) {{
                        request.headers.Authorization = `Bearer ${{token}}`;
                    }}
                    if (apiKey) {{
                        request.headers['X-API-Key'] = apiKey;
                    }}

                    return request;
                }},
                responseInterceptor: (response) => {{
                    const token = response.headers.get('Authorization');
                    const apiKey = response.headers.get('X-API-Key');

                    if (token) {{
                        localStorage.setItem('jwt_token', token.replace('Bearer ', ''));
                    }}
                    if (apiKey) {{
                        localStorage.setItem('api_key', apiKey);
                    }}

                    return response;
                }}
            }});
        }};
        </script>
    </body>
    </html>
    """


@docs_bp.route('/static/<path:filename>')
def docs_static(filename):
    """Serve static files for documentation."""
    try:
        static_dir = os.path.join(current_app.root_path, 'static', 'docs')
        return send_from_directory(static_dir, filename)
    except FileNotFoundError:
        # Fallback to serving from templates directory
        templates_dir = os.path.join(current_app.root_path, '..', 'templates')
        return send_from_directory(templates_dir, filename)