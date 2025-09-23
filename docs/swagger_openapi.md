# API Documentation với Swagger/OpenAPI

## Tổng quan
Hệ thống API documentation sử dụng Swagger/OpenAPI để tạo interactive documentation, client SDK generation, và testing interface.

## Flask-Swagger Setup

### Installation và Configuration
```bash
pip install flask-swagger-ui flask-swagger
```

### Basic Setup
```python
# app/swagger.py
from flask_swagger_ui import get_swaggerui_blueprint
from flask_swagger import swagger

def create_swagger_blueprint(app):
    """Create Swagger UI blueprint"""

    # Swagger UI configuration
    SWAGGER_URL = '/api/v1/docs'
    API_URL = '/api/v1/swagger.json'

    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "TTS API",
            'docExpansion': 'list',
            'filter': True,
            'showExtensions': True,
            'showCommonExtensions': True,
            'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch'],
            'validatorUrl': None,
            'tagsSorter': 'alpha',
            'operationsSorter': 'alpha'
        }
    )

    return swaggerui_blueprint

def generate_swagger_spec(app):
    """Generate OpenAPI specification"""

    swag = swagger(app)
    swag['info'] = {
        'title': 'TTS API',
        'description': 'Text-to-Speech API using Google Gemini',
        'version': '1.0.0',
        'contact': {
            'name': 'API Support',
            'email': 'support@tts-api.com'
        },
        'license': {
            'name': 'MIT',
            'url': 'https://opensource.org/licenses/MIT'
        }
    }

    swag['host'] = 'api.tts-service.com'
    swag['basePath'] = '/api/v1'
    swag['schemes'] = ['https', 'http']

    # Security definitions
    swag['securityDefinitions'] = {
        'Bearer': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header',
            'description': 'JWT Authorization header using the Bearer scheme. Example: "Authorization: Bearer {token}"'
        },
        'APIKey': {
            'type': 'apiKey',
            'name': 'X-API-Key',
            'in': 'header',
            'description': 'API Key authentication'
        }
    }

    # Tags for organizing endpoints
    swag['tags'] = [
        {
            'name': 'Authentication',
            'description': 'User authentication and authorization'
        },
        {
            'name': 'TTS',
            'description': 'Text-to-Speech operations'
        },
        {
            'name': 'User Management',
            'description': 'User profile and settings'
        },
        {
            'name': 'Admin',
            'description': 'Administrative operations'
        },
        {
            'name': 'Health',
            'description': 'Health checks and monitoring'
        }
    ]

    return swag
```

## API Documentation Decorators

### Swagger Documentation Decorator
```python
# utils/swagger_docs.py
from functools import wraps
from flask import request
import json

def document_endpoint(summary: str, description: str = None,
                     tags: list = None, responses: dict = None,
                     parameters: list = None, security: list = None):
    """Decorator to document API endpoints"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add documentation metadata
        wrapper._swagger_summary = summary
        wrapper._swagger_description = description
        wrapper._swagger_tags = tags or []
        wrapper._swagger_responses = responses or {}
        wrapper._swagger_parameters = parameters or []
        wrapper._swagger_security = security or []

        return wrapper
    return decorator

def document_model(model_class):
    """Decorator to document Pydantic models"""
    model_class._swagger_model = True
    return model_class
```

## Documented Endpoints

### Authentication Endpoints
```python
# routes/auth_docs.py
from utils.swagger_docs import document_endpoint
from utils.validation.models import UserRegistrationSchema, UserLoginSchema

@auth_bp.route('/register', methods=['POST'])
@document_endpoint(
    summary='Register new user',
    description='Create a new user account with username, email, and password',
    tags=['Authentication'],
    responses={
        201: {
            'description': 'User created successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'user_id': {'type': 'integer'},
                    'api_key': {'type': 'string'}
                }
            }
        },
        400: {'description': 'Invalid input data'},
        409: {'description': 'User already exists'}
    },
    parameters=[{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': UserRegistrationSchema
    }]
)
def register():
    """Register endpoint implementation"""
    pass

@auth_bp.route('/login', methods=['POST'])
@document_endpoint(
    summary='User login',
    description='Authenticate user and return JWT tokens',
    tags=['Authentication'],
    responses={
        200: {
            'description': 'Login successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'access_token': {'type': 'string'},
                    'refresh_token': {'type': 'string'},
                    'user': {'$ref': '#/definitions/User'}
                }
            }
        },
        401: {'description': 'Invalid credentials'}
    },
    parameters=[{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'username': {'type': 'string'},
                'password': {'type': 'string'}
            },
            'required': ['username', 'password']
        }
    }]
)
def login():
    """Login endpoint implementation"""
    pass
```

### TTS Endpoints
```python
# routes/tts_docs.py
from utils.swagger_docs import document_endpoint
from utils.validation.models import TTSRequestSchema, PaginationSchema

@tts_bp.route('/generate', methods=['POST'])
@document_endpoint(
    summary='Generate audio from text',
    description='Convert text to speech using Google Gemini AI',
    tags=['TTS'],
    security=[{'Bearer': []}],
    responses={
        202: {
            'description': 'Audio generation started',
            'schema': {
                'type': 'object',
                'properties': {
                    'request_id': {'type': 'string'},
                    'status': {'type': 'string'},
                    'message': {'type': 'string'},
                    'estimated_time': {'type': 'string'}
                }
            }
        },
        400: {'description': 'Invalid request data'},
        401: {'description': 'Unauthorized'},
        429: {'description': 'Rate limit exceeded'}
    },
    parameters=[{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': TTSRequestSchema
    }]
)
def generate_audio():
    """TTS generation endpoint implementation"""
    pass

@tts_bp.route('/status/<int:request_id>', methods=['GET'])
@document_endpoint(
    summary='Get TTS request status',
    description='Check the status and progress of a TTS request',
    tags=['TTS'],
    security=[{'Bearer': []}],
    responses={
        200: {
            'description': 'Request status retrieved',
            'schema': {
                'type': 'object',
                'properties': {
                    'request_id': {'type': 'integer'},
                    'status': {'type': 'string'},
                    'progress': {'type': 'integer'},
                    'download_url': {'type': 'string'}
                }
            }
        },
        404: {'description': 'Request not found'}
    },
    parameters=[{
        'name': 'request_id',
        'in': 'path',
        'type': 'integer',
        'required': True,
        'description': 'TTS request ID'
    }]
)
def get_request_status(request_id):
    """Status check endpoint implementation"""
    pass

@tts_bp.route('/result/<int:request_id>', methods=['GET'])
@document_endpoint(
    summary='Download audio file',
    description='Download the generated audio file',
    tags=['TTS'],
    security=[{'Bearer': []}],
    responses={
        200: {
            'description': 'Audio file',
            'schema': {
                'type': 'file'
            }
        },
        404: {'description': 'File not found'},
        403: {'description': 'Access denied'}
    },
    parameters=[{
        'name': 'request_id',
        'in': 'path',
        'type': 'integer',
        'required': True,
        'description': 'TTS request ID'
    }],
    produces=['audio/wav', 'audio/mpeg', 'audio/ogg']
)
def download_audio(request_id):
    """Audio download endpoint implementation"""
    pass
```

## Model Documentation

### Pydantic Model Documentation
```python
# utils/swagger_docs.py
from pydantic import BaseModel
from typing import Optional, List
import json

def generate_model_schema(model_class):
    """Generate OpenAPI schema from Pydantic model"""
    schema = model_class.schema()

    # Convert Pydantic schema to OpenAPI format
    return {
        'type': 'object',
        'properties': schema.get('properties', {}),
        'required': schema.get('required', [])
    }

def document_models():
    """Generate documentation for all models"""
    models = {}

    # User model
    models['User'] = {
        'type': 'object',
        'properties': {
            'id': {'type': 'integer', 'description': 'User ID'},
            'username': {'type': 'string', 'description': 'Username'},
            'email': {'type': 'string', 'description': 'Email address'},
            'is_premium': {'type': 'boolean', 'description': 'Premium user status'},
            'created_at': {'type': 'string', 'format': 'date-time', 'description': 'Account creation date'}
        }
    }

    # TTS Request model
    models['TTSRequest'] = generate_model_schema(TTSRequestSchema)

    # Audio Request model
    models['AudioRequest'] = {
        'type': 'object',
        'properties': {
            'id': {'type': 'integer', 'description': 'Request ID'},
            'user_id': {'type': 'integer', 'description': 'User ID'},
            'text_content': {'type': 'string', 'description': 'Original text'},
            'voice_name': {'type': 'string', 'description': 'Voice used'},
            'output_format': {'type': 'string', 'description': 'Audio format'},
            'status': {'type': 'string', 'description': 'Request status'},
            'created_at': {'type': 'string', 'format': 'date-time'},
            'updated_at': {'type': 'string', 'format': 'date-time'}
        }
    }

    return models
```

## Advanced Documentation Features

### Response Examples
```python
# utils/swagger_docs.py
def add_response_examples():
    """Add response examples to documentation"""
    examples = {
        'TTSRequestSuccess': {
            'summary': 'Successful TTS request',
            'value': {
                'request_id': 12345,
                'status': 'queued',
                'message': 'Audio generation started',
                'estimated_time': '10-30 seconds'
            }
        },
        'TTSRequestError': {
            'summary': 'TTS request error',
            'value': {
                'error': 'VALIDATION_ERROR',
                'message': 'Text is too long',
                'details': {
                    'text': 'Text must be less than 5000 characters'
                }
            }
        },
        'UserLoginSuccess': {
            'summary': 'Successful login',
            'value': {
                'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...',
                'refresh_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...',
                'user': {
                    'id': 1,
                    'username': 'testuser',
                    'email': 'test@example.com',
                    'is_premium': false
                }
            }
        }
    }
    return examples
```

### Parameter Documentation
```python
# utils/swagger_docs.py
def document_parameters():
    """Document common parameters"""
    parameters = {
        'AuthorizationHeader': {
            'name': 'Authorization',
            'in': 'header',
            'type': 'string',
            'required': True,
            'description': 'JWT token (Bearer)',
            'default': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
        },
        'APIKeyHeader': {
            'name': 'X-API-Key',
            'in': 'header',
            'type': 'string',
            'required': True,
            'description': 'API key for authentication',
            'default': 'sk-abc123...'
        },
        'RequestID': {
            'name': 'request_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'TTS request ID'
        },
        'PaginationParams': {
            'name': 'pagination',
            'in': 'query',
            'type': 'object',
            'properties': {
                'page': {'type': 'integer', 'minimum': 1, 'default': 1},
                'per_page': {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 10},
                'sort_by': {'type': 'string', 'default': 'created_at'},
                'sort_order': {'type': 'string', 'enum': ['asc', 'desc'], 'default': 'desc'}
            }
        }
    }
    return parameters
```

## Interactive Documentation

### Swagger UI Customization
```python
# app/swagger_customization.py
def customize_swagger_ui():
    """Customize Swagger UI appearance and behavior"""

    custom_css = """
    .swagger-ui .topbar { display: none }
    .swagger-ui .info {
        margin: 20px 0;
    }
    .swagger-ui .info .title {
        color: #2E86AB;
        font-size: 36px;
    }
    .swagger-ui .info .description {
        font-size: 16px;
    }
    .swagger-ui .btn.authorize {
        background-color: #2E86AB;
    }
    .swagger-ui .btn.execute {
        background-color: #A23B72;
    }
    """

    swagger_config = {
        'validatorUrl': None,
        'docExpansion': 'list',
        'filter': True,
        'showExtensions': True,
        'showCommonExtensions': True,
        'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch'],
        'tagsSorter': 'alpha',
        'operationsSorter': 'alpha',
        'defaultModelRendering': 'schema',
        'defaultModelsExpandDepth': 3,
        'defaultModelExpandDepth': 3,
        'tryItOutEnabled': True,
        'requestInterceptor': """
            request => {
                // Add authentication if available
                const token = localStorage.getItem('access_token');
                if (token) {
                    request.headers.Authorization = `Bearer ${token}`;
                }
                return request;
            }
        """,
        'responseInterceptor': """
            response => {
                // Log responses for debugging
                console.log('API Response:', response);
                return response;
            }
        """
    }

    return custom_css, swagger_config
```

### API Testing Interface
```python
# utils/swagger_docs.py
def add_testing_features():
    """Add testing features to Swagger UI"""

    testing_features = {
        'tryItOutEnabled': True,
        'requestInterceptor': """
            (request) => {
                // Add common headers
                request.headers['Content-Type'] = 'application/json';

                // Add authentication from localStorage
                const token = localStorage.getItem('jwt_token');
                if (token && !request.headers.Authorization) {
                    request.headers.Authorization = `Bearer ${token}`;
                }

                return request;
            }
        """,
        'responseInterceptor': """
            (response) => {
                // Store token if received
                if (response.obj && response.obj.access_token) {
                    localStorage.setItem('jwt_token', response.obj.access_token);
                }

                // Log for debugging
                console.log('Response:', response);

                return response;
            }
        """,
        'onComplete': """
            () => {
                // Add custom functionality
                console.log('API documentation loaded');
            }
        """
    }

    return testing_features
```

## Client SDK Generation

### OpenAPI Generator Setup
```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Generate Python client
openapi-generator-cli generate \
    -i swagger.json \
    -g python \
    -o client/python \
    --package-name tts-api-client

# Generate JavaScript client
openapi-generator-cli generate \
    -i swagger.json \
    -g javascript \
    -o client/javascript \
    --package-name tts-api-client

# Generate TypeScript client
openapi-generator-cli generate \
    -i swagger.json \
    -g typescript-axios \
    -o client/typescript \
    --package-name tts-api-client
```

### Custom Client Templates
```yaml
# openapi-generator-config.yaml
packageName: tts-api-client
packageVersion: 1.0.0
packageDescription: TTS API Client
packageUrl: https://github.com/company/tts-api-client

files:
  ApiClient.mustache:
    templateType: ApiClient
    destinationFilename: ApiClient.py

  Configuration.mustache:
    templateType: SupportingFiles
    destinationFilename: configuration.py

globalProperties:
  apiTests: false
  modelTests: false
  apiDocs: false
  modelDocs: false
```

## Documentation Deployment

### Static Documentation
```python
# app/docs_generator.py
import json
from flask import current_app

def generate_static_docs():
    """Generate static API documentation"""

    # Generate OpenAPI spec
    swag = generate_swagger_spec(current_app)

    # Save to file
    with open('static/swagger.json', 'w') as f:
        json.dump(swag, f, indent=2)

    # Generate HTML documentation
    generate_html_docs(swag)

def generate_html_docs(swag):
    """Generate HTML documentation from OpenAPI spec"""

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TTS API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css">
        <style>
            html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
            *, *:before, *:after { box-sizing: inherit; }
            body { margin: 0; background: #fafafa; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: '/static/swagger.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
                    plugins: [SwaggerUIBundle.plugins.DownloadUrl],
                    layout: "StandaloneLayout",
                    validatorUrl: null,
                    tryItOutEnabled: true
                });
            };
        </script>
    </body>
    </html>
    """

    with open('static/docs.html', 'w') as f:
        f.write(html_template)
```

### Documentation Routes
```python
# routes/docs.py
from flask import Blueprint, jsonify, send_file, render_template
import json
import os

docs_bp = Blueprint('docs', __name__)

@docs_bp.route('/swagger.json')
def swagger_json():
    """Serve OpenAPI specification"""
    from app.swagger import generate_swagger_spec
    from flask import current_app

    swag = generate_swagger_spec(current_app)
    return jsonify(swag)

@docs_bp.route('/docs')
def api_docs():
    """Serve interactive API documentation"""
    return render_template('docs.html')

@docs_bp.route('/docs/static/<path:filename>')
def docs_static(filename):
    """Serve static documentation files"""
    return send_file(os.path.join('static', filename))

@docs_bp.route('/redoc')
def redoc_docs():
    """Serve ReDoc documentation"""
    return render_template('redoc.html')

@docs_bp.route('/rapidoc')
def rapidoc_docs():
    """Serve RapiDoc documentation"""
    return render_template('rapidoc.html')
```

## Testing và Validation

### Documentation Tests
```python
# tests/test_docs.py
import json
from app.swagger import generate_swagger_spec

def test_swagger_spec_generation(app):
    """Test OpenAPI specification generation"""
    with app.app_context():
        swag = generate_swagger_spec(app)

        # Check basic structure
        assert 'info' in swag
        assert 'paths' in swag
        assert 'components' in swag

        # Check info section
        assert swag['info']['title'] == 'TTS API'
        assert swag['info']['version'] == '1.0.0'

        # Check security definitions
        assert 'securityDefinitions' in swag
        assert 'Bearer' in swag['securityDefinitions']

def test_endpoint_documentation(app):
    """Test endpoint documentation"""
    with app.test_client() as client:
        response = client.get('/api/v1/swagger.json')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert '/api/v1/auth/login' in data['paths']
        assert '/api/v1/tts/generate' in data['paths']

def test_documentation_accessibility(app):
    """Test documentation accessibility"""
    with app.test_client() as client:
        # Test Swagger UI
        response = client.get('/api/v1/docs')
        assert response.status_code == 200

        # Test OpenAPI spec
        response = client.get('/api/v1/swagger.json')
        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'application/json'
```

### Documentation Validation
```python
# utils/docs_validator.py
import json
import requests
from typing import Dict, Any, List

class DocumentationValidator:
    """Validate API documentation"""

    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.spec = self._load_spec()

    def _load_spec(self) -> Dict[str, Any]:
        """Load OpenAPI specification"""
        response = requests.get(self.swagger_url)
        response.raise_for_status()
        return response.json()

    def validate_structure(self) -> List[str]:
        """Validate basic structure"""
        errors = []

        required_fields = ['info', 'paths', 'swagger']
        for field in required_fields:
            if field not in self.spec:
                errors.append(f"Missing required field: {field}")

        if 'info' in self.spec:
            info = self.spec['info']
            if 'title' not in info:
                errors.append("Missing info.title")
            if 'version' not in info:
                errors.append("Missing info.version")

        return errors

    def validate_endpoints(self) -> List[str]:
        """Validate endpoint documentation"""
        errors = []

        if 'paths' not in self.spec:
            return ["No paths defined"]

        for path, path_item in self.spec['paths'].items():
            for method, operation in path_item.items():
                if method.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    continue

                # Check for required operation fields
                if 'responses' not in operation:
                    errors.append(f"Missing responses for {method} {path}")

                if 'summary' not in operation:
                    errors.append(f"Missing summary for {method} {path}")

        return errors

    def validate_models(self) -> List[str]:
        """Validate model definitions"""
        errors = []

        if 'definitions' in self.spec:
            for model_name, model_def in self.spec['definitions'].items():
                if 'type' not in model_def:
                    errors.append(f"Model {model_name} missing type")
                if 'properties' not in model_def:
                    errors.append(f"Model {model_name} missing properties")

        return errors

    def validate_security(self) -> List[str]:
        """Validate security definitions"""
        errors = []

        if 'securityDefinitions' not in self.spec:
            errors.append("Missing security definitions")

        if 'Bearer' not in self.spec.get('securityDefinitions', {}):
            errors.append("Missing Bearer token security definition")

        return errors

    def run_all_validations(self) -> Dict[str, List[str]]:
        """Run all validations"""
        return {
            'structure': self.validate_structure(),
            'endpoints': self.validate_endpoints(),
            'models': self.validate_models(),
            'security': self.validate_security()
        }
```

## Configuration

### Documentation Settings
```python
# config/docs_config.py
class DocumentationConfig:
    """Documentation configuration"""

    # Basic settings
    API_TITLE = "TTS API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Text-to-Speech API using Google Gemini"

    # Server settings
    API_HOST = "api.tts-service.com"
    API_SCHEMES = ["https", "http"]

    # Documentation URLs
    SWAGGER_URL = "/api/v1/docs"
    API_SPEC_URL = "/api/v1/swagger.json"

    # Features
    ENABLE_TRY_IT_OUT = True
    ENABLE_AUTHORIZATION = True
    ENABLE_FILTERING = True
    ENABLE_VALIDATION = True

    # Customization
    CUSTOM_CSS = """
    .swagger-ui .topbar { display: none }
    .swagger-ui .info .title { color: #2E86AB; }
    """

    # Contact information
    CONTACT_NAME = "API Support"
    CONTACT_EMAIL = "support@tts-api.com"
    CONTACT_URL = "https://tts-api.com/support"

    # License
    LICENSE_NAME = "MIT"
    LICENSE_URL = "https://opensource.org/licenses/MIT"

    # Terms of service
    TERMS_OF_SERVICE = "https://tts-api.com/terms"
```

### Environment Variables
```bash
# Documentation settings
API_TITLE="TTS API"
API_VERSION="1.0.0"
API_HOST="api.tts-service.com"
ENABLE_SWAGGER=true
SWAGGER_URL="/api/v1/docs"
ENABLE_REDOC=true
ENABLE_RAPIDOC=true
```

## Best Practices

### 1. Documentation Standards
- Use clear, descriptive summaries
- Provide detailed descriptions
- Include request/response examples
- Document all parameters and responses
- Use consistent naming conventions

### 2. Security Documentation
- Document authentication methods
- Explain security requirements
- Provide examples of secure usage
- Document rate limiting
- Include error codes and messages

### 3. User Experience
- Organize endpoints by functionality
- Use clear parameter names
- Provide realistic examples
- Include error scenarios
- Document status codes

### 4. Maintenance
- Keep documentation in sync with code
- Use automated validation
- Generate client SDKs
- Version documentation
- Archive old versions

### 5. Testing
- Test all documented endpoints
- Validate examples
- Check error responses
- Verify security requirements
- Test interactive features

## Deployment

### Production Setup
```python
# Production documentation configuration
PRODUCTION_DOCS_CONFIG = {
    'api_host': 'https://api.tts-service.com',
    'enable_try_it_out': False,  # Disable in production
    'enable_validation': True,
    'custom_css': PRODUCTION_CSS,
    'terms_of_service': 'https://tts-api.com/terms',
    'contact_email': 'api@tts-service.com'
}
```

### CDN Integration
```html
<!-- CDN hosted Swagger UI -->
<link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css">
<script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
<script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-standalone-preset.js"></script>
```

### Docker Configuration
```dockerfile
# Dockerfile with documentation
FROM python:3.11-slim

# Install documentation dependencies
RUN pip install flask-swagger-ui flask-swagger

# Copy documentation files
COPY docs/ /app/docs/
COPY static/ /app/static/

# Expose documentation port
EXPOSE 5000

CMD ["python", "app/main.py"]
```

## Summary

API documentation system provides:

1. **Interactive Documentation**: Swagger UI for testing and exploration
2. **Client SDK Generation**: Automatic client library generation
3. **Comprehensive Coverage**: Complete endpoint and model documentation
4. **Security Documentation**: Authentication and authorization details
5. **Testing Interface**: Built-in API testing capabilities

Key components:
- Flask-Swagger for OpenAPI generation
- Swagger UI for interactive documentation
- Custom decorators for endpoint documentation
- Model documentation with Pydantic integration
- Client SDK generation
- Documentation validation and testing