import requests
import json
import time

# Configuration
BASE_URL = 'http://localhost:5000'
TIMEOUT = 10  # seconds

# Test all documentation endpoints
endpoints = [
    ('/api/v1/docs/', 'Custom docs template'),
    ('/api/v1/docs/ui', 'Swagger UI'),
    ('/api/v1/docs/swagger.json', 'OpenAPI spec'),
    ('/api/v1/docs/openapi.json', 'OpenAPI spec (alias)'),
    ('/api/v1/docs/health', 'Docs health check')
]

def test_endpoint(endpoint, description):
    """Test a single endpoint and return results."""
    try:
        start_time = time.time()
        r = requests.get(f'{BASE_URL}{endpoint}', timeout=TIMEOUT)
        elapsed = time.time() - start_time
        status = f'{r.status_code} ({elapsed:.2f}s)'

        print(f'{endpoint:30} | {status:15} | {description}')

        if 'swagger.json' in endpoint or 'openapi.json' in endpoint:
            if r.status_code == 200:
                spec = r.json()
                openapi_version = spec.get('openapi', 'N/A')
                title = spec.get('info', {}).get('title', 'N/A')
                description = spec.get('info', {}).get('description', 'N/A')[:50] + '...'
                paths_count = len(spec.get('paths', {}))
                print(f'{"":30} | {"":15} | OpenAPI version: {openapi_version}')
                print(f'{"":30} | {"":15} | Title: {title}')
                print(f'{"":30} | {"":15} | Description: {description}')
                print(f'{"":30} | {"":15} | Paths count: {paths_count}')
            else:
                print(f'{"":30} | {"":15} | Failed to fetch spec: {r.status_code}')

        return r.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f'{endpoint:30} | ERR           | {str(e)}')
        return False
    except json.JSONDecodeError:
        print(f'{endpoint:30} | ERR           | Invalid JSON response')
        return False

print('Testing all documentation endpoints:')
print('=' * 60)

success_count = 0
total_endpoints = len(endpoints)

for endpoint, description in endpoints:
    if test_endpoint(endpoint, description):
        success_count += 1

print('-' * 60)

# Test template content
print('Testing docs template content:')
r = requests.get(f'{BASE_URL}/api/v1/docs/', timeout=TIMEOUT)
if r.status_code == 200:
    content = r.text
    checks = [
        ('Flask TTS API Documentation', 'Title found'),
        ('Authentication Setup', 'Auth section found'),
        ('Interactive Testing', 'Features section found'),
        ('Open Swagger UI Documentation', 'Link to Swagger UI found'),
        ('Endpoints', 'Endpoints section found'),
        ('Examples', 'Examples section found'),
        ('<title>', 'HTML title tag found'),
        ('swagger-ui', 'Swagger UI integration found')
    ]

    check_results = []
    for check, desc in checks:
        found = check in content
        status = 'YES' if found else 'NO'
        check_results.append((desc, status))
        print(f'{desc:30} | {status}')

    # Summary of content checks
    passed_checks = sum(1 for _, status in check_results if status == 'YES')
    print(f'{"Content checks passed""30"} | {passed_checks}/{len(checks)}')
else:
    print(f'Docs endpoint failed: {r.status_code}')

print('=' * 60)
print(f'Testing complete. Endpoints successful: {success_count}/{total_endpoints}')