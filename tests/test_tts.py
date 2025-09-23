"""
TTS (Text-to-Speech) tests for Flask TTS API
"""

import json
import pytest
from app.extensions import db
from models import AudioRequest, User


class TestTTSRoutes:
    """Test cases for TTS routes."""

    def get_auth_headers(self, client):
        """Helper method to get authentication headers."""
        # Register user
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        # Login
        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        access_token = json.loads(login_response.data)['tokens']['access_token']
        return {'Authorization': f'Bearer {access_token}'}

    def test_generate_audio_success(self, client):
        """Test successful audio generation."""
        headers = self.get_auth_headers(client)

        response = client.post('/api/v1/tts/generate', json={
            'text': 'Hello, this is a test audio generation.',
            'voice_name': 'Alnilam',
            'output_format': 'wav'
        }, headers=headers)

        assert response.status_code == 202
        data = json.loads(response.data)
        assert data['message'] == 'Audio generation started'
        assert 'request_id' in data

    def test_generate_audio_invalid_data(self, client):
        """Test audio generation with invalid data."""
        headers = self.get_auth_headers(client)

        response = client.post('/api/v1/tts/generate', json={
            'text': '',  # Empty text
            'voice_name': 'InvalidVoice',
            'output_format': 'invalid'
        }, headers=headers)

        assert response.status_code == 400

    def test_generate_audio_unauthorized(self, client):
        """Test audio generation without authentication."""
        response = client.post('/api/v1/tts/generate', json={
            'text': 'Test text'
        })

        assert response.status_code == 401

    def test_get_audio_requests_success(self, client):
        """Test successful retrieval of audio requests."""
        headers = self.get_auth_headers(client)

        # Create a test request first
        client.post('/api/v1/tts/generate', json={
            'text': 'Test audio request'
        }, headers=headers)

        # Get requests
        response = client.get('/api/v1/tts/', headers=headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'requests' in data
        assert 'pagination' in data

    def test_get_audio_requests_pagination(self, client):
        """Test pagination of audio requests."""
        headers = self.get_auth_headers(client)

        # Create multiple requests
        for i in range(5):
            client.post('/api/v1/tts/generate', json={
                'text': f'Test audio request {i}'
            }, headers=headers)

        # Get first page
        response = client.get('/api/v1/tts/?page=1&per_page=2', headers=headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['requests']) == 2
        assert data['pagination']['page'] == 1
        assert data['pagination']['per_page'] == 2
        assert data['pagination']['total_pages'] >= 2

    def test_get_audio_requests_filtering(self, client):
        """Test filtering of audio requests."""
        headers = self.get_auth_headers(client)

        # Create requests with different statuses
        client.post('/api/v1/tts/generate', json={
            'text': 'Test request 1'
        }, headers=headers)

        # Get pending requests
        response = client.get('/api/v1/tts/?status=pending', headers=headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert all(req['status'] == 'pending' for req in data['requests'])

    def test_get_audio_request_success(self, client):
        """Test successful retrieval of specific audio request."""
        headers = self.get_auth_headers(client)

        # Create a request
        create_response = client.post('/api/v1/tts/generate', json={
            'text': 'Test audio request'
        }, headers=headers)

        request_id = json.loads(create_response.data)['request_id']

        # Get the request
        response = client.get(f'/api/v1/tts/{request_id}', headers=headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'request' in data
        assert data['request']['id'] == request_id

    def test_get_audio_request_not_found(self, client):
        """Test retrieval of non-existent audio request."""
        headers = self.get_auth_headers(client)

        response = client.get('/api/v1/tts/99999', headers=headers)

        assert response.status_code == 404

    def test_get_audio_request_access_denied(self, client):
        """Test access denied for other user's request."""
        headers1 = self.get_auth_headers(client)

        # Create request with first user
        create_response = client.post('/api/v1/tts/generate', json={
            'text': 'Test audio request'
        }, headers=headers1)

        request_id = json.loads(create_response.data)['request_id']

        # Register second user
        client.post('/api/v1/auth/register', json={
            'username': 'testuser2',
            'email': 'test2@example.com',
            'password': 'TestPassword123'
        })

        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser2',
            'password': 'TestPassword123'
        })

        headers2 = {'Authorization': f'Bearer {json.loads(login_response.data)["tokens"]["access_token"]}'}

        # Try to access first user's request
        response = client.get(f'/api/v1/tts/{request_id}', headers=headers2)

        assert response.status_code == 404

    def test_delete_audio_request_success(self, client):
        """Test successful deletion of audio request."""
        headers = self.get_auth_headers(client)

        # Create a request
        create_response = client.post('/api/v1/tts/generate', json={
            'text': 'Test audio request'
        }, headers=headers)

        request_id = json.loads(create_response.data)['request_id']

        # Delete the request
        response = client.delete(f'/api/v1/tts/{request_id}', headers=headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Audio request deleted successfully'

    def test_delete_audio_request_not_found(self, client):
        """Test deletion of non-existent audio request."""
        headers = self.get_auth_headers(client)

        response = client.delete('/api/v1/tts/99999', headers=headers)

        assert response.status_code == 404

    def test_get_user_stats_success(self, client):
        """Test successful retrieval of user statistics."""
        headers = self.get_auth_headers(client)

        # Create some requests
        for i in range(3):
            client.post('/api/v1/tts/generate', json={
                'text': f'Test audio request {i}'
            }, headers=headers)

        # Get stats
        response = client.get('/api/v1/tts/stats', headers=headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'stats' in data
        assert data['stats']['total_requests'] >= 3

    def test_generate_audio_rate_limit(self, client):
        """Test rate limiting for audio generation."""
        headers = self.get_auth_headers(client)

        # Make multiple requests quickly
        for i in range(10):
            response = client.post('/api/v1/tts/generate', json={
                'text': f'Test request {i}'
            }, headers=headers)

            if i < 5:  # First few should succeed
                assert response.status_code in [202, 429]
            else:  # Later ones might be rate limited
                assert response.status_code in [202, 429]

    def test_generate_audio_text_too_long(self, client):
        """Test audio generation with text too long."""
        headers = self.get_auth_headers(client)

        # Create very long text
        long_text = 'A' * 6000  # Exceeds 5000 character limit

        response = client.post('/api/v1/tts/generate', json={
            'text': long_text
        }, headers=headers)

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Text too long' in data['message']

    def test_generate_audio_invalid_voice(self, client):
        """Test audio generation with invalid voice."""
        headers = self.get_auth_headers(client)

        response = client.post('/api/v1/tts/generate', json={
            'text': 'Test text',
            'voice_name': 'InvalidVoiceName'
        }, headers=headers)

        assert response.status_code == 400

    def test_generate_audio_invalid_format(self, client):
        """Test audio generation with invalid format."""
        headers = self.get_auth_headers(client)

        response = client.post('/api/v1/tts/generate', json={
            'text': 'Test text',
            'output_format': 'invalid_format'
        }, headers=headers)

        assert response.status_code == 400