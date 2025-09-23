"""
Authentication tests for Flask TTS API
"""

import json
import pytest
from app.extensions import db
from models import User


class TestAuthRoutes:
    """Test cases for authentication routes."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['message'] == 'User registered successfully'
        assert 'user' in data
        assert 'api_key' in data
        assert 'tokens' in data

    def test_register_duplicate_username(self, client):
        """Test registration with duplicate username."""
        # Register first user
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test1@example.com',
            'password': 'TestPassword123'
        })

        # Try to register with same username
        response = client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test2@example.com',
            'password': 'TestPassword123'
        })

        assert response.status_code == 409
        data = json.loads(response.data)
        assert 'Username already exists' in data['message']

    def test_register_duplicate_email(self, client):
        """Test registration with duplicate email."""
        # Register first user
        client.post('/api/v1/auth/register', json={
            'username': 'testuser1',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        # Try to register with same email
        response = client.post('/api/v1/auth/register', json={
            'username': 'testuser2',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        assert response.status_code == 409
        data = json.loads(response.data)
        assert 'Email already exists' in data['message']

    def test_register_invalid_data(self, client):
        """Test registration with invalid data."""
        response = client.post('/api/v1/auth/register', json={
            'username': 'te',  # Too short
            'email': 'invalid-email',
            'password': '123'  # Too short
        })

        assert response.status_code == 400

    def test_login_success(self, client):
        """Test successful user login."""
        # Register user first
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        # Login
        response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Login successful'
        assert 'user' in data
        assert 'tokens' in data

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post('/api/v1/auth/login', json={
            'username': 'nonexistent',
            'password': 'wrongpassword'
        })

        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'Invalid credentials' in data['message']

    def test_login_disabled_account(self, client):
        """Test login with disabled account."""
        # Register and disable user
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        # Manually disable user in database
        with client.application.app_context():
            user = User.get_by_username('testuser', db.session)
            user.is_active = False
            db.session.commit()

        # Try to login
        response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'Account disabled' in data['message']

    def test_refresh_token_success(self, client):
        """Test successful token refresh."""
        # Register and login
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        refresh_token = json.loads(login_response.data)['tokens']['refresh_token']

        # Refresh token
        response = client.post('/api/v1/auth/refresh', headers={
            'Authorization': f'Bearer {refresh_token}'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'access_token' in data

    def test_get_profile_success(self, client):
        """Test successful profile retrieval."""
        # Register and login
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        access_token = json.loads(login_response.data)['tokens']['access_token']

        # Get profile
        response = client.get('/api/v1/auth/profile', headers={
            'Authorization': f'Bearer {access_token}'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'user' in data
        assert data['user']['username'] == 'testuser'

    def test_update_profile_success(self, client):
        """Test successful profile update."""
        # Register and login
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        access_token = json.loads(login_response.data)['tokens']['access_token']

        # Update profile
        response = client.put('/api/v1/auth/profile', json={
            'email': 'newemail@example.com',
            'current_password': 'TestPassword123',
            'new_password': 'NewPassword123'
        }, headers={
            'Authorization': f'Bearer {access_token}'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Profile updated successfully'

    def test_regenerate_api_key_success(self, client):
        """Test successful API key regeneration."""
        # Register and login
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        access_token = json.loads(login_response.data)['tokens']['access_token']

        # Regenerate API key
        response = client.post('/api/v1/auth/api-key', headers={
            'Authorization': f'Bearer {access_token}'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'api_key' in data
        assert data['message'] == 'API key regenerated successfully'

    def test_logout_success(self, client):
        """Test successful logout."""
        # Register and login
        client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123'
        })

        login_response = client.post('/api/v1/auth/login', json={
            'username': 'testuser',
            'password': 'TestPassword123'
        })

        access_token = json.loads(login_response.data)['tokens']['access_token']

        # Logout
        response = client.post('/api/v1/auth/logout', headers={
            'Authorization': f'Bearer {access_token}'
        })

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Logout successful'

    def test_unauthorized_access(self, client):
        """Test access without authentication."""
        response = client.get('/api/v1/auth/profile')
        assert response.status_code == 401

    def test_invalid_token(self, client):
        """Test access with invalid token."""
        response = client.get('/api/v1/auth/profile', headers={
            'Authorization': 'Bearer invalid_token'
        })
        assert response.status_code == 401