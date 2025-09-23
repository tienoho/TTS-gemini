"""
Authentication routes for Flask TTS API
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Tuple

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, create_refresh_token, get_jwt_identity, jwt_required
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from models import User
from utils.security import SecurityUtils
from utils.validators import UserLoginSchema, UserRegistrationSchema, sanitize_error_message

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register() -> Tuple[Dict, int]:
    """Register a new user.

    Returns:
        JSON response with user data and tokens
    """
    try:
        # Validate request data
        schema = UserRegistrationSchema()
        data = schema.load(request.get_json() or {})
# Create new user with database constraints handling race condition
user = User(
    username=data['username'],
    email=data['email'],
    password=data['password']
)

# Generate API key
api_key = user.generate_api_key()

# Save user - let database handle uniqueness constraints
try:
    db.session.add(user)
    db.session.commit()
except IntegrityError:
    db.session.rollback()
    return jsonify({
        'error': 'Username or email already exists',
        'message': 'Please choose a different username or email address'
    }), 409

    # Create tokens
    access_token = create_access_token(identity=user.id)
    refresh_token = create_refresh_token(identity=user.id)

    return jsonify({
        'message': 'User registered successfully',
        'user': user.to_dict(),
        'api_key': api_key,
        'tokens': {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer'
        }
    }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Registration failed',
            'message': sanitize_error_message(str(e))
        }), 400


@auth_bp.route('/login', methods=['POST'])
def login() -> Tuple[Dict, int]:
    """Authenticate user and return tokens.

    Returns:
        JSON response with user data and tokens
    """
    try:
        # Validate request data
        schema = UserLoginSchema()
        data = schema.load(request.get_json() or {})

        # Find user
        user = User.get_by_username(data['username'], db.session)
        if not user or not user.check_password(data['password']):
            return jsonify({
                'error': 'Invalid credentials',
                'message': 'Username or password is incorrect'
            }), 401

        # Check if user is active
        if not user.is_active:
            return jsonify({
                'error': 'Account disabled',
                'message': 'Your account has been disabled'
            }), 401

        # Update last login
        user.updated_at = datetime.utcnow()
        db.session.commit()

        # Create tokens
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)

        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'tokens': {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer'
            }
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Login failed',
            'message': sanitize_error_message(str(e))
        }), 400


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required()
def refresh_token() -> Tuple[Dict, int]:
    """Refresh access token using refresh token.

    Returns:
        JSON response with new access token
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user
        user = db.session.get(User, current_user_id)
        if not user or not user.is_active:
            return jsonify({
                'error': 'User not found',
                'message': 'User account not found or disabled'
            }), 404

        # Create new access token
        access_token = create_access_token(identity=user.id)

        return jsonify({
            'message': 'Token refreshed successfully',
            'access_token': access_token,
            'token_type': 'Bearer'
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Token refresh failed',
            'message': sanitize_error_message(str(e))
        }), 400


@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile() -> Tuple[Dict, int]:
    """Get current user profile.

    Returns:
        JSON response with user profile data
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({
                'error': 'User not found',
                'message': 'User account not found'
            }), 404

        return jsonify({
            'user': user.to_dict_with_api_key()
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Profile retrieval failed',
            'message': sanitize_error_message(str(e))
        }), 400


@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile() -> Tuple[Dict, int]:
    """Update user profile.

    Returns:
        JSON response with updated user data
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}

        # Get user
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({
                'error': 'User not found',
                'message': 'User account not found'
            }), 404

        # Update email if provided
        if 'email' in data:
            if not SecurityUtils.validate_email(data['email']):
                return jsonify({
                    'error': 'Invalid email',
                    'message': 'Please provide a valid email address'
                }), 400

            # Check if email is already taken
            existing_user = User.get_by_email(data['email'], db.session)
            if existing_user and existing_user.id != user.id:
                return jsonify({
                    'error': 'Email already exists',
                    'message': 'Please use a different email address'
                }), 409

            user.email = data['email']

        # Update password if provided
        if 'current_password' in data and 'new_password' in data:
            if not user.check_password(data['current_password']):
                return jsonify({
                    'error': 'Invalid password',
                    'message': 'Current password is incorrect'
                }), 400

            user.set_password(data['new_password'])

        # Update timestamp
        user.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Profile update failed',
            'message': sanitize_error_message(str(e))
        }), 400


@auth_bp.route('/api-key', methods=['POST'])
@jwt_required()
def regenerate_api_key() -> Tuple[Dict, int]:
    """Regenerate user's API key.

    Returns:
        JSON response with new API key
    """
    try:
        current_user_id = get_jwt_identity()

        # Get user
        user = db.session.get(User, current_user_id)
        if not user:
            return jsonify({
                'error': 'User not found',
                'message': 'User account not found'
            }), 404

        # Generate new API key
        api_key = user.rotate_api_key()
        db.session.commit()

        return jsonify({
            'message': 'API key regenerated successfully',
            'api_key': api_key
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'API key regeneration failed',
            'message': sanitize_error_message(str(e))
        }), 400


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout() -> Tuple[Dict, int]:
    """Logout user (client-side token removal).

    Returns:
        JSON response confirming logout
    """
    try:
        # In a stateless JWT implementation, logout is handled client-side
        # You could implement token blacklisting here if needed

        return jsonify({
            'message': 'Logout successful'
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Logout failed',
            'message': sanitize_error_message(str(e))
        }), 400