"""
Tenant-aware TTS API routes
"""

from flask import Blueprint, request, jsonify, g
from sqlalchemy.orm import Session

from models.tenancy import TenantAwareAudioRequest, TenantAwareAudioFile
from utils.tenant_middleware import (
    require_organization_context,
    require_organization_admin,
    organization_rate_limit,
    log_organization_request,
    track_organization_usage,
    validate_resource_availability
)
from utils.tenant_manager import tenant_manager

tenant_tts_bp = Blueprint('tenant_tts', __name__, url_prefix='/api/tenant/tts')


@tenant_tts_bp.route('/generate', methods=['POST'])
@require_organization_context
@organization_rate_limit(max_requests=50, window_seconds=60)
@log_organization_request
@track_organization_usage('requests', count=1)
@validate_resource_availability('requests', 1)
def generate_audio():
    """Generate audio with tenant isolation."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Text is required'
            }), 400

        db_session: Session = g.db_session

        # Create tenant-aware audio request
        audio_request = TenantAwareAudioRequest(
            organization_id=g.current_organization.id,
            text=data['text'],
            voice_name=data.get('voice_name', 'Alnilam'),
            output_format=data.get('output_format', 'wav'),
            speed=data.get('speed', 1.0),
            pitch=data.get('pitch', 0.0),
            user_id=getattr(g, 'current_user', {}).get('id')
        )

        db_session.add(audio_request)
        db_session.commit()

        return jsonify({
            'request_id': audio_request.id,
            'status': audio_request.status,
            'message': 'Audio generation request created successfully',
            'estimated_cost': audio_request.cost_per_character * len(audio_request.text)
        }), 201

    except Exception as e:
        return jsonify({
            'error': 'Generation failed',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/requests', methods=['GET'])
@require_organization_context
@organization_rate_limit(max_requests=30, window_seconds=60)
@log_organization_request
def get_requests():
    """Get audio requests for the organization."""
    try:
        db_session: Session = g.db_session

        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 10)), 100)
        status = request.args.get('status')

        # Build query
        query = TenantAwareAudioRequest.tenant_aware_query(db_session)

        if status:
            query = query.filter(TenantAwareAudioRequest.status == status)

        # Paginate
        offset = (page - 1) * per_page
        requests = query.offset(offset).limit(per_page).all()

        # Get total count
        total_query = TenantAwareAudioRequest.tenant_aware_query(db_session)
        if status:
            total_query = total_query.filter(TenantAwareAudioRequest.status == status)
        total = total_query.count()

        return jsonify({
            'requests': [req.to_dict() for req in requests],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch requests',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/requests/<int:request_id>', methods=['GET'])
@require_organization_context
@log_organization_request
def get_request(request_id):
    """Get specific audio request."""
    try:
        db_session: Session = g.db_session

        request_obj = TenantAwareAudioRequest.get_by_id_tenant_aware(db_session, request_id)
        if not request_obj:
            return jsonify({
                'error': 'Request not found',
                'message': 'Audio request not found or access denied'
            }), 404

        return jsonify(request_obj.to_dict())

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch request',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/files', methods=['GET'])
@require_organization_context
@organization_rate_limit(max_requests=30, window_seconds=60)
@log_organization_request
def get_files():
    """Get audio files for the organization."""
    try:
        db_session: Session = g.db_session

        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 10)), 100)
        mime_type = request.args.get('mime_type')

        # Build query
        query = TenantAwareAudioFile.tenant_aware_query(db_session)

        if mime_type:
            query = query.filter(TenantAwareAudioFile.mime_type == mime_type)

        # Paginate
        offset = (page - 1) * per_page
        files = query.offset(offset).limit(per_page).all()

        # Get total count
        total_query = TenantAwareAudioFile.tenant_aware_query(db_session)
        if mime_type:
            total_query = total_query.filter(TenantAwareAudioFile.mime_type == mime_type)
        total = total_query.count()

        return jsonify({
            'files': [file.to_dict() for file in files],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch files',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/usage', methods=['GET'])
@require_organization_context
@organization_rate_limit(max_requests=20, window_seconds=60)
@log_organization_request
def get_usage():
    """Get organization usage statistics."""
    try:
        db_session: Session = g.db_session

        # Get date range from query parameters
        from datetime import datetime, timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)  # Default to last 30 days

        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))

        # Get usage summary
        usage_summary = tenant_manager.get_usage_summary(
            g.current_organization.id,
            start_date,
            end_date,
            db_session
        )

        # Get organization limits
        limits_check = tenant_manager.check_organization_limits(g.current_organization.id, db_session)

        return jsonify({
            'usage': usage_summary,
            'limits': limits_check['limits'],
            'within_limits': limits_check['within_limits'],
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch usage',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/organization', methods=['GET'])
@require_organization_context
@log_organization_request
def get_organization_info():
    """Get organization information."""
    try:
        return jsonify({
            'organization': g.current_organization.to_dict(),
            'member_count': len(g.current_organization.members) if hasattr(g.current_organization, 'members') else 0,
            'resource_count': len(g.current_organization.resources) if hasattr(g.current_organization, 'resources') else 0
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch organization info',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/admin/members', methods=['GET'])
@require_organization_admin
@organization_rate_limit(max_requests=10, window_seconds=60)
@log_organization_request
def get_organization_members():
    """Get organization members (admin only)."""
    try:
        db_session: Session = g.db_session

        members = []
        for member in g.current_organization.members:
            if member.is_active:
                members.append({
                    'id': member.id,
                    'user_id': member.user_id,
                    'role': member.role.value,
                    'joined_at': member.joined_at.isoformat() if member.joined_at else None,
                    'last_active_at': member.last_active_at.isoformat() if member.last_active_at else None
                })

        return jsonify({
            'members': members,
            'total': len(members)
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch members',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/admin/resources', methods=['GET'])
@require_organization_admin
@organization_rate_limit(max_requests=10, window_seconds=60)
@log_organization_request
def get_organization_resources():
    """Get organization resources (admin only)."""
    try:
        db_session: Session = g.db_session

        resources = []
        for resource in g.current_organization.resources:
            if resource.is_active:
                resources.append({
                    'id': resource.id,
                    'resource_type': resource.resource_type,
                    'resource_name': resource.resource_name,
                    'allocated_amount': resource.allocated_amount,
                    'used_amount': resource.used_amount,
                    'unit': resource.unit,
                    'soft_limit': resource.soft_limit,
                    'hard_limit': resource.hard_limit
                })

        return jsonify({
            'resources': resources,
            'total': len(resources)
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch resources',
            'message': str(e)
        }), 500


@tenant_tts_bp.route('/admin/billing', methods=['GET'])
@require_organization_admin
@organization_rate_limit(max_requests=5, window_seconds=60)
@log_organization_request
def get_billing_info():
    """Get organization billing information (admin only)."""
    try:
        billing_records = []
        for record in g.current_organization.billing_records:
            billing_records.append({
                'id': record.id,
                'billing_period_start': record.billing_period_start.isoformat(),
                'billing_period_end': record.billing_period_end.isoformat(),
                'amount': record.amount,
                'currency': record.currency,
                'status': record.status,
                'total_requests': record.total_requests,
                'total_audio_seconds': record.total_audio_seconds,
                'total_storage_bytes': record.total_storage_bytes
            })

        return jsonify({
            'billing_records': billing_records,
            'total_records': len(billing_records),
            'current_month_cost': g.current_organization.current_month_cost,
            'total_cost': g.current_organization.total_cost
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch billing info',
            'message': str(e)
        }), 500