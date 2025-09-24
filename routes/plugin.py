"""
Plugin management routes for TTS system
"""

import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from app.extensions import db
from models import Plugin, PluginStatus, PluginType
from utils.plugin_manager import plugin_manager
from utils.plugin_security import security_manager
from utils.validators import sanitize_error_message
from config.plugin import plugin_config

plugin_bp = Blueprint('plugin', __name__, url_prefix='/api/v1/plugins')


@plugin_bp.route('', methods=['GET'])
def list_plugins() -> Tuple[Dict, int]:
    """List all plugins.

    Query parameters:
    - status: Filter by status (active, disabled, error, pending)
    - type: Filter by type (tts, audio_enhancement, webhook, integration, custom)
    - limit: Limit number of results (default: 50)
    - offset: Offset for pagination (default: 0)

    Returns:
        JSON response with plugins list
    """
    try:
        # Get query parameters
        status_filter = request.args.get('status')
        type_filter = request.args.get('type')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # Build query
        query = db.session.query(Plugin)

        if status_filter:
            try:
                status_enum = PluginStatus(status_filter)
                query = query.filter(Plugin.status == status_enum)
            except ValueError:
                return jsonify({
                    'error': 'Invalid status',
                    'message': f'Status must be one of: {", ".join([s.value for s in PluginStatus])}'
                }), 400

        if type_filter:
            try:
                type_enum = PluginType(type_filter)
                query = query.filter(Plugin.plugin_type == type_enum)
            except ValueError:
                return jsonify({
                    'error': 'Invalid type',
                    'message': f'Type must be one of: {", ".join([t.value for t in PluginType])}'
                }), 400

        # Get total count
        total = query.count()

        # Apply pagination
        plugins = query.offset(offset).limit(limit).all()

        # Convert to dict
        plugins_data = [plugin.to_dict() for plugin in plugins]

        return jsonify({
            'plugins': plugins_data,
            'total': total,
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to list plugins',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/active', methods=['GET'])
def list_active_plugins() -> Tuple[Dict, int]:
    """List active plugins.

    Returns:
        JSON response with active plugins
    """
    try:
        active_plugins = plugin_manager.get_active_plugins()

        plugins_data = []
        for name, plugin in active_plugins.items():
            plugin_info = plugin.get_plugin_info()
            plugin_info['status'] = 'active'
            plugins_data.append(plugin_info)

        return jsonify({
            'plugins': plugins_data,
            'total': len(plugins_data)
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to list active plugins',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>', methods=['GET'])
def get_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Get plugin information.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with plugin information
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Get plugin instance if active
        plugin_instance = plugin_manager.get_plugin_info(plugin_name)

        response_data = plugin.to_dict()
        if plugin_instance:
            response_data['runtime_info'] = plugin_instance

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('', methods=['POST'])
def install_plugin() -> Tuple[Dict, int]:
    """Install a new plugin.

    Request body:
    - name: Plugin name (required)
    - display_name: Display name (required)
    - description: Plugin description (optional)
    - version: Plugin version (optional, default: 1.0.0)
    - plugin_type: Plugin type (optional, default: custom)
    - author: Plugin author (optional)
    - homepage: Plugin homepage (optional)
    - license: Plugin license (optional)
    - dependencies: List of dependencies (optional)
    - config_schema: Configuration schema (optional)
    - default_config: Default configuration (optional)

    Returns:
        JSON response with installation result
    """
    try:
        data = request.get_json() or {}

        # Validate required fields
        if not data.get('name'):
            return jsonify({
                'error': 'Missing required field',
                'message': 'Plugin name is required'
            }), 400

        if not data.get('display_name'):
            return jsonify({
                'error': 'Missing required field',
                'message': 'Plugin display name is required'
            }), 400

        # Check if plugin already exists
        existing_plugin = Plugin.get_by_name(data['name'], db.session)
        if existing_plugin:
            return jsonify({
                'error': 'Plugin already exists',
                'message': f'Plugin {data["name"]} is already installed'
            }), 409

        # Create plugin record
        plugin = Plugin(
            name=data['name'],
            display_name=data['display_name'],
            description=data.get('description'),
            version=data.get('version', '1.0.0'),
            plugin_type=PluginType(data['plugin_type']) if data.get('plugin_type') else PluginType.CUSTOM,
            author=data.get('author'),
            homepage=data.get('homepage'),
            license=data.get('license'),
            dependencies=data.get('dependencies', []),
            config_schema=data.get('config_schema', {}),
            default_config=data.get('default_config', {})
        )

        # Save to database
        db.session.add(plugin)
        db.session.commit()

        return jsonify({
            'message': 'Plugin installed successfully',
            'plugin': plugin.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to install plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/upload', methods=['POST'])
def upload_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Upload plugin file.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with upload result
    """
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please provide a plugin file'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400

        # Validate file extension
        if not file.filename.endswith('.py'):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Only Python files (.py) are allowed'
            }), 400

        # Create plugin directory if it doesn't exist
        plugin_dir = Path(plugin_config.PLUGIN_DIR) / plugin_name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        filename = secure_filename(file.filename)
        file_path = plugin_dir / filename
        file.save(str(file_path))

        return jsonify({
            'message': 'Plugin file uploaded successfully',
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to upload plugin file',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/load', methods=['POST'])
def load_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Load a plugin into memory.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with load result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Check if plugin is already active
        if plugin_manager.is_plugin_active(plugin_name):
            return jsonify({
                'message': 'Plugin is already loaded',
                'plugin': plugin.to_dict()
            }), 200

        # Find plugin file
        plugin_dir = Path(plugin_config.PLUGIN_DIR) / plugin_name
        plugin_file = None

        if plugin_dir.exists():
            # Look for Python files in plugin directory
            for py_file in plugin_dir.glob('*.py'):
                if py_file.name != '__pycache__':
                    plugin_file = py_file
                    break

        if not plugin_file:
            return jsonify({
                'error': 'Plugin file not found',
                'message': f'No Python file found for plugin {plugin_name}'
            }), 404

        # Load plugin
        result = plugin_manager.load_plugin(str(plugin_file))

        if result.success:
            # Update plugin status
            plugin.status = PluginStatus.ACTIVE
            plugin.last_executed_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                'message': 'Plugin loaded successfully',
                'plugin': plugin.to_dict(),
                'load_info': {
                    'load_time': result.load_time,
                    'dependencies_resolved': result.dependencies_resolved,
                    'dependencies_failed': result.dependencies_failed,
                    'security_violations': len(result.security_violations)
                }
            }), 200
        else:
            # Update plugin status to error
            plugin.status = PluginStatus.ERROR
            plugin.last_error = result.error_message
            plugin.last_error_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                'error': 'Failed to load plugin',
                'message': result.error_message,
                'security_violations': [v.__dict__ for v in result.security_violations]
            }), 500

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to load plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/enable', methods=['POST'])
def enable_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Enable a plugin.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with enable result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Check if plugin is already active
        if plugin_manager.is_plugin_active(plugin_name):
            return jsonify({
                'message': 'Plugin is already enabled',
                'plugin': plugin.to_dict()
            }), 200

        # Enable plugin
        success = plugin_manager.enable_plugin(plugin_name)

        if success:
            # Update plugin status
            plugin.status = PluginStatus.ACTIVE
            db.session.commit()

            return jsonify({
                'message': 'Plugin enabled successfully',
                'plugin': plugin.to_dict()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to enable plugin',
                'message': f'Could not enable plugin {plugin_name}'
            }), 500

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to enable plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/disable', methods=['POST'])
def disable_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Disable a plugin.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with disable result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Check if plugin is already disabled
        if not plugin_manager.is_plugin_active(plugin_name):
            return jsonify({
                'message': 'Plugin is already disabled',
                'plugin': plugin.to_dict()
            }), 200

        # Disable plugin
        success = plugin_manager.disable_plugin(plugin_name)

        if success:
            # Update plugin status
            plugin.status = PluginStatus.DISABLED
            db.session.commit()

            return jsonify({
                'message': 'Plugin disabled successfully',
                'plugin': plugin.to_dict()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to disable plugin',
                'message': f'Could not disable plugin {plugin_name}'
            }), 500

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to disable plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/reload', methods=['POST'])
def reload_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Reload a plugin.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with reload result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Reload plugin
        success = plugin_manager.reload_plugin(plugin_name)

        if success:
            # Update plugin status
            plugin.status = PluginStatus.ACTIVE
            plugin.last_executed_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                'message': 'Plugin reloaded successfully',
                'plugin': plugin.to_dict()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to reload plugin',
                'message': f'Could not reload plugin {plugin_name}'
            }), 500

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to reload plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>', methods=['PUT'])
def update_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Update plugin information.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with update result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        data = request.get_json() or {}

        # Update fields
        updatable_fields = [
            'display_name', 'description', 'version', 'author',
            'homepage', 'license', 'config_schema', 'default_config',
            'current_config', 'memory_limit', 'timeout_limit'
        ]

        for field in updatable_fields:
            if field in data:
                setattr(plugin, field, data[field])

        # Update timestamp
        plugin.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'message': 'Plugin updated successfully',
            'plugin': plugin.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to update plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>', methods=['DELETE'])
def uninstall_plugin(plugin_name: str) -> Tuple[Dict, int]:
    """Uninstall a plugin.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with uninstall result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Disable plugin first if active
        if plugin_manager.is_plugin_active(plugin_name):
            plugin_manager.disable_plugin(plugin_name)

        # Remove plugin files
        plugin_dir = Path(plugin_config.PLUGIN_DIR) / plugin_name
        if plugin_dir.exists():
            import shutil
            shutil.rmtree(plugin_dir)

        # Remove from database
        db.session.delete(plugin)
        db.session.commit()

        return jsonify({
            'message': 'Plugin uninstalled successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to uninstall plugin',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/config', methods=['GET'])
def get_plugin_config(plugin_name: str) -> Tuple[Dict, int]:
    """Get plugin configuration.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with plugin configuration
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        return jsonify({
            'plugin_name': plugin_name,
            'config_schema': plugin.config_schema,
            'default_config': plugin.default_config,
            'current_config': plugin.current_config
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get plugin config',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/config', methods=['PUT'])
def update_plugin_config(plugin_name: str) -> Tuple[Dict, int]:
    """Update plugin configuration.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with update result
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        data = request.get_json() or {}

        # Update current config
        if 'config' in data:
            plugin.current_config = data['config']

        # Update timestamp
        plugin.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'message': 'Plugin configuration updated successfully',
            'plugin_name': plugin_name,
            'current_config': plugin.current_config
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Failed to update plugin config',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/<plugin_name>/statistics', methods=['GET'])
def get_plugin_statistics(plugin_name: str) -> Tuple[Dict, int]:
    """Get plugin statistics.

    Args:
        plugin_name: Name of the plugin

    Returns:
        JSON response with plugin statistics
    """
    try:
        # Get plugin from database
        plugin = Plugin.get_by_name(plugin_name, db.session)
        if not plugin:
            return jsonify({
                'error': 'Plugin not found',
                'message': f'Plugin {plugin_name} not found'
            }), 404

        # Get manager statistics
        manager_stats = plugin_manager.get_statistics()

        # Get security violations
        security_violations = security_manager.get_security_violations(plugin_name)

        return jsonify({
            'plugin_name': plugin_name,
            'database_stats': {
                'execution_count': plugin.execution_count,
                'error_count': plugin.error_count,
                'last_executed_at': plugin.last_executed_at.isoformat() if plugin.last_executed_at else None,
                'last_error_at': plugin.last_error_at.isoformat() if plugin.last_error_at else None,
                'installed_at': plugin.installed_at.isoformat() if plugin.installed_at else None
            },
            'manager_stats': manager_stats,
            'security_violations': len(security_violations),
            'security_violations_details': [v.__dict__ for v in security_violations]
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get plugin statistics',
            'message': sanitize_error_message(str(e))
        }), 500


@plugin_bp.route('/statistics', methods=['GET'])
def get_all_plugin_statistics() -> Tuple[Dict, int]:
    """Get statistics for all plugins.

    Returns:
        JSON response with all plugin statistics
    """
    try:
        # Get all plugins
        plugins = db.session.query(Plugin).all()

        total_plugins = len(plugins)
        active_plugins = sum(1 for p in plugins if p.status == PluginStatus.ACTIVE)
        disabled_plugins = sum(1 for p in plugins if p.status == PluginStatus.DISABLED)
        error_plugins = sum(1 for p in plugins if p.status == PluginStatus.ERROR)

        # Get manager statistics
        manager_stats = plugin_manager.get_statistics()

        # Get security report
        security_report = security_manager.get_security_report()

        return jsonify({
            'total_plugins': total_plugins,
            'active_plugins': active_plugins,
            'disabled_plugins': disabled_plugins,
            'error_plugins': error_plugins,
            'manager_stats': manager_stats,
            'security_report': security_report
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Failed to get plugin statistics',
            'message': sanitize_error_message(str(e))
        }), 500