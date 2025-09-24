#!/usr/bin/env python3
"""
Test script to verify PluginConfig loads without validation errors
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config.plugin import PluginConfig, plugin_config
    print("✅ PluginConfig loaded successfully!")
    print(f"Plugin directory: {plugin_config.PLUGIN_DIR}")
    print(f"Security enabled: {plugin_config.PLUGIN_SECURITY_ENABLED}")
    print(f"Max plugin size: {plugin_config.MAX_PLUGIN_SIZE}")
except Exception as e:
    print(f"❌ Error loading PluginConfig: {e}")
    sys.exit(1)