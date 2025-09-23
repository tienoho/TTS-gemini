# Plugin Examples

This directory contains sample plugins demonstrating how to create plugins for the TTS system using the plugin interface.

## Available Sample Plugins

### 1. TTS Plugin Sample (`tts_plugin_sample.py`)

A sample Text-to-Speech plugin that demonstrates:
- Basic plugin structure and lifecycle management
- Hook system integration (pre/post TTS processing)
- Event handling
- Performance metrics tracking
- Configuration validation

**Features:**
- Text preprocessing and validation
- Voice and language support
- Audio generation simulation
- Quality metrics calculation

**Usage:**
```python
from plugins.examples.tts_plugin_sample import SampleTTSPlugin

# Create and initialize plugin
plugin = SampleTTSPlugin()
await plugin.initialize({
    'api_key': 'your-api-key',
    'supported_languages': ['vi', 'en', 'ja'],
    'supported_voices': ['male', 'female', 'neutral']
})

# Use plugin
result = await plugin.synthesize("Hello, world!", voice="male", language="en")
if result['success']:
    audio_data = result['audio_data']
    metadata = result['metadata']
```

### 2. Audio Enhancement Plugin Sample (`audio_enhancement_plugin_sample.py`)

A sample audio enhancement plugin that demonstrates:
- Audio processing capabilities
- Multiple enhancement types
- Parameter validation
- Quality metrics calculation
- Error handling

**Features:**
- Noise reduction
- Volume normalization
- Echo cancellation
- Real-time processing simulation
- Batch processing support

**Enhancement Types:**
- `noise_reduction`: Remove background noise
- `volume_normalization`: Normalize audio levels
- `echo_cancellation`: Remove echo from audio

**Usage:**
```python
from plugins.examples.audio_enhancement_plugin_sample import SampleAudioEnhancementPlugin

# Create and initialize plugin
plugin = SampleAudioEnhancementPlugin()
await plugin.initialize({
    'supported_formats': ['mp3', 'wav', 'ogg'],
    'enhancement_types': ['noise_reduction', 'volume_normalization']
})

# Enhance audio
enhanced_audio = await plugin.enhance(
    audio_data,
    enhancement_type="noise_reduction",
    parameters={'strength': 0.7, 'threshold': -35.0}
)
```

### 3. Webhook Plugin Sample (`webhook_plugin_sample.py`)

A sample webhook plugin that demonstrates:
- Event-driven architecture
- HTTP webhook delivery
- Retry mechanisms
- Event filtering
- Custom headers and authentication

**Features:**
- Multiple event type support
- Configurable retry logic
- Event filtering capabilities
- Custom headers support
- Timeout handling

**Supported Events:**
- `tts.completed`: TTS synthesis completed
- `tts.failed`: TTS synthesis failed
- `audio.enhanced`: Audio enhancement completed
- `plugin.loaded`: Plugin loaded successfully
- `plugin.error`: Plugin error occurred

**Usage:**
```python
from plugins.examples.webhook_plugin_sample import SampleWebhookPlugin

# Create and initialize plugin
plugin = SampleWebhookPlugin()
await plugin.initialize({
    'webhook_url': 'https://your-webhook-endpoint.com/webhook',
    'headers': {'Authorization': 'Bearer your-token'},
    'timeout': 15,
    'retry_attempts': 3
})

# Send webhook
success = await plugin.send_webhook('tts.completed', {
    'text': 'Hello, world!',
    'voice': 'male',
    'duration': 2.5
})
```

## Plugin Development Guidelines

### 1. Plugin Structure

All plugins should inherit from the appropriate base class:
- `TTSPlugin` for TTS functionality
- `AudioEnhancementPlugin` for audio processing
- `WebhookPlugin` for webhook functionality
- `PluginBase` for custom functionality

### 2. Required Methods

Every plugin must implement:
- `get_plugin_info()`: Return plugin metadata
- `initialize(config)`: Initialize plugin with configuration
- `cleanup()`: Cleanup plugin resources

### 3. Configuration

Plugins should validate configuration in `validate_config()`:
```python
def validate_config(self, config: Dict[str, Any]) -> bool:
    """Validate plugin configuration."""
    required_fields = ['api_key', 'endpoint']
    for field in required_fields:
        if field not in config:
            self.log_warning(f"Missing required config field: {field}")
            return False
    return True
```

### 4. Error Handling

Use proper error handling and logging:
```python
try:
    # Plugin logic here
    result = await self.process_data(data)
except Exception as e:
    self.log_error(f"Processing failed: {e}")
    self._performance_metrics['error_count'] += 1
    raise
```

### 5. Performance Monitoring

Track performance metrics:
```python
start_time = time.time()
try:
    # Plugin logic here
    result = await self.execute_task()
    processing_time = time.time() - start_time

    self._performance_metrics['execution_count'] += 1
    self._performance_metrics['total_execution_time'] += processing_time
    self._performance_metrics['last_execution_time'] = processing_time
finally:
    # Cleanup code
    pass
```

### 6. Hook System

Register and use hooks for extensibility:
```python
# Register hooks
self.register_hook(HookType.PRE_TTS, self.pre_tts_hook)
self.register_hook(HookType.POST_TTS, self.post_tts_hook)

# Implement hook methods
async def pre_tts_hook(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-processing hook."""
    processed_text = text.strip()
    return {'modified_text': processed_text}
```

### 7. Event System

Emit and handle events:
```python
# Emit event
await self.emit_event(EventType.TTS_REQUEST, EventContext(
    event_type=EventType.TTS_REQUEST,
    plugin_name=self.name,
    data={'text_length': len(text), 'voice': voice}
))

# Register event handler
self.register_event_handler(EventType.CUSTOM, self.handle_custom_event)
```

### 8. Security Considerations

- Validate all input data
- Use secure defaults
- Implement proper error handling
- Log security-relevant events
- Follow principle of least privilege

### 9. Testing

Test plugins thoroughly:
```python
import pytest

class TestSampleTTSPlugin:
    @pytest.fixture
    async def plugin(self):
        plugin = SampleTTSPlugin()
        await plugin.initialize()
        yield plugin
        await plugin.cleanup()

    async def test_synthesize(self, plugin):
        result = await plugin.synthesize("Hello, world!")
        assert result['success'] == True
        assert 'audio_data' in result
        assert 'metadata' in result
```

## Running the Examples

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run Plugin Manager:**
```python
from utils.plugin_manager import plugin_manager

# Initialize plugin manager
await plugin_manager.initialize()

# Load sample plugins
results = await plugin_manager.load_all_plugins('plugins/examples')

# Check loaded plugins
active_plugins = plugin_manager.get_active_plugins()
print(f"Loaded {len(active_plugins)} plugins")
```

3. **Use Plugins:**
```python
# Get TTS plugin
tts_plugin = plugin_manager.get_plugin_info('sample_tts')
if tts_plugin:
    result = await tts_plugin.synthesize("Hello, world!")
    print(f"TTS Result: {result['success']}")

# Get audio enhancement plugin
enhancement_plugin = plugin_manager.get_plugin_info('sample_audio_enhancement')
if enhancement_plugin:
    enhanced_audio = await enhancement_plugin.enhance(audio_data, 'noise_reduction')
    print(f"Enhanced audio size: {len(enhanced_audio)} bytes")
```

## Best Practices

1. **Keep it Simple:** Start with basic functionality and add features incrementally
2. **Handle Errors:** Always implement proper error handling and logging
3. **Document Code:** Add docstrings and comments for all public methods
4. **Validate Input:** Always validate input data and configuration
5. **Monitor Performance:** Track performance metrics and optimize bottlenecks
6. **Test Thoroughly:** Write unit tests and integration tests
7. **Follow Security:** Implement security best practices
8. **Use Hooks:** Leverage the hook system for extensibility
9. **Handle Events:** Use the event system for loose coupling
10. **Log Appropriately:** Use appropriate log levels and structured logging

## Troubleshooting

### Common Issues

1. **Plugin Not Loading:**
   - Check plugin file syntax
   - Verify dependencies are installed
   - Check file permissions

2. **Configuration Errors:**
   - Validate configuration schema
   - Check required fields
   - Verify data types

3. **Performance Issues:**
   - Monitor execution time
   - Check memory usage
   - Optimize algorithms

4. **Security Violations:**
   - Review code for security issues
   - Check file permissions
   - Validate network access

### Debug Mode

Enable debug logging to troubleshoot issues:
```python
import logging
logging.getLogger('plugin').setLevel(logging.DEBUG)
```

## Contributing

When creating new plugins:

1. Follow the established patterns
2. Include comprehensive tests
3. Add documentation
4. Handle edge cases
5. Implement proper error handling
6. Follow security best practices

For more information, see the main plugin documentation in the `docs/` directory.