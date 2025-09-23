# Cloud Storage Integration (AWS S3/Google Cloud Storage)

## Tổng quan
Hệ thống cloud storage được sử dụng để lưu trữ audio files một cách scalable và cost-effective, thay vì lưu trữ local.

## AWS S3 Integration

### Configuration
```python
# config/development.py
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'tts-audio-files')
S3_CUSTOM_DOMAIN = os.getenv('S3_CUSTOM_DOMAIN')

# Storage settings
STORAGE_TYPE = 's3'  # 's3' or 'gcs' or 'local'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = ['wav', 'mp3', 'ogg', 'flac', 'aac']
```

### S3 Storage Manager
```python
# utils/cloud_storage/s3_manager.py
import boto3
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

class S3StorageManager:
    def __init__(self, access_key: str, secret_key: str, region: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.region = region

        # Configure boto3 client
        self.config = Config(
            region_name=region,
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'}
        )

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=self.config
        )

        self.s3_resource = boto3.resource(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=self.config
        )

    def upload_file(self, file_path: str, object_key: str,
                   metadata: Dict[str, str] = None, public_read: bool = False) -> str:
        """Upload file to S3"""
        try:
            # Set ACL based on public_read parameter
            acl = 'public-read' if public_read else 'private'

            # Upload file
            extra_args = {'ACL': acl}
            if metadata:
                extra_args['Metadata'] = metadata

            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )

            # Generate URL
            if public_read:
                url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{object_key}"
            else:
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': object_key},
                    ExpiresIn=3600  # 1 hour
                )

            return url

        except ClientError as e:
            raise Exception(f"S3 upload failed: {e}")
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")

    def download_file(self, object_key: str, download_path: str) -> bool:
        """Download file from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, object_key, download_path)
            return True
        except ClientError as e:
            raise Exception(f"S3 download failed: {e}")

    def delete_file(self, object_key: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            raise Exception(f"S3 delete failed: {e}")

    def file_exists(self, object_key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError:
            return False

    def get_file_metadata(self, object_key: str) -> Dict[str, Any]:
        """Get file metadata from S3"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', ''),
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            raise Exception(f"Failed to get metadata: {e}")

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for file access"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise Exception(f"Failed to generate presigned URL: {e}")

    def list_files(self, prefix: str = '', max_keys: int = 100) -> list:
        """List files in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'storage_class': obj.get('StorageClass', 'STANDARD')
                    })

            return files
        except ClientError as e:
            raise Exception(f"Failed to list files: {e}")
```

## Google Cloud Storage Integration

### Configuration
```python
# config/development.py
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME', 'tts-audio-files')
GCP_CREDENTIALS_PATH = os.getenv('GCP_CREDENTIALS_PATH', '/app/credentials.json')
```

### GCS Storage Manager
```python
# utils/cloud_storage/gcs_manager.py
from google.cloud import storage
from google.oauth2 import service_account
import os
from typing import Dict, Any, Optional

class GCSStorageManager:
    def __init__(self, project_id: str, bucket_name: str, credentials_path: str = None):
        self.project_id = project_id
        self.bucket_name = bucket_name

        # Initialize client
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = storage.Client(project=project_id, credentials=credentials)
        else:
            self.client = storage.Client(project=project_id)

        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, file_path: str, object_key: str,
                   metadata: Dict[str, str] = None, public_read: bool = False) -> str:
        """Upload file to GCS"""
        try:
            blob = self.bucket.blob(object_key)

            # Set metadata
            if metadata:
                blob.metadata = metadata

            # Upload file
            blob.upload_from_filename(file_path)

            # Set ACL if public
            if public_read:
                blob.make_public()

            # Generate URL
            if public_read:
                return blob.public_url
            else:
                return blob.generate_signed_url(
                    version="v4",
                    expiration=3600,  # 1 hour
                    method="GET"
                )

        except Exception as e:
            raise Exception(f"GCS upload failed: {e}")

    def download_file(self, object_key: str, download_path: str) -> bool:
        """Download file from GCS"""
        try:
            blob = self.bucket.blob(object_key)
            blob.download_to_filename(download_path)
            return True
        except Exception as e:
            raise Exception(f"GCS download failed: {e}")

    def delete_file(self, object_key: str) -> bool:
        """Delete file from GCS"""
        try:
            blob = self.bucket.blob(object_key)
            blob.delete()
            return True
        except Exception as e:
            raise Exception(f"GCS delete failed: {e}")

    def file_exists(self, object_key: str) -> bool:
        """Check if file exists in GCS"""
        try:
            blob = self.bucket.blob(object_key)
            return blob.exists()
        except Exception:
            return False

    def get_file_metadata(self, object_key: str) -> Dict[str, Any]:
        """Get file metadata from GCS"""
        try:
            blob = self.bucket.blob(object_key)
            blob.reload()  # Refresh metadata

            return {
                'size': blob.size,
                'last_modified': blob.updated,
                'content_type': blob.content_type,
                'metadata': dict(blob.metadata) if blob.metadata else {}
            }
        except Exception as e:
            raise Exception(f"Failed to get metadata: {e}")

    def generate_signed_url(self, object_key: str, expiration: int = 3600) -> str:
        """Generate signed URL for file access"""
        try:
            blob = self.bucket.blob(object_key)
            return blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
        except Exception as e:
            raise Exception(f"Failed to generate signed URL: {e}")
```

## Unified Storage Interface

### Storage Factory
```python
# utils/cloud_storage/storage_factory.py
from typing import Optional
from .s3_manager import S3StorageManager
from .gcs_manager import GCSStorageManager

class StorageFactory:
    @staticmethod
    def create_storage_manager(storage_type: str, **kwargs) -> Optional:
        """Create storage manager based on type"""
        if storage_type.lower() == 's3':
            return S3StorageManager(
                access_key=kwargs.get('access_key'),
                secret_key=kwargs.get('secret_key'),
                region=kwargs.get('region'),
                bucket_name=kwargs.get('bucket_name')
            )
        elif storage_type.lower() == 'gcs':
            return GCSStorageManager(
                project_id=kwargs.get('project_id'),
                bucket_name=kwargs.get('bucket_name'),
                credentials_path=kwargs.get('credentials_path')
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

# Usage
storage_manager = StorageFactory.create_storage_manager(
    storage_type='s3',
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    region=AWS_REGION,
    bucket_name=S3_BUCKET_NAME
)
```

### Audio File Storage Service
```python
# utils/cloud_storage/audio_storage.py
import os
import hashlib
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse

class AudioStorageService:
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager

    def store_audio_file(self, file_path: str, user_id: int,
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store audio file and return storage info"""
        # Generate unique filename
        file_hash = self._calculate_file_hash(file_path)
        filename = f"{user_id}/{file_hash}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"

        # Prepare metadata
        storage_metadata = {
            'user_id': str(user_id),
            'file_hash': file_hash,
            'upload_date': datetime.utcnow().isoformat(),
            'original_filename': os.path.basename(file_path)
        }
        if metadata:
            storage_metadata.update(metadata)

        try:
            # Upload file
            public_url = self.storage_manager.upload_file(
                file_path=file_path,
                object_key=filename,
                metadata=storage_metadata,
                public_read=False  # Private by default
            )

            # Get file info
            file_info = self.storage_manager.get_file_metadata(filename)

            return {
                'filename': filename,
                'public_url': public_url,
                'file_size': file_info['size'],
                'file_hash': file_hash,
                'metadata': storage_metadata,
                'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat()
            }

        except Exception as e:
            raise Exception(f"Failed to store audio file: {e}")

    def get_audio_file(self, filename: str) -> str:
        """Get audio file URL"""
        try:
            return self.storage_manager.generate_presigned_url(filename)
        except Exception as e:
            raise Exception(f"Failed to get audio file: {e}")

    def delete_audio_file(self, filename: str) -> bool:
        """Delete audio file"""
        try:
            return self.storage_manager.delete_file(filename)
        except Exception as e:
            raise Exception(f"Failed to delete audio file: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old files"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        # List files and delete old ones
        files = self.storage_manager.list_files()
        deleted_count = 0

        for file_info in files:
            # Check if file is old based on metadata
            try:
                metadata = self.storage_manager.get_file_metadata(file_info['key'])
                upload_date = metadata.get('metadata', {}).get('upload_date')

                if upload_date:
                    upload_datetime = datetime.fromisoformat(upload_date)
                    if upload_datetime < cutoff_date:
                        self.storage_manager.delete_file(file_info['key'])
                        deleted_count += 1
            except:
                continue

        return deleted_count
```

## Database Integration

### Updated AudioFile Model
```python
# models/audio_file.py (updated)
class AudioFile(Base):
    __tablename__ = 'audio_files'

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey('audio_requests.id'), nullable=False)
    storage_type = Column(String(10), default='local')  # 'local', 's3', 'gcs'
    storage_path = Column(String(500), nullable=False)  # S3 key or GCS object name
    public_url = Column(String(500), nullable=True)  # Public URL if available
    file_path = Column(String(500), nullable=True)  # Local path if stored locally
    filename = Column(String(255), nullable=False)
    mime_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)
    duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    request = relationship("AudioRequest", back_populates="audio_files")

    def get_download_url(self) -> str:
        """Get download URL for file"""
        if self.storage_type == 'local':
            return self.file_path
        elif self.public_url:
            return self.public_url
        else:
            # Generate signed URL
            from utils.cloud_storage.storage_factory import StorageFactory
            storage_manager = StorageFactory.create_storage_manager(self.storage_type)
            return storage_manager.generate_presigned_url(self.storage_path)
```

## API Integration

### Updated TTS Routes
```python
# routes/tts.py (updated)
from utils.cloud_storage.audio_storage import AudioStorageService
from utils.cloud_storage.storage_factory import StorageFactory

@tts_bp.route('/generate', methods=['POST'])
@jwt_required()
def generate_audio():
    """Generate audio with cloud storage"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        # Validate and process request
        # ... existing validation code ...

        # Create audio request
        audio_request = AudioRequest(
            user_id=current_user_id,
            text_content=text,
            voice_name=validated_data.get('voice_name', 'Alnilam'),
            output_format=validated_data.get('output_format', 'wav'),
            speed=validated_data.get('speed', 1.0),
            pitch=validated_data.get('pitch', 0.0),
            metadata={
                'ip_address': request.remote_addr,
                'user_agent': request.user_agent.string,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

        # Mark as processing
        audio_request.mark_as_processing()
        db.session.add(audio_request)
        db.session.commit()

        # Process audio asynchronously
        process_audio_task.delay(audio_request.id, current_user_id)

        return jsonify({
            'message': 'Audio generation started',
            'request_id': audio_request.id,
            'status': audio_request.status,
            'estimated_time': '10-30 seconds'
        }), 202

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': 'Audio generation failed',
            'message': sanitize_error_message(str(e))
        }), 400

def process_audio_task(request_id: int, user_id: int = None):
    """Process audio generation with cloud storage"""
    try:
        # Get request
        audio_request = db.session.query(AudioRequest).get(request_id)
        if not audio_request:
            return

        # Initialize storage service
        storage_manager = StorageFactory.create_storage_manager(
            storage_type=current_app.config.get('STORAGE_TYPE', 'local')
        )
        audio_storage = AudioStorageService(storage_manager)

        # Generate audio
        audio_processor = AudioProcessor(current_app.config['GEMINI_API_KEY'])
        audio_data, mime_type = asyncio.run(audio_processor.generate_audio(
            text=audio_request.text_content,
            voice_name=audio_request.voice_name,
            output_format=audio_request.output_format
        ))

        # Calculate file hash
        file_hash = SecurityUtils.calculate_audio_hash(audio_data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{audio_request.output_format}"
        ) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Store in cloud storage
        storage_info = audio_storage.store_audio_file(
            file_path=temp_file_path,
            user_id=audio_request.user_id,
            metadata={
                'request_id': str(request_id),
                'voice_name': audio_request.voice_name,
                'output_format': audio_request.output_format,
                'processing_time': datetime.utcnow().isoformat()
            }
        )

        # Create audio file record
        audio_file = AudioFile(
            request_id=request_id,
            storage_type=current_app.config.get('STORAGE_TYPE', 'local'),
            storage_path=storage_info['filename'],
            public_url=storage_info.get('public_url'),
            filename=storage_info['filename'],
            mime_type=mime_type,
            file_size=storage_info['file_size'],
            checksum=file_hash
        )

        # Save to database
        db.session.add(audio_file)

        # Update request status
        audio_request.mark_as_completed()
        db.session.commit()

        # Clean up temporary file
        os.unlink(temp_file_path)

    except Exception as e:
        # Mark as failed
        audio_request.mark_as_failed(str(e))
        db.session.commit()
```

## CDN Integration

### CloudFront Setup
```python
# utils/cloud_storage/cdn_manager.py
import boto3
from typing import Dict, Any

class CloudFrontManager:
    def __init__(self, distribution_id: str, access_key: str, secret_key: str):
        self.distribution_id = distribution_id
        self.cloudfront_client = boto3.client(
            'cloudfront',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def invalidate_cache(self, paths: list):
        """Invalidate CloudFront cache"""
        try:
            self.cloudfront_client.create_invalidation(
                DistributionId=self.distribution_id,
                InvalidationBatch={
                    'CallerReference': str(time.time()),
                    'Paths': {
                        'Quantity': len(paths),
                        'Items': paths
                    }
                }
            )
            return True
        except Exception as e:
            print(f"CloudFront invalidation failed: {e}")
            return False

    def get_distribution_config(self) -> Dict[str, Any]:
        """Get CloudFront distribution configuration"""
        try:
            response = self.cloudfront_client.get_distribution(
                Id=self.distribution_id
            )
            return response['Distribution']
        except Exception as e:
            raise Exception(f"Failed to get distribution config: {e}")
```

## Security Features

### Access Control
```python
# utils/cloud_storage/security.py
from typing import Dict, Any
import re

class StorageSecurity:
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path for security"""
        if not file_path:
            return False

        # Check for path traversal
        if '..' in file_path or file_path.startswith('/'):
            return False

        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        if any(char in file_path for char in dangerous_chars):
            return False

        # Check length
        if len(file_path) > 500:
            return False

        return True

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename"""
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')

        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext

        return filename.strip()

    @staticmethod
    def generate_secure_filename(user_id: int, file_hash: str, extension: str) -> str:
        """Generate secure filename"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{user_id}/{file_hash}_{timestamp}.{extension}"

    @staticmethod
    def validate_file_size(file_size: int, max_size: int = 50 * 1024 * 1024) -> bool:
        """Validate file size"""
        return 0 < file_size <= max_size

    @staticmethod
    def validate_file_type(mime_type: str, allowed_types: list) -> bool:
        """Validate file MIME type"""
        return mime_type.lower() in allowed_types
```

## Monitoring và Metrics

### Storage Metrics
```python
# utils/cloud_storage/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Storage metrics
STORAGE_OPERATIONS_TOTAL = Counter(
    'storage_operations_total',
    'Total storage operations',
    ['operation', 'storage_type', 'status']
)

STORAGE_OPERATION_DURATION = Histogram(
    'storage_operation_duration_seconds',
    'Storage operation duration',
    ['operation', 'storage_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

STORAGE_SIZE_BYTES = Gauge(
    'storage_size_bytes',
    'Total storage size in bytes',
    ['storage_type']
)

def record_storage_operation(operation: str, storage_type: str, success: bool, duration: float):
    """Record storage operation metrics"""
    status = 'success' if success else 'failed'

    STORAGE_OPERATIONS_TOTAL.labels(
        operation=operation,
        storage_type=storage_type,
        status=status
    ).inc()

    STORAGE_OPERATION_DURATION.labels(
        operation=operation,
        storage_type=storage_type
    ).observe(duration)

def update_storage_size(storage_type: str, size_bytes: int):
    """Update storage size metric"""
    STORAGE_SIZE_BYTES.labels(storage_type=storage_type).set(size_bytes)
```

## Configuration Examples

### AWS S3 Configuration
```yaml
# .env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=tts-audio-files
STORAGE_TYPE=s3

# Optional
S3_CUSTOM_DOMAIN=cdn.tts-service.com
CLOUDFRONT_DISTRIBUTION_ID=your_distribution_id
```

### Google Cloud Storage Configuration
```yaml
# .env
GCP_PROJECT_ID=your_project_id
GCP_BUCKET_NAME=tts-audio-files
GCP_CREDENTIALS_PATH=/app/credentials.json
STORAGE_TYPE=gcs
```

### Local Storage Configuration (Development)
```yaml
# .env
STORAGE_TYPE=local
LOCAL_STORAGE_PATH=/app/uploads/audio
```

## Error Handling

### Storage Error Types
```python
# utils/cloud_storage/exceptions.py
class StorageError(Exception):
    """Base storage exception"""
    pass

class FileNotFoundError(StorageError):
    """File not found"""
    pass

class PermissionDeniedError(StorageError):
    """Access denied"""
    pass

class QuotaExceededError(StorageError):
    """Storage quota exceeded"""
    pass

class NetworkError(StorageError):
    """Network connectivity issues"""
    pass

class InvalidConfigurationError(StorageError):
    """Invalid storage configuration"""
    pass
```

### Retry Logic
```python
# utils/cloud_storage/retry.py
import time
from typing import Callable, Any
from functools import wraps

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator for storage operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        sleep_time = delay * (backoff ** attempt)
                        time.sleep(sleep_time)
                    else:
                        raise last_exception

            return None
        return wrapper
    return decorator

# Usage
@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def upload_with_retry(self, file_path: str, object_key: str):
    """Upload file with retry logic"""
    return self.storage_manager.upload_file(file_path, object_key)
```

## Testing

### Unit Tests
```python
# tests/test_cloud_storage.py
import pytest
from unittest.mock import Mock, patch
from utils.cloud_storage.s3_manager import S3StorageManager

def test_s3_upload_success():
    """Test successful S3 upload"""
    with patch('boto3.client') as mock_client:
        mock_s3 = Mock()
        mock_client.return_value = mock_s3

        manager = S3StorageManager('key', 'secret', 'us-east-1', 'bucket')

        # Mock successful upload
        mock_s3.upload_file.return_value = None
        mock_s3.generate_presigned_url.return_value = 'https://signed-url'

        result = manager.upload_file('/path/file.wav', 'user/file.wav')

        assert result == 'https://signed-url'
        mock_s3.upload_file.assert_called_once()

def test_s3_upload_failure():
    """Test S3 upload failure"""
    with patch('boto3.client') as mock_client:
        mock_s3 = Mock()
        mock_client.return_value = mock_s3

        manager = S3StorageManager('key', 'secret', 'us-east-1', 'bucket')

        # Mock upload failure
        from botocore.exceptions import ClientError
        mock_s3.upload_file.side_effect = ClientError({'Error': {}}, 'UploadFile')

        with pytest.raises(Exception, match="S3 upload failed"):
            manager.upload_file('/path/file.wav', 'user/file.wav')
```

### Integration Tests
```python
# tests/test_storage_integration.py
def test_audio_storage_service():
    """Test audio storage service"""
    storage_manager = StorageFactory.create_storage_manager('local')
    audio_storage = AudioStorageService(storage_manager)

    # Create test file
    test_file = '/tmp/test_audio.wav'
    with open(test_file, 'wb') as f:
        f.write(b'test audio data')

    # Store file
    result = audio_storage.store_audio_file(
        file_path=test_file,
        user_id=1,
        metadata={'test': 'true'}
    )

    assert 'filename' in result
    assert 'file_size' in result
    assert 'file_hash' in result

    # Clean up
    os.unlink(test_file)
```

## Deployment

### Docker Configuration
```dockerfile
# Dockerfile with cloud storage support
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV STORAGE_TYPE=s3
ENV AWS_REGION=us-east-1

# Run application
CMD ["python", "app/main.py"]
```

### Kubernetes Configuration
```yaml
# k8s storage class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: s3-storage-class
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
  fsType: ext4
```

### Environment Variables
```bash
# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=tts-audio-files

# Google Cloud Storage
GCP_PROJECT_ID=your_project_id
GCP_BUCKET_NAME=tts-audio-files
GCP_CREDENTIALS_PATH=/app/credentials.json

# Common settings
STORAGE_TYPE=s3
MAX_FILE_SIZE=52428800  # 50MB
ALLOWED_EXTENSIONS=wav,mp3,ogg,flac,aac
```

## Best Practices

### 1. Security
- Use IAM roles instead of access keys when possible
- Implement proper access controls
- Encrypt data at rest and in transit
- Regular security audits

### 2. Performance
- Use CDN for public files
- Implement caching strategies
- Optimize file sizes
- Monitor storage performance

### 3. Cost Optimization
- Use appropriate storage classes
- Implement lifecycle policies
- Clean up old files regularly
- Monitor storage costs

### 4. Reliability
- Implement retry logic
- Use multiple storage providers
- Regular backup and recovery testing
- Monitor storage health

### 5. Scalability
- Use distributed storage systems
- Implement horizontal scaling
- Monitor capacity planning
- Use auto-scaling when possible

## Summary

Cloud storage integration provides:

1. **Scalability**: Unlimited storage capacity
2. **Reliability**: High availability and durability
3. **Performance**: Fast global access with CDN
4. **Security**: Encryption and access controls
5. **Cost-effectiveness**: Pay-as-you-go pricing

Key components:
- S3/GCS storage managers
- Unified storage interface
- Security and validation
- Monitoring and metrics
- Error handling and retry logic