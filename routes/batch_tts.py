"""
Batch TTS API Routes
"""

import asyncio
import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime
import zipfile
import io
from pathlib import Path

from flask import Blueprint, request, jsonify, send_file, Response
from pydantic import ValidationError

from models.batch_request import (
    BatchRequest, BatchResponse, BatchStatusResponse,
    BatchResultsResponse, BatchListResponse, TTSItem
)
from utils.batch_processor import BatchProcessor, ProcessingConfig
from utils.progress_streamer import progress_streamer
from utils.redis_manager import redis_manager
from utils.auth import require_auth
from utils.exceptions import ValidationException

# Blueprint setup
batch_bp = Blueprint('batch_tts', __name__, url_prefix='/tts/batch')
logger = logging.getLogger(__name__)


def get_batch_processor() -> BatchProcessor:
    """Get batch processor instance"""
    # This would be initialized with proper dependencies in production
    return BatchProcessor(
        tts_service=None,  # Will be injected
        progress_streamer=progress_streamer,
        redis_manager=redis_manager
    )


@batch_bp.route('', methods=['POST'])
@require_auth
async def submit_batch():
    """
    Submit a batch TTS request
    ---
    tags:
      - Batch TTS
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - items
          properties:
            name:
              type: string
              maxLength: 200
            items:
              type: array
              items:
                $ref: '#/definitions/TTSItem'
              minItems: 1
              maxItems: 100
            priority:
              type: string
              enum: [low, normal, high]
              default: normal
            webhook_url:
              type: string
              format: uri
            metadata:
              type: object
    responses:
      201:
        description: Batch request submitted successfully
        schema:
          $ref: '#/definitions/BatchResponse'
      400:
        description: Invalid request data
      401:
        description: Unauthorized
      429:
        description: Rate limit exceeded
    """
    try:
        # Parse and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate items count
        items_data = data.get('items', [])
        if not items_data:
            return jsonify({"error": "At least one item required"}), 400

        if len(items_data) > 100:
            return jsonify({"error": "Maximum 100 items per batch allowed"}), 400

        # Create TTS items
        items = []
        for item_data in items_data:
            try:
                item = TTSItem(**item_data)
                items.append(item)
            except ValidationError as e:
                return jsonify({
                    "error": "Invalid item data",
                    "details": str(e)
                }), 400

        # Create batch request
        batch_request = BatchRequest(
            name=data.get('name'),
            items=items,
            priority=data.get('priority', 'normal'),
            webhook_url=data.get('webhook_url'),
            metadata=data.get('metadata')
        )

        # Get batch processor
        processor = get_batch_processor()

        # Submit to queue for processing
        # TODO: Implement queue management
        pass

        # Start processing in background
        asyncio.create_task(processor.process_batch(batch_request))

        # Return response
        from models.batch_request import BatchStatus
        response = BatchResponse(
            batch_id=batch_request.id,
            status=BatchStatus.PENDING,
            message=f"Batch request submitted successfully. {len(items)} items queued for processing.",
            created_at=batch_request.created_at,
            estimated_completion=datetime.utcnow().replace(minute=30)  # Estimated 30 minutes
        )

        return jsonify(response.dict()), 201

    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting batch: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@batch_bp.route('/<uuid:batch_id>', methods=['GET'])
@require_auth
async def get_batch_status(batch_id: UUID):
    """
    Get batch processing status
    ---
    tags:
      - Batch TTS
    parameters:
      - name: batch_id
        in: path
        required: true
        type: string
        format: uuid
    responses:
      200:
        description: Batch status retrieved successfully
        schema:
          $ref: '#/definitions/BatchStatusResponse'
      404:
        description: Batch not found
      401:
        description: Unauthorized
    """
    try:
        # Get batch processor
        processor = get_batch_processor()

        # Get batch status from Redis
        batch_data = await processor.get_batch_status(batch_id)

        if not batch_data:
            return jsonify({"error": f"Batch {batch_id} not found"}), 404

        # Convert to response format
        response = BatchStatusResponse(
            batch_id=batch_id,
            status=batch_data.get('status', 'unknown'),
            name=batch_data.get('name'),
            priority=batch_data.get('priority', 'normal'),
            created_at=datetime.fromisoformat(batch_data['created_at']),
            started_at=datetime.fromisoformat(batch_data['started_at']) if batch_data.get('started_at') else None,
            completed_at=datetime.fromisoformat(batch_data['completed_at']) if batch_data.get('completed_at') else None,
            progress={
                "total_items": batch_data.get('total_items', 0),
                "completed_items": batch_data.get('completed_items', 0),
                "failed_items": batch_data.get('failed_items', 0),
                "pending_items": batch_data.get('pending_items', 0),
                "percentage": (batch_data.get('completed_items', 0) / batch_data.get('total_items', 1)) * 100
            },
            total_items=batch_data.get('total_items', 0),
            completed_items=batch_data.get('completed_items', 0),
            failed_items=batch_data.get('failed_items', 0),
            pending_items=batch_data.get('pending_items', 0),
            estimated_completion=datetime.utcnow().replace(hour=1),  # Estimated completion
            results=batch_data.get('results', []),
            errors=batch_data.get('errors', []),
            metadata=batch_data.get('metadata')
        )

        return jsonify(response.dict()), 200

    except NotFoundException as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@batch_bp.route('/<uuid:batch_id>/results', methods=['GET'])
@require_auth
async def get_batch_results(batch_id: UUID):
    """
    Get batch processing results
    ---
    tags:
      - Batch TTS
    parameters:
      - name: batch_id
        in: path
        required: true
        type: string
        format: uuid
      - name: include_audio_urls
        in: query
        type: boolean
        default: false
        description: Include audio URLs in response
    responses:
      200:
        description: Batch results retrieved successfully
        schema:
          $ref: '#/definitions/BatchResultsResponse'
      404:
        description: Batch not found
      401:
        description: Unauthorized
    """
    try:
        # Get batch processor
        processor = get_batch_processor()

        # Get batch status
        batch_data = await processor.get_batch_status(batch_id)

        if not batch_data:
            return jsonify({"error": f"Batch {batch_id} not found"}), 404

        # Parse results
        results = []
        for result_data in batch_data.get('results', []):
            # Convert dict back to BatchItemResult if needed
            if isinstance(result_data, dict):
                results.append(result_data)
            else:
                results.append(result_data)

        # Get summary
        summary = batch_data.get('summary', {})

        # Create response
        response = BatchResultsResponse(
            batch_id=batch_id,
            status=batch_data.get('status', 'unknown'),
            total_items=len(results),
            results=results,
            summary=summary,
            download_url=f"/tts/batch/{batch_id}/download" if batch_data.get('status') == 'completed' else None,
            expires_at=datetime.utcnow().replace(hour=24)  # Expires in 24 hours
        )

        return jsonify(response.dict()), 200

    except Exception as e:
        logger.error(f"Error getting batch results: {str(e)}")
        if "not found" in str(e).lower():
            return jsonify({"error": str(e)}), 404
        return jsonify({"error": "Internal server error"}), 500


@batch_bp.route('/<uuid:batch_id>', methods=['DELETE'])
@require_auth
async def cancel_batch(batch_id: UUID):
    """
    Cancel a batch processing request
    ---
    tags:
      - Batch TTS
    parameters:
      - name: batch_id
        in: path
        required: true
        type: string
        format: uuid
    responses:
      200:
        description: Batch cancelled successfully
      404:
        description: Batch not found
      400:
        description: Batch cannot be cancelled (already completed/failed)
      401:
        description: Unauthorized
    """
    try:
        # Get batch processor
        processor = get_batch_processor()

        # Cancel batch
        success = await processor.cancel_batch(batch_id)

        if not success:
            return jsonify({
                "error": "Batch cannot be cancelled (not found or already completed/failed)"
            }), 400

        return jsonify({
            "message": f"Batch {batch_id} cancelled successfully",
            "batch_id": str(batch_id),
            "status": "cancelled"
        }), 200

    except Exception as e:
        logger.error(f"Error cancelling batch: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@batch_bp.route('/<uuid:batch_id>/download', methods=['GET'])
@require_auth
async def download_batch_results(batch_id: UUID):
    """
    Download all batch results as ZIP file
    ---
    tags:
      - Batch TTS
    parameters:
      - name: batch_id
        in: path
        required: true
        type: string
        format: uuid
    responses:
      200:
        description: ZIP file containing all batch results
        content:
          application/zip:
            schema:
              type: string
              format: binary
      404:
        description: Batch not found or not completed
      401:
        description: Unauthorized
    """
    try:
        # Get batch processor
        processor = get_batch_processor()

        # Get batch status
        batch_data = await processor.get_batch_status(batch_id)

        if not batch_data:
            return jsonify({"error": f"Batch {batch_id} not found"}), 404

        if batch_data.get('status') != 'completed':
            return jsonify({
                "error": "Batch not completed yet. Cannot download results."
            }), 400

        # Get results
        results = batch_data.get('results', [])
        if not results:
            return jsonify({"error": "No results available"}), 404

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add results JSON
            results_json = {
                "batch_id": str(batch_id),
                "total_items": len(results),
                "results": results,
                "summary": batch_data.get('summary', {})
            }

            zip_file.writestr('results.json', str(results_json).replace("'", '"'))

            # Add individual audio files (placeholder - in real implementation,
            # you'd fetch actual audio files from storage)
            for i, result in enumerate(results):
                if result.get('audio_url'):
                    # In real implementation, download audio file and add to ZIP
                    # For now, create a placeholder file
                    zip_file.writestr(
                        f"audio_{i+1}.mp3",
                        f"Audio file for item {result.get('item_id', 'unknown')}"
                    )

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'batch_{batch_id}.zip'
        )

    except Exception as e:
        logger.error(f"Error downloading batch results: {str(e)}")
        if "not found" in str(e).lower():
            return jsonify({"error": str(e)}), 404
        return jsonify({"error": "Internal server error"}), 500


@batch_bp.route('', methods=['GET'])
@require_auth
async def list_batches():
    """
    List all batches for the authenticated user
    ---
    tags:
      - Batch TTS
    parameters:
      - name: page
        in: query
        type: integer
        default: 1
        minimum: 1
      - name: page_size
        in: query
        type: integer
        default: 20
        minimum: 1
        maximum: 100
      - name: status
        in: query
        type: string
        enum: [pending, processing, completed, failed, cancelled]
      - name: priority
        in: query
        type: string
        enum: [low, normal, high]
    responses:
      200:
        description: List of batches retrieved successfully
        schema:
          $ref: '#/definitions/BatchListResponse'
      401:
        description: Unauthorized
    """
    try:
        # Parse query parameters
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        status_filter = request.args.get('status')
        priority_filter = request.args.get('priority')

        # In real implementation, you'd filter by user_id
        # For now, return mock data
        batches = [
            {
                "id": "batch-1",
                "name": "Sample Batch 1",
                "status": "completed",
                "priority": "normal",
                "total_items": 10,
                "completed_items": 10,
                "created_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
        ]

        # Apply filters
        if status_filter:
            batches = [b for b in batches if b['status'] == status_filter]
        if priority_filter:
            batches = [b for b in batches if b['priority'] == priority_filter]

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_batches = batches[start_idx:end_idx]

        response = BatchListResponse(
            batches=paginated_batches,
            total_count=len(batches),
            page=page,
            page_size=page_size,
            has_more=end_idx < len(batches)
        )

        return jsonify(response.dict()), 200

    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Error handlers
@batch_bp.errorhandler(ValidationException)
def handle_validation_error(e):
    return jsonify({"error": "Validation failed", "details": str(e)}), 400




# Register blueprint
def register_batch_routes(app):
    """Register batch routes with the Flask app"""
    app.register_blueprint(batch_bp)