from flask import render_template, jsonify, current_app, request
from app.main import bp
from app.utils.queue import QueueManager
from app.utils.image import ImageManager
from app.models import AVAILABLE_MODELS
import json

@bp.route('/')
@bp.route('/index')
def index():
    """Render the main page"""
    recent_images_raw = ImageManager.get_recent_images(limit=12)
    # Parse metadata JSON
    recent_images = []
    for img in recent_images_raw:
        try:
            img_dict = dict(img) # Convert Row object to dict
            if isinstance(img_dict.get('metadata'), str):
                 img_dict['metadata'] = json.loads(img_dict['metadata'])
            else:
                 # Assume it's already a dict or handle appropriately
                 img_dict['metadata'] = img_dict.get('metadata', {})
            recent_images.append(img_dict)
        except (json.JSONDecodeError, TypeError) as e:
            current_app.logger.error(f"Error parsing metadata for recent image {img.get('id', '')}: {e}")
            img_dict = dict(img)
            img_dict['metadata'] = {} # Add empty dict on error
            recent_images.append(img_dict)

    return render_template('index.html',
                         recent_images=recent_images,
                         available_models=AVAILABLE_MODELS)

@bp.route('/gallery')
def gallery():
    """Render the gallery page"""
    page = request.args.get('page', 1, type=int)
    model_id = request.args.get('model', None)
    sort_by = request.args.get('sort', 'newest')

    # Get images with pagination
    result_raw = ImageManager.get_all_images(
        page=page,
        per_page=24,
        model_id=model_id
    )

    # Parse metadata JSON for main page load
    images_parsed = []
    for img in result_raw['images']:
        try:
            img_dict = dict(img) # Convert Row object to dict
            if isinstance(img_dict.get('metadata'), str):
                 img_dict['metadata'] = json.loads(img_dict['metadata'])
            else:
                 img_dict['metadata'] = img_dict.get('metadata', {})
            images_parsed.append(img_dict)
        except (json.JSONDecodeError, TypeError) as e:
            current_app.logger.error(f"Error parsing metadata for gallery image {img.get('id', '')}: {e}")
            img_dict = dict(img)
            img_dict['metadata'] = {} # Add empty dict on error
            images_parsed.append(img_dict)

    # Handle AJAX requests for infinite scroll
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # For AJAX, we might only need basic info, or parse here too if needed by JS
        # For simplicity, returning basic info as before.
        # If JS needs parsed metadata, parse it here like above.
        ajax_images = []
        for img in result_raw['images']:
            img_dict = dict(img)
            # Parse metadata ONLY if needed by infinite scroll JS
            # metadata = json.loads(img_dict['metadata']) if isinstance(img_dict.get('metadata'), str) else img_dict.get('metadata', {})
            ajax_images.append({
                'id': img_dict['id'],
                # Use parsed metadata values if needed by JS, otherwise keep raw strings
                'prompt': img_dict.get('prompt', ''), # Get prompt safely
                'model_id': img_dict.get('model_id', 'Unknown'), # Get model_id safely
                'created_at': img_dict['created_at'].isoformat() if img_dict['created_at'] else None,
                # Pass parsed type if JS needs it
                # 'media_type': metadata.get('type', 'image')
            })
        return jsonify({'images': ajax_images})

    return render_template('gallery.html',
                         images=images_parsed, # Pass parsed images
                         total_pages=result_raw['pages'],
                         current_page=result_raw['current_page'],
                         selected_model=model_id,
                         sort_by=sort_by,
                         available_models=AVAILABLE_MODELS)