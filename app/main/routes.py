from flask import render_template, jsonify, current_app, request
from app.main import bp
from app.utils.queue import QueueManager
from app.utils.image import ImageManager
from app.models import AVAILABLE_MODELS

@bp.route('/')
@bp.route('/index')
def index():
    """Render the main page"""
    recent_images = ImageManager.get_recent_images(limit=12)
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
    result = ImageManager.get_all_images(
        page=page,
        per_page=24,
        model_id=model_id
    )

    # Handle AJAX requests for infinite scroll
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'images': [{
                'id': img['id'],
                'prompt': img['prompt'],
                'model_id': img['model_id'],
                'created_at': img['created_at'].isoformat() if img['created_at'] else None
            } for img in result['images']]
        })

    return render_template('gallery.html',
                         images=result['images'],
                         total_pages=result['pages'],
                         current_page=result['current_page'],
                         selected_model=model_id,
                         sort_by=sort_by,
                         available_models=AVAILABLE_MODELS)