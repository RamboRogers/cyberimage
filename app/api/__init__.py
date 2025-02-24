"""
API Blueprint for CyberImage
"""
from flask import Blueprint

bp = Blueprint("api", __name__)

# Import routes after blueprint creation to avoid circular imports
from app.api import routes