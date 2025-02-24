"""
Database utilities for CyberImage
"""
import sqlite3
import click
import os
from flask import current_app, g
from flask.cli import with_appcontext
import json
from datetime import datetime

def get_db():
    """Get database connection, initializing if necessary"""
    if "db" not in g:
        # Ensure the database directory exists
        db_path = current_app.config["DATABASE"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Check if we need to initialize the database
        needs_init = not os.path.exists(db_path)

        # Connect to the database
        g.db = sqlite3.connect(
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

        # Initialize if needed
        if needs_init:
            current_app.logger.info("Database does not exist, initializing...")
            init_db()
            current_app.logger.info("Database initialized successfully")

    return g.db

def close_db(e=None):
    """Close database connection"""
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database schema"""
    db = g.db  # Use existing connection from get_db()

    # Read and execute schema
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        db.executescript(f.read())

def init_app(app):
    """Register database functions with the Flask app"""
    app.teardown_appcontext(close_db)
    # We don't need the init-db command anymore since it's automatic
    # But we'll keep it for manual reinitialization if needed
    app.cli.add_command(init_db_command)

@click.command("init-db")
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo("Initialized the database.")