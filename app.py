"""Gunicorn entry point: gunicorn --workers 1 --threads 2 app:app."""
from ml_project.web import create_app

app = create_app()
