"""
Wrapper MLflow UI — réécrit le header Host avant que le middleware
de protection DNS rebinding de MLflow ne le lise.
MLflow n'accepte que localhost/127.0.0.1 par défaut ; on force ce header
au niveau WSGI pour que les requêtes externes (IP VPS) passent.
"""
import os

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:////app/mlruns/mlruns.db")
os.environ.setdefault("MLFLOW_DEFAULT_ARTIFACT_ROOT", "/app/mlruns/artifacts")

from mlflow.server import app as _mlflow_app  # noqa: E402

_original_wsgi = _mlflow_app.wsgi_app


def _fix_host_middleware(environ, start_response):
    environ["HTTP_HOST"] = "localhost"
    return _original_wsgi(environ, start_response)


_mlflow_app.wsgi_app = _fix_host_middleware
app = _mlflow_app
