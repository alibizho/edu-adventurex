"""Shared pytest setup.

`test_ml_service_contract.py` imports the GPU service's own `schemas` / `server` modules to prove
they still agree with `app/schemas.py` — the two are separate deployments (this backend runs on
CPU, the engine runs on a rented GPU box) with no shared package, so nothing but a test keeps the
wire contract honest. `ml-service` has a hyphen and cannot be imported as a package, hence the
sys.path entry.

Importing it is cheap and torch-free: the new `server.py` imports `engine` lazily inside its
lifespan, and `schemas.py` is pure Pydantic.
"""
import sys
from pathlib import Path

ML_SERVICE_DIR = Path(__file__).resolve().parents[2] / "ml-service"

if ML_SERVICE_DIR.is_dir():
    sys.path.insert(0, str(ML_SERVICE_DIR))
