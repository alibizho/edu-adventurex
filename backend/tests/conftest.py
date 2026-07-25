import sys
from pathlib import Path

ML_SERVICE_DIR = Path(__file__).resolve().parents[2] / "ml-service"

if ML_SERVICE_DIR.is_dir():
    sys.path.insert(0, str(ML_SERVICE_DIR))
