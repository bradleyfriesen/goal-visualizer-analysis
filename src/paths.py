from pathlib import Path
import os


# Project locations
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


# Shared data directory
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "data"

DATA_DIR = Path(
    os.environ.get("NHL_ANALYTICS_DATA_DIR", DEFAULT_DATA_DIR)
).expanduser().resolve()


# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRACKING_DATA_DIR = PROCESSED_DATA_DIR / "nhl-tracking-pipeline" / "tracking"
COLOUR_PATH = PROCESSED_DATA_DIR / "nhl_team_colours.json"
