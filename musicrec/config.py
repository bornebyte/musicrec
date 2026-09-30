import os
import re
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://musicrec:musicrec@localhost:5433/musicrec")
MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/music"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
AUDIO_EXT = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aac"}

# ---- audio analysis ----
SR = 22050
CLIP_SECONDS = 45            # timbre/rhythm features taken from the middle of the track
LID_SR = 16000
LID_WINDOW = 8               # seconds per language-ID window
LID_WINDOWS = 3              # windows per song (at ~20%, 50%, 75%) averaged
WORKERS = int(os.environ.get("WORKERS", "4"))
LANGID_ENABLED = os.environ.get("LANGID", "1") != "0"

FEATURE_DIM = 81
LANG_DIM = 256

# ---- language grouping (per SONG, folder is only a tie-break hint) ----
LANG_K = int(os.environ.get("LANG_K", "0"))              # 0 = auto (number of frequent detected languages)
LANG_MIN_GROUP = int(os.environ.get("LANG_MIN_GROUP", "8"))

# ---- ranking ----
BLOCK_WEIGHTS = {"timbre": 1.0, "harmony": 0.7, "rhythm": 0.9, "energy": 1.0}
LANG_BONUS = 0.25
MOOD_BONUS = 0.05
COPLAY_WEIGHT = 0.15
MAX_MOOD_CLUSTERS = 5


def folder_stem(name: str) -> str:
    """'hindi 2' -> 'hindi' (used only as a hint for naming / fallback)."""
    return re.sub(r"[\s_\-]*\d+$", "", name.strip().lower()) or name.lower()
