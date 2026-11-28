"""Assemble labelled tracking frames for possession modelling.

Joins every annotation file in data/ to its game's tracking data and caches
the result, so feature code and models never touch the raw season files.

The cache rebuilds automatically when any annotation file is newer than it,
so annotating another goal is picked up on the next call.
"""

import re
import warnings
from pathlib import Path

import pandas as pd

from src.tracking_data import load_tracking_game


# Resolved from this file's location so scripts work from any directory.
# Swap for the equivalents in paths.py if you prefer.
REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = REPO_ROOT / "data"
CACHE_DIR = ANNOTATION_DIR / "cache"
LABELED_FRAMES_CACHE = CACHE_DIR / "labeled_frames.parquet"

# Annotation files are named [GAME_ID]_[GOAL_NUMBER].parquet
ANNOTATION_STEM = re.compile(r"\d+_\d+")

# Annotations use "frame"; tracking data uses "frame_idx".
KEY_COLUMNS = ["game_id", "event_id", "frame_idx"]

# puck_state codes written by the annotation tool. Adjust these (and
# STATE_COL) to match what is actually stored in your parquets.
STATE_COL = "puck_state"
POSSESSED = "possessed"
FREE = "free"
CONTESTED = "contested"
UNDETERMINED = "undetermined"
FACEOFF = "faceoff"
POST_GOAL = "post_goal"

# Frames with these labels are not part of open play and are not modelled.
EXCLUDED_STATES = [FACEOFF, POST_GOAL]


def drop_excluded_frames(df, state_col=STATE_COL):
    """
    Remove faceoff and post-goal frames.

    Call this AFTER computing features, not before. The cache keeps these
    frames on purpose: velocities and rolling windows near the edges of
    the usable frames should be computed from the real neighbouring frames,
    not from a sequence that was cut short.
    """
    return df[~df[state_col].isin(EXCLUDED_STATES)].reset_index(drop=True)


def season_from_game_id(game_id):
    """NHL game IDs start with the season's start year: 2025020001 -> '20252026'."""
    start_year = int(str(int(game_id))[:4])
    return f"{start_year}{start_year + 1}"


def annotation_files(annotation_dir=ANNOTATION_DIR):
    """Saved annotation parquets, excluding anything in the cache folder."""
    return sorted(
        p
        for p in Path(annotation_dir).glob("*.parquet")
        if ANNOTATION_STEM.fullmatch(p.stem)
    )


def load_annotations(files):
    """Concatenate annotation files into one table keyed like the tracking data."""
    annotations = pd.concat(
        [pd.read_parquet(p) for p in files],
        ignore_index=True,
    )
    annotations = annotations.rename(columns={"frame": "frame_idx"})
    for col in KEY_COLUMNS:
        annotations[col] = annotations[col].astype("Int64")
    return annotations


def build_labeled_frames(refresh=False):
    """
    Tracking data for every annotated goal, with the labels attached.

    The result is long format: one row per entity (player or puck) per
    annotated frame, with the frame's annotation columns repeated on each
    row. Only annotated frames are kept, so goals you haven't annotated
    yet (and frames past the goal frame) are left out.

    Adds ``goal_key`` ("<game_id>_<event_id>"), one value per goal.

    Parameters
    ----------
    refresh : bool
        Rebuild the cache even if it looks up to date.
    """
    
    # Find all files
    files = annotation_files()
    if not files:
        raise FileNotFoundError(f"No annotation files found in {ANNOTATION_DIR}")

    # Check freshness. If no new files, return
    newest_annotation = max(p.stat().st_mtime for p in files)
    if (
        not refresh
        and LABELED_FRAMES_CACHE.exists()
        and LABELED_FRAMES_CACHE.stat().st_mtime >= newest_annotation
    ):
        return pd.read_parquet(LABELED_FRAMES_CACHE)

    # Load all annotation files in table
    annotations = load_annotations(files)

    parts = []
    for game_id, game_annotations in annotations.groupby("game_id"):
        tracking = load_tracking_game(season_from_game_id(game_id), int(game_id))

        # many_to_one: many entities per frame, exactly one label row per frame.
        # Raises if the same frame was annotated twice.
        part = tracking.merge(
            game_annotations,
            on=KEY_COLUMNS,
            how="inner",
            validate="many_to_one",
        )

        matched = part[["event_id", "frame_idx"]].drop_duplicates().shape[0]
        if matched != len(game_annotations):
            warnings.warn(
                f"Game {game_id}: {len(game_annotations)} annotated frames but "
                f"only {matched} matched tracking frames.",
                stacklevel=2,
            )
        parts.append(part)

    labeled = pd.concat(parts, ignore_index=True)
    labeled["goal_key"] = (
        labeled["game_id"].astype(str) + "_" + labeled["event_id"].astype(str)
    )
    labeled = labeled.sort_values(
        ["game_id", "event_id", "frame_idx", "entity_type"]
    ).reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(LABELED_FRAMES_CACHE, index=False)
    return labeled
