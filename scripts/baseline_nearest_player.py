"""Baseline: the nearest player to the puck, if close enough, has possession.

Run from the repo root:  python -m scripts.baseline_nearest_player
"""

import numpy as np
import pandas as pd

from src.dataset import (
    FREE,
    POSSESSED,
    STATE_COL,
    build_labeled_frames,
    drop_excluded_frames,
)
from src.features import (
    FRAME_KEYS,
    add_all_features,
    add_puck_distance,
    add_velocity,
    add_relative_velocity,
    add_radial_speed,
)


# Possessor column in the annotation parquets. Adjust if yours differs
# (check with: build_labeled_frames().columns). The puck_state codes
# live in dataset.py.
POSSESSOR_COL = "possessor_id"

# In the data's coordinate units. NHL rinks are 200 ft long, so if the
# "max |x_att|" printed below is near 100 the data is in feet (5 ft ~ 1.5 m).
THRESHOLDS = [2, 3, 4, 5, 6, 8, 10]


def predict_nearest_player(df, max_dist):
    """One row per frame: labels, nearest player, and the baseline's guess."""
    players = df[df["dist_to_puck"].notna()]
    nearest = players.loc[players.groupby(FRAME_KEYS)["dist_to_puck"].idxmin()]
    nearest = nearest.set_index(FRAME_KEYS)

    frames = (
        df.drop_duplicates(FRAME_KEYS)[FRAME_KEYS + ["goal_key", STATE_COL, POSSESSOR_COL]]
        .set_index(FRAME_KEYS)
    )
    frames["nearest_player_id"] = nearest["player_id"]
    frames["nearest_dist"] = nearest["dist_to_puck"]

    close = frames["nearest_dist"] <= max_dist  # NaN compares False -> free
    frames["pred_state"] = np.where(close, POSSESSED, FREE)
    frames["pred_possessor"] = frames["nearest_player_id"].where(close)
    return frames.reset_index()


def summarize(frames):
    """Headline numbers for one threshold."""
    true_possessed = frames[STATE_COL] == POSSESSED
    pred_possessed = frames["pred_state"] == POSSESSED
    tp = (true_possessed & pred_possessed).sum()
    
    # Possessor accuracy counts a missed possession frame as wrong.
    right_player = (frames["pred_possessor"] == frames[POSSESSOR_COL]).fillna(False)

    return {
        "state_acc": (frames[STATE_COL] == frames["pred_state"]).mean(),
        "poss_precision": tp / max(pred_possessed.sum(), 1),
        "poss_recall": tp / max(true_possessed.sum(), 1),
        "possessor_acc": (right_player & true_possessed).sum() / max(true_possessed.sum(), 1),
    }


def main():
    df = build_labeled_frames()
    df = add_all_features(df)

    # Check the codes before filtering: every state should be a code you know.
    print("frames per puck_state (before filtering):")
    # print(df.drop_duplicates(FRAME_KEYS)[STATE_COL].value_counts().sort_index())
    counts = df.drop_duplicates(FRAME_KEYS)[STATE_COL].value_counts().sort_index()
    summary = pd.DataFrame({
        "frames": counts,
        "percent": (100 * counts / counts.sum()).round(1),
    })
    print(summary)

    df = drop_excluded_frames(df)
    
    print("frames per puck_state (after filtering):")
    # print(df.drop_duplicates(FRAME_KEYS)[STATE_COL].value_counts().sort_index())
    counts = df.drop_duplicates(FRAME_KEYS)[STATE_COL].value_counts().sort_index()
    summary = pd.DataFrame({
        "frames": counts,
        "percent": (100 * counts / counts.sum()).round(1),
    })
    print(summary)
    
    print(f"{df['goal_key'].nunique()} goals, {df['game_id'].nunique()} games")
    # Units check. ~100 means => feet
    # print(f"max |x_att|: {df['x_att'].abs().max():.1f}")

    rows = {}
    for d in THRESHOLDS:
        rows[d] = summarize(predict_nearest_player(df, d))
    print(pd.DataFrame(rows).T.round(3).rename_axis("max_dist"))

    # Confusion matrix of true vs predicted state at a middle threshold
    best = predict_nearest_player(df, THRESHOLDS[len(THRESHOLDS) // 2])
    print(pd.crosstab(best[STATE_COL], best["pred_state"]))


if __name__ == "__main__":
    main()
