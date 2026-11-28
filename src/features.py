"""Feature functions for possession modelling.

Tracking data is long format: one row per entity (player or puck) per frame.
Each function takes a DataFrame and returns it with new columns added.
"""

import numpy as np
import pandas as pd


FRAME_KEYS = ["game_id", "event_id", "frame_idx"]

# Tracking frame rate. Confirm against your data (see the sanity check in
# the add_velocity docstring); it only scales velocities by a constant.
FRAMES_PER_SECOND = 10.0


def _require(df, columns, run_first):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}: run {run_first} first.")


def add_all_features(df):
    df = add_puck_distance(df)
    df = add_velocity(df)
    df = add_relative_velocity(df)
    df = add_radial_speed(df)
    return df


def add_puck_distance(df, x_col="x_att", y_col="y_att"):
    """
    Add puck_x, puck_y and dist_to_puck.

    dist_to_puck is NaN on puck rows, and on any frame where the puck or
    the player has no coordinates. Distances are in the units of x_col/y_col.
    """
    puck = (
        df.loc[df["entity_type"] == "puck", FRAME_KEYS + [x_col, y_col]]
        .rename(columns={x_col: "puck_x", y_col: "puck_y"})
    )

    # many_to_one raises if a frame somehow has two puck rows.
    out = df.merge(puck, on=FRAME_KEYS, how="left", validate="many_to_one")

    distance = np.hypot(out[x_col] - out["puck_x"], out[y_col] - out["puck_y"])
    out["dist_to_puck"] = distance.where(out["entity_type"] == "player")
    return out


def add_velocity(df, x_col="x_att", y_col="y_att", fps=FRAMES_PER_SECOND):
    """
    Add vx, vy and speed (coordinate units per second) for every entity.

    Each entity (a player, or the puck) is differentiated separately within
    each goal, so nothing leaks across goals or between entities.
    np.gradient uses central differences in the interior of an entity's
    track and one-sided differences at its first and last frame, so goals
    without extra leading or trailing frames still get a value. It also
    takes the real frame numbers as the time axis, so a missing frame
    widens the step instead of corrupting the estimate.

    Sanity check: player speeds should top out around 25-35 ft/s. If they
    look several times too large or small, fps is wrong.
    """
    out = df.copy()
    x = out[x_col].to_numpy(dtype=float, na_value=np.nan)
    y = out[y_col].to_numpy(dtype=float, na_value=np.nan)
    frame = out["frame_idx"].to_numpy(dtype=float, na_value=np.nan)

    vx = np.full(len(out), np.nan)
    vy = np.full(len(out), np.nan)

    # One key per entity per goal. The puck has no player_id, so it gets "puck".
    entity = out["player_id"].astype("string").fillna("puck")
    groups = out.groupby([out["game_id"], out["event_id"], entity]).indices

    for pos in groups.values():  # pos = positions of one entity's rows
        if len(pos) < 2:
            continue  # a single frame has no derivative
        pos = pos[np.argsort(frame[pos])]  # time order
        t = frame[pos] / fps
        vx[pos] = np.gradient(x[pos], t)
        vy[pos] = np.gradient(y[pos], t)

    out["vx"] = vx
    out["vy"] = vy
    out["speed"] = np.hypot(vx, vy)
    return out


def add_relative_velocity(df):
    """
    Add puck_vx, puck_vy, rel_vx, rel_vy, and rel_speed.
    
    The relative velocity is the puck's velocity as seen by each player
    (puck velocity minus player velocity), so a puck moving with its
    carrier should give close to zero. It is only defined on player rows
    and is NaN in puck rows. Require's add_velocity to have been run first.
    """
    _require(df, ["vx", "vy"], "add_relative_velocity")
            
    puck = (df.loc[df["entity_type"] == "puck", FRAME_KEYS + ["vx", "vy"]]
        .rename(columns={"vx": "puck_vx", "vy": "puck_vy"})
    )
    
    out = df.merge(puck, on=FRAME_KEYS, how="left", validate="many_to_one")
    
    is_player = out["entity_type"] == "player"
    out["rel_vx"] = (out["puck_vx"] - out["vx"]).where(is_player)
    out["rel_vy"] = (out["puck_vy"] - out["vy"]).where(is_player)
    out["rel_speed"] = np.hypot(out["rel_vx"], out["rel_vy"])
    return out


def add_radial_speed(df, x_col="x_att", y_col="y_att"):
    """
    Add rad_speed.
    
    The radial speed is the rate of change of the distance between the
    player and the puck, so a player in possession should have a low
    radial speed. It can be calculated as the relative velocity
    projected onto the unit vector between the puck and the player.
    It is defined on player rows, and NaN in puck rows, and when the
    player and puck position are equal. 
    
    Requires add_relative_velocity to be run first.
    """
    _require(df,
        ["puck_x", "puck_y", "dist_to_puck", "rel_vx", "rel_vy"],
        ["add_puck_distance","add_relative_velocity"])
    
    out = df.copy()
    
    dx = out["puck_x"] - out[x_col]  # from the player to the puck
    dy = out["puck_y"] - out[y_col]
    
    out["rad_speed"] = (out["rel_vx"] * dx + out["rel_vy"] * dy) / np.hypot(dx, dy)
    return out
