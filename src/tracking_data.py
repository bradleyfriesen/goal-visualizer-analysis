from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.paths import PROCESSED_DATA_DIR, COLOUR_PATH



# Columns required for loading and preparing tracking data
REQUIRED_COLUMNS = [
	"game_id",
	"event_id",
	"frame_idx",
	"seconds_to_goal",
	"period",
	"time_in_period",
	"home_team_id",
	"away_team_id",
	"entity_type",
	"player_id",
	"team_id",
	"team_abbrev",
	"sweater_number",
	"is_scorer",
	"x",
	"y",
	"x_att",
	"y_att",
]

# This list is not necessarily exhaustive
INTEGER_COLUMNS = [
	"game_id",
	"event_id",
	"frame_idx",
	"player_id",
	"team_id",
	"sweater_number",
]


def load_tracking_game(
	season,
	game_id,
	columns=None,
):
	"""
	Load tracking data for one game.

	Parameters
	----------
	season : str
		Season identifier, e.g. "20252026".
	game_id : int
		NHL game ID. Usually of the form 20250001. (year, game #)
	columns : list[str] or None
		Optional columns to load. Required visualization columns are
		automatically included. None loads every column.

	Returns
	-------
	pandas.DataFrame
		Sorted and cleaned tracking data for the selected game.
	"""
	
	tracking_path = PROCESSED_DATA_DIR / "nhl-tracking-pipeline" / "tracking" / f"{season}.parquet"

	if not tracking_path.exists():
		raise FileNotFoundError(
			f"Tracking data not found: {tracking_path}"
		)

	# Include required columns when a restricted column list is supplied.
	if columns is not None:
		columns = list(dict.fromkeys(REQUIRED_COLUMNS + list(columns)))

	tracking = pd.read_parquet(
		tracking_path,
		filters=[("game_id", "==", game_id)],
		columns=columns,
	)

	if tracking.empty:
		raise ValueError(
			f"No tracking data found for game {game_id} "
			f"in season {season}."
		)

	# Sort values by event, frame, and player(puck)
	tracking = tracking.sort_values(
		["event_id", "frame_idx", "entity_type"]
	).reset_index(drop=True)

	# Convert ID and index columns to nullable integers.
	for col in INTEGER_COLUMNS:
		if col in tracking.columns:
			tracking[col] = pd.to_numeric(
				tracking[col],
				errors="coerce",
			).astype("Int64")

	return tracking


def select_goal(
	tracking,
	goal_number=None,
	event_id=None,
):
	"""
	Select a goal from a game's tracking data.

	Select either by chronological goal number (one-based) or by event ID.

	Parameters
	----------
	tracking : pandas.DataFrame
		Tracking data for one game.
	goal_number : int or None
		Chronological goal number, starting at 1.
	event_id : int or None
		Explicit event ID.

	Returns
	-------
	tuple
		event_data, frames, coord_x, coord_y
	"""
	if (goal_number is None) == (event_id is None):
		raise ValueError(
			"Specify exactly one of goal_number or event_id."
		)

	available_events = np.sort(
		tracking["event_id"].dropna().unique()
	)

	if goal_number is not None:
		if not isinstance(goal_number, (int, np.integer)):
			raise TypeError("goal_number must be an integer.")

		if goal_number < 1 or goal_number > len(available_events):
			raise IndexError(
				f"Goal {goal_number} requested, but this game contains "
				f"only {len(available_events)} goals."
			)

		selected_event_id = available_events[goal_number - 1]

	else:
		if not isinstance(event_id, (int, np.integer)):
			raise TypeError("event_id must be an integer.")

		if event_id not in available_events:
			raise ValueError(
				f"Event ID {event_id} was not found in the tracking data. "
				f"Available event IDs: {available_events.tolist()}"
			)

		selected_event_id = event_id

	event_data = tracking.loc[
		tracking["event_id"] == selected_event_id
	].copy()

	# Prefer attack-oriented coordinates to keep attacking direction
	# consistent across events.
	coord_x, coord_y = "x_att", "y_att"

	if event_data[[coord_x, coord_y]].notna().sum().min() == 0:
		coord_x, coord_y = "x", "y"

	event_data = event_data.dropna(
		subset=[coord_x, coord_y]
	).copy()

	frames = np.sort(event_data["frame_idx"].unique())

	if len(frames) == 0:
		raise ValueError(
			f"Event {selected_event_id} has no frames "
			"with valid coordinates."
		)

	return event_data, frames, coord_x, coord_y



def load_team_colours(colour_path=COLOUR_PATH):
	"""
	Load NHL team colours, keyed by team ID.

	Returns
	-------
	dict[int, dict]
		Mapping from NHL team ID to its complete colour information.
	"""

	if not COLOUR_PATH.exists():
		raise FileNotFoundError(
			f"Team colour file not found: {COLOUR_PATH}"
		)

	with open(COLOUR_PATH, encoding="utf-8") as f:
		team_data = json.load(f)

	team_colours = {
		int(team["team_id"]): team["colors"]
		for team in team_data
	}

	return team_colours



def load_game_info(season: str, game_id: int,) -> pd.DataFrame:
	"""
	Load metadata for a specific game.
		Returns
	-------
	pd.DataFrame
		Mapping from NHL game ID to its game information.
	"""

	path = PROCESSED_DATA_DIR / "nhl-tracking-pipeline" / "games" / f"{season}.parquet"
	games = pd.read_parquet(path)

	game_info = games.loc[games["game_id"] == game_id].copy()

	if game_info.empty:
		raise ValueError(
			f"Game ID {game_id} not found in season {season}."
		)

	return game_info
