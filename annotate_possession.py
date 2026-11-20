"""Annotate puck possession for the goals of one game.

Usage
-----
	python annotate_possession.py 2025020001               # every goal in the game
	python annotate_possession.py 2025020001 --goals 1 3   # just goals 1 and 3
	python annotate_possession.py 2025020001 --out-dir data

Each goal opens in its own replay window. Annotate it (see the key list at the
bottom of the window), then close the window: unsaved changes are saved
automatically and the next goal opens. Reopening a goal resumes from its file.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sportypy.surfaces.hockey import NHLRink

from src.tracking_annotations import (
	DEFAULT_OUT_DIR,
	AnnotatingController,
	PossessionAnnotator,
	frames_through_goal,
)
from src.tracking_data import (
	load_team_colours,
	load_tracking_game,
	select_goal,
	load_game_info,
)
from src.tracking_viz import ReplayVisualizer


def season_from_game_id(game_id):
	"""2025020001 -> '20252026' (the season starts in the game ID's year)."""
	year = game_id // 1_000_000
	return f"{year}{year + 1}"


def annotate_goal(tracking, goal_number, team_colours, out_dir):
	"""Open one goal in an annotation window; returns when the window closes."""
	event_data, frames, coord_x, coord_y = select_goal(
		tracking, goal_number=goal_number
	)
	frames = frames_through_goal(event_data, frames)  # stop at the goal

	home_team_id = int(event_data["home_team_id"].dropna().iloc[0])
	away_team_id = int(event_data["away_team_id"].dropna().iloc[0])

	replay = ReplayVisualizer(
		data=event_data,
		frames=frames,
		rink=NHLRink(),
		home_team_colour=team_colours[home_team_id]["hex"][0],
		away_team_colour=team_colours[away_team_id]["hex"][0],
		x_col=coord_x,
		y_col=coord_y,
	)
	annotator = PossessionAnnotator(
		replay, event_data, goal_number, out_dir=out_dir
	)
	controller = AnnotatingController(replay, annotator, interval=100)  # 10 FPS

	controller.show()  # blocks until the window is closed (which autosaves)

	store = annotator.store
	print(
		f"Goal {goal_number}: {store.n_keyframes} keyframes, "
		f"{store.n_unannotated()} frames unannotated. "
		+ (f"Saved to {annotator.path}." if annotator.path.exists() else "Nothing saved.")
	)
	plt.close(replay.fig)


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	parser.add_argument("game_id", type=int, help="NHL game ID, e.g. 2025020001")
	parser.add_argument("--season", help="e.g. 20252026 (default: from the game ID)")
	parser.add_argument("--goals", type=int, nargs="+", help="goal numbers (default: all)")
	parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where parquet files go")
	args = parser.parse_args(argv)

	season = args.season or season_from_game_id(args.game_id)
	tracking = load_tracking_game(season=season, game_id=args.game_id)
	team_colours = load_team_colours()

	n_goals = len(np.sort(tracking["event_id"].dropna().unique()))
	goals = args.goals or list(range(1, n_goals + 1))
	
	game_info = load_game_info(season, args.game_id)

	print(f"Date: {game_info['game_date'].iloc[0]}, "
		f"Matchup: {game_info['away_team_abbrev'].iloc[0]} @ {game_info['home_team_abbrev'].iloc[0]}."
	)
	print(f"Game {args.game_id} ({season}): {n_goals} goals; annotating {goals}.")
	for goal_number in goals:
		print(f"--- Goal {goal_number} of {n_goals} ---")
		annotate_goal(tracking, goal_number, team_colours, args.out_dir)


if __name__ == "__main__":
	main()
