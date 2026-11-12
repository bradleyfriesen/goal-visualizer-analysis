from sportypy.surfaces.hockey import NHLRink

from src.tracking_data import (
    load_tracking_game,
    load_game_info,
    select_goal,
    load_team_colours,
)
from src.tracking_viz import ReplayVisualizer, ReplayController

# Test selection
SEASON = "20252026"
GAME_ID = 2025020001
GOAL_NUMBER = 1

game_info = load_game_info(SEASON, GAME_ID)

print(f"Date: {game_info['game_date'].iloc[0]}")
print(
    f"Matchup: {game_info['away_team_abbrev'].iloc[0]} "
    f"@ {game_info['home_team_abbrev'].iloc[0]}"
)

# Load tracking data
tracking = load_tracking_game(
	season=SEASON,
	game_id=GAME_ID,
)

print(f"Loaded {len(tracking):,} tracking rows.")


# Select goal
event_data, frames, coord_x, coord_y = select_goal(
	tracking,
	goal_number=GOAL_NUMBER,
)

print(f"Selected season {SEASON}.")
print(f"Selected game {GAME_ID}.")
print(f"Selected goal {GOAL_NUMBER}.")
print(f"Event ID: {event_data['event_id'].iloc[0]}")
# print(f"Frames: {len(frames)} ({frames[0]} to {frames[-1]})")
# print(f"Coordinates: {coord_x}, {coord_y}")


# Load team colours
team_colours = load_team_colours()

home_team_id = int(event_data["home_team_id"].dropna().iloc[0])
away_team_id = int(event_data["away_team_id"].dropna().iloc[0])

# Create rink and visualizer
rink = NHLRink()

replay = ReplayVisualizer(
	data=event_data,
	frames=frames,
	rink=rink,
	home_team_colour=team_colours[home_team_id]["hex"][0],
	away_team_colour=team_colours[away_team_id]["hex"][0],
	x_col=coord_x,
	y_col=coord_y,
)

controller = ReplayController(
    replay,
    interval=100,  # 10 FPS
)

controller.show()
