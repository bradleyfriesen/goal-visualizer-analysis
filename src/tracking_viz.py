import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
from matplotlib.widgets import Slider, Button
from IPython.display import display

import time


class ReplayVisualizer:
	# Visualize player and puck tracking data as a frame-by-frame replay.

	def __init__(
		self,
		data,
		frames,
		rink,
		home_team_colour,
		away_team_colour,
		x_col="x_att",
		y_col="y_att",
		trail_frames=6,
		highlight_scorer=True,
		figsize=(12, 6.5),
	):
		# Check arguments are good
		self.data = data.copy()
		self.frames = np.asarray(frames)
		self.x_col = x_col
		self.y_col = y_col
		
		self.trail_frames = int(trail_frames)
		self.trail_alpha = np.linspace(0.12, 0.65, self.trail_frames)
		self.trail_sizes = np.linspace(12, 35, self.trail_frames)
		
		self.highlight_scorer = highlight_scorer

		self.home_team_id = int(self.data["home_team_id"].dropna().iloc[0])
		self.away_team_id = int(self.data["away_team_id"].dropna().iloc[0])

		self.player_data = self.data.loc[
			self.data["entity_type"].str.lower().eq("player")
		].copy()

		self.puck_data = self.data.loc[
			self.data["entity_type"].str.lower().eq("puck")
		].copy()

		# Stable player keys for this event
		self.player_keys = (
			self.player_data[["team_id", "sweater_number", "player_id", "team_abbrev"]]
			.drop_duplicates()
			.sort_values(["team_id", "sweater_number"])
			.to_dict("records")
		)
		
		# Compute frame data
		self._precompute_data()

		# Assign team colours by id.
		self.team_colours = {
			self.home_team_id: home_team_colour,
			self.away_team_id: away_team_colour,
		}
		
		# Assign trail colours
		self.trail_colours = {}

		for team_id, colour in self.team_colours.items():
			rgba = np.tile(
				to_rgba(colour),
				(self.trail_frames, 1),
			)
			rgba[:, 3] = self.trail_alpha
			self.trail_colours[team_id] = rgba

		self.puck_trail_colour = np.tile(
			to_rgba("black"),
			(self.trail_frames, 1),
		)
		self.puck_trail_colour[:, 3] = self.trail_alpha

		# Initialize figure and rink
		self.fig, self.ax = plt.subplots(figsize=figsize)
		rink.draw(ax=self.ax)
		self.ax.set_aspect("equal", adjustable="box")
		
		home_team = self.data.loc[
			self.data["team_id"].eq(self.home_team_id),
			"team_abbrev",
		].iloc[0]

		away_team = self.data.loc[
			self.data["team_id"].eq(self.away_team_id),
			"team_abbrev",
		].iloc[0]

		self.ax.set_title(
		    f"{away_team} at {home_team}"
		)
		
		self.clock_text = self.ax.text(
			0.015,
			0.985,
			"",
			transform=self.ax.transAxes,
			ha="left",
			va="top",
			fontsize=11,
			bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4),
			zorder=20,
		)

		# Create one player marker, jersey label, and trail artist per player.
		self.player_artists = {}
		self.player_trails = {}
		self.player_labels = {}
		self.scorer_rings = {}

		for player in self.player_keys:
			# Players labelled by team, number
			key = (player["team_id"], player["sweater_number"])
			team_id = int(player["team_id"])
			colour = self.team_colours[team_id]

			if team_id == self.home_team_id:
				face_colour = colour
				edge_colour = "white"
				label_colour = "white"
			elif team_id == self.away_team_id:
				face_colour = "white"
				edge_colour = colour
				label_colour = colour
			else:
				raise ValueError(f"Unexpected player team ID: {team_id}")

			# Player marker
			marker = self.ax.scatter(
				[],
				[],
				s=230,
				c=[face_colour],
				edgecolors=edge_colour,
				linewidths=1.5,
				zorder=103,
			)

			# Separate outer ring, shown only for the scorer
			scorer_ring = self.ax.scatter(
				[],
				[],
				s=450,
				facecolors="none",
				edgecolors=colour,
				linewidths=2.5,
				zorder=102,
			)

			# Jersey number
			label = self.ax.text(
				0,
				0,
				str(int(player["sweater_number"])),
				ha="center",
				va="center",
				fontsize=8,
				fontweight="bold",
				color=label_colour,
				zorder=104,
			)

			# Player trail
			trail = self.ax.scatter(
				[],
				[],
				s=[],
				c=[],
				marker="o",
				linewidths=0,
				zorder=101,
			)

			self.player_artists[key] = marker
			self.player_labels[key] = label
			self.player_trails[key] = trail
			self.scorer_rings[key] = scorer_ring

		# Puck and puck trail.
		self.puck_artist = self.ax.scatter(
			[],
			[],
			s=45,
			c="black",
			edgecolors="white",
			linewidths=0.7,
			zorder=106,
		)
		self.puck_trail = self.ax.scatter(
			[],
			[],
			s=[],
			c=[],
			marker="o",
			linewidths=0,
			zorder=102,
		)

		self.current_frame_position = 0
		self.update(0)
		
		
		
		

	@staticmethod
	def _finite_xy(rows, x_col, y_col):
		rows = rows.dropna(subset=[x_col, y_col])
		if rows.empty:
			return np.empty((0, 2))
		return rows[[x_col, y_col]].to_numpy(dtype=float)

	def _trail(self, rows, frame_position):
		start = max(0, frame_position - self.trail_frames + 1)
		past_frames = self.frames[start : frame_position + 1]

		rows = rows.loc[rows["frame_idx"].isin(past_frames)]
		rows = rows.sort_values("frame_idx")

		return self._finite_xy(rows, self.x_col, self.y_col)

	def _set_trail(self, artist, xy, rgba):
	    n = len(xy)
	
	    if n == 0:
	        artist.set_offsets(np.empty((0, 2)))
	        return
	
	    artist.set_offsets(xy)
	    artist.set_sizes(self.trail_sizes[-n:])
	    artist.set_facecolors(rgba[-n:])
		
	def _precompute_data(self):
		n_frames = len(self.frames)
		n_players = len(self.player_keys)
	
		# Player positions: (frame, player, coordinate)
		self.player_xy = np.full(
			(n_frames, n_players, 2), np.nan
		)

		scorer_mask = self.player_data["is_scorer"].astype("boolean").fillna(False)
		scorer = self.player_data.loc[scorer_mask]		

		if scorer.empty:
		    self.scorer_key = None
		else:
		    row = scorer.iloc[0]
		    self.scorer_key = (
		        row["team_id"],
		        row["sweater_number"],
		    )
	
		for j, player in enumerate(self.player_keys):
			rows = self.player_data.loc[
				self.player_data["team_id"].eq(player["team_id"])
				& self.player_data["sweater_number"].eq(
					player["sweater_number"]
				)
			]
	
			rows = rows.drop_duplicates("frame_idx")
			rows = rows.set_index("frame_idx").reindex(self.frames)
	
			self.player_xy[:, j, :] = rows[
				[self.x_col, self.y_col]
			].to_numpy(dtype=float)
		
		# Puck positions
		puck = (
			self.puck_data
			.drop_duplicates("frame_idx")
			.set_index("frame_idx")
			.reindex(self.frames)
		)
	
		self.puck_xy = puck[
			[self.x_col, self.y_col]
		].to_numpy(dtype=float)

		self.player_valid = np.isfinite(self.player_xy).all(axis=2)
		self.puck_valid = np.isfinite(self.puck_xy).all(axis=1)

		# Precompute period and game clock.
		clock_data = (
			self.data
			.groupby("frame_idx", sort=False)
			.first()
			.reindex(self.frames)
		)
		
		periods = clock_data["period"].to_numpy()
		seconds_to_goal = clock_data["seconds_to_goal"].to_numpy(dtype=float)
		
		goal_time = clock_data["time_in_period"].to_numpy()
		
		self.clock_strings = []
		
		for period, seconds, goal_clock in zip(
			periods, seconds_to_goal, goal_time
		):
			if pd.isna(period) or pd.isna(seconds) or pd.isna(goal_clock):
				self.clock_strings.append("")
				continue
		
			minutes, seconds_part = map(int, goal_clock.split(":"))
			goal_seconds = minutes * 60 + seconds_part
		
			clock_seconds = goal_seconds + seconds
		
			minutes, seconds_part = divmod(int(round(clock_seconds)), 60)
		
			self.clock_strings.append(
				f"P{int(period)} {minutes:02d}:{seconds_part:02d}"
			)

	def update(self, frame_position):
		frame_position = int(
			np.clip(frame_position, 0, len(self.frames) - 1)
		)
	
		self.current_frame_position = frame_position
		frame_idx = self.frames[frame_position]
	
		# Update players
		for j, player in enumerate(self.player_keys):
			key = (player["team_id"], player["sweater_number"])
	
			xy = self.player_xy[frame_position, j]
	
			marker = self.player_artists[key]
			label = self.player_labels[key]
			ring = self.scorer_rings[key]
			trail_artist = self.player_trails[key]
	
			if not self.player_valid[frame_position, j]:
				marker.set_offsets(np.empty((0, 2)))
				ring.set_offsets(np.empty((0, 2)))
				trail_artist.set_offsets(np.empty((0, 2)))
				label.set_visible(False)
				continue
	
			marker.set_offsets([xy])
			label.set_position(xy)
			label.set_visible(True)
	
			if self.highlight_scorer and key == self.scorer_key:
				ring.set_offsets([xy])
			else:
				ring.set_offsets(np.empty((0, 2)))
			
			# Update trail
			start = max(0, frame_position - self.trail_frames + 1)
	
			trail_xy = self.player_xy[start:frame_position + 1, j]
			trail_xy = trail_xy[self.player_valid[start:frame_position + 1, j]]
	
			self._set_trail(
				trail_artist,
				trail_xy,
				self.trail_colours[int(player["team_id"])],
			)
	
		# Update puck
		puck_xy = self.puck_xy[frame_position]
	
		if self.puck_valid[frame_position]:
			self.puck_artist.set_offsets([puck_xy])
		else:
			self.puck_artist.set_offsets(np.empty((0, 2)))
	
		start = max(0, frame_position - self.trail_frames + 1)
	
		puck_trail_xy = self.puck_xy[start:frame_position + 1]
		puck_trail_xy = puck_trail_xy[
			self.puck_valid[start:frame_position + 1]
		]
	
		self._set_trail(
			self.puck_trail,
			puck_trail_xy,
			self.puck_trail_colour,
		)	
		
		# Update clock
		self.clock_text.set_text(
			f"Frame {frame_idx}  |  {self.clock_strings[frame_position]}"
		)
	
		self.fig.canvas.draw_idle()
	
	def show(self):
		plt.show()


class ReplayController:
	"""
	Interactive controller for a ReplayVisualizer.

	Provides a frame slider, playback controls, and single-frame
	stepping.
	"""

	def __init__(self, replay, interval=100):
		self.replay = replay
		self.interval = interval

		self.n_frames = len(replay.frames)
		self.position = 0

		if self.n_frames == 0:
			raise ValueError("Replay contains no frames.")

		# Figure and animation timer
		self.fig = replay.fig
		self.timer = self.fig.canvas.new_timer(interval=self.interval)
		self.timer.add_callback(self._advance_frame)

		# Reserve space below the rink for controls.
		self.fig.subplots_adjust(bottom=0.23)

		# Slider — full-width row
		slider_ax = self.fig.add_axes([0.15, 0.12, 0.70, 0.03])

		self.slider = Slider(
			ax=slider_ax,
			label="Frame",
			valmin=0,
			valmax=self.n_frames - 1,
			valinit=0,
			valstep=1,
		)

		self.slider.on_changed(self._slider_changed)

		# Buttons — separate row below slider
		previous_ax = self.fig.add_axes([0.20, 0.025, 0.12, 0.06])
		play_ax = self.fig.add_axes([0.36, 0.025, 0.12, 0.06])
		pause_ax = self.fig.add_axes([0.52, 0.025, 0.12, 0.06])
		next_ax = self.fig.add_axes([0.68, 0.025, 0.12, 0.06])

		self.previous_button = Button(previous_ax, "Previous")
		self.play_button = Button(play_ax, "Play")
		self.pause_button = Button(pause_ax, "Pause")
		self.next_button = Button(next_ax, "Next")

		self.previous_button.on_clicked(self.previous_frame)
		self.play_button.on_clicked(self.play)
		self.pause_button.on_clicked(self.pause)
		self.next_button.on_clicked(self.next_frame)

		# Draw initial frame.
		self.replay.update(0)

		################################################################
		self.fig.canvas.draw()

		start = time.perf_counter()

		for i in range(min(100, self.n_frames)):
			self.replay.update(i)
			self.fig.canvas.draw()

		elapsed = time.perf_counter() - start

		print(
			f"Average frame rendering time: "
			f"{1000 * elapsed / min(100, self.n_frames):.1f} ms"
		)
		################################################################

	def _set_frame(self, position):
		"""Set and display a frame by its position."""
		self.position = int(np.clip(position, 0, self.n_frames - 1))

		# Synchronize slider without triggering its callback.
		self.slider.eventson = False
		self.slider.set_val(self.position)
		self.slider.eventson = True

		self.replay.update(self.position)

	def _slider_changed(self, position):
		"""Handle manual slider movement."""
		self.pause()
		self._set_frame(position)

	def _advance_frame(self):
		"""Advance one frame during playback."""
		if self.position >= self.n_frames - 1:
			self.pause()
			return

		self._set_frame(self.position + 1)

	def play(self, event=None):
		"""Start playback."""
		if self.position >= self.n_frames - 1:
			self._set_frame(0)

		self.timer.start()

	def pause(self, event=None):
		"""Pause playback."""
		self.timer.stop()

	def previous_frame(self, event=None):
		"""Step backward one frame."""
		self.pause()
		self._set_frame(self.position - 1)

	def next_frame(self, event=None):
		"""Step forward one frame."""
		self.pause()
		self._set_frame(self.position + 1)

	def show(self):
		"""Display the interactive replay."""
		import matplotlib.pyplot as plt

		plt.show()
