import numpy as np
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
		
		# Assign team colours by id.
		self.team_colours = {
			self.home_team_id: home_team_colour,
			self.away_team_id: away_team_colour,
		}
		
		# Initialize figure and rink
		self.fig, self.ax = plt.subplots(figsize=figsize)
		rink.draw(ax=self.ax)
		self.ax.set_aspect("equal", adjustable="box")
		self.ax.set_title("NHL tracking replay")
		
		self.clock_text = self.ax.text(
			0.015, 0.985, "", transform=self.ax.transAxes,
			ha="left", va="top", fontsize=11,
			bbox=dict(
				facecolor="white",
				alpha=0.85,
				edgecolor="none",
				pad=4
			),
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
				[], [], s=230, c=[face_colour], edgecolors=edge_colour,
				linewidths=1.5, zorder=103,
			)

			# Separate outer ring, shown only for the scorer
			scorer_ring = self.ax.scatter(
				[], [], s=450, facecolors="none", edgecolors=colour,
				linewidths=2.5, zorder=102,
			)

			# Jersey number
			label = self.ax.text(
				0, 0, str(int(player["sweater_number"])),
				ha="center", va="center", fontsize=8,
				fontweight="bold", color=label_colour, zorder=104,
			)

			# Player trail
			trail = self.ax.scatter(
				[], [], s=[], c=[], marker="o", linewidths=0, zorder=101,
			)

			self.player_artists[key] = marker
			self.player_labels[key] = label
			self.player_trails[key] = trail
			self.scorer_rings[key] = scorer_ring

		# Puck and puck trail.
		self.puck_artist = self.ax.scatter(
			[], [], s=45, c="black", edgecolors="white",
			linewidths=0.7, zorder=106,
		)
		self.puck_trail = self.ax.scatter(
			[], [], s=[], c=[], marker="o", linewidths=0, zorder=102,
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
		past_frames = self.frames[start:frame_position + 1]
	
		rows = rows.loc[rows["frame_idx"].isin(past_frames)]
		rows = rows.sort_values("frame_idx")

		return self._finite_xy(rows, self.x_col, self.y_col)
		
	def _set_trail(self, artist, xy, base_colour):
		if len(xy) == 0:
			artist.set_offsets(np.empty((0, 2)))
			artist.set_sizes([])
			artist.set_array(None)
			artist.set_facecolors([])
			return
	
		alpha = np.linspace(0.12, 0.65, len(xy))
		rgba = np.tile(to_rgba(base_colour), (len(xy), 1))
		rgba[:, 3] = alpha
	
		sizes = np.linspace(12, 35, len(xy))
	
		artist.set_offsets(xy)
		artist.set_sizes(sizes)
		artist.set_array(None)
		artist.set_facecolors(rgba)
		
	def update(self, frame_position):
		frame_position = int(
			np.clip(frame_position, 0, len(self.frames) - 1)
		)
		self.current_frame_position = frame_position
		frame_idx = self.frames[frame_position]

		current = self.data.loc[self.data["frame_idx"] == frame_idx]
		
		current_players = current.loc[
			current["entity_type"].str.lower().eq("player")
		]

		# Update player positions, labels, scorer outlines, and trails.
		for player in self.player_keys:
			key = (player["team_id"], player["sweater_number"])
			
			mask = (
				current_players["team_id"].eq(player["team_id"])
				& current_players["sweater_number"].eq(player["sweater_number"])
			)
			
			row = current_players.loc[mask]

			marker = self.player_artists[key]
			label = self.player_labels[key]
			trail_artist = self.player_trails[key]
			ring = self.scorer_rings[key]
			
			if row.empty:
				marker.set_offsets(np.empty((0, 2)))
				ring.set_offsets(np.empty((0, 2)))
				label.set_visible(False)
				trail_artist.set_offsets(np.empty((0, 2)))
				trail_artist.set_sizes([])
				trail_artist.set_facecolors([])
				continue

			row = row.iloc[0]
			x, y = row[self.x_col], row[self.y_col]
			marker.set_offsets([[x, y]])
			label.set_position((x, y))
			label.set_visible(True)

			is_scorer = bool(row.get("is_scorer", False))

			if self.highlight_scorer and is_scorer:
				ring.set_offsets([[x, y]])
			else:
				ring.set_offsets(np.empty((0, 2)))
				
			history = self.player_data.loc[
				self.player_data["team_id"].eq(player["team_id"])
				& self.player_data["sweater_number"].eq(
					player["sweater_number"]
				)
			]
			
			trail_xy = self._trail(history, frame_position)
			
			self._set_trail(
				trail_artist,
				trail_xy,
				self.team_colours[int(player["team_id"])],
			)

		# Update puck and its trail.
		current_puck = current.loc[
			current["entity_type"].str.lower().eq("puck")
		]
		
		if current_puck.empty:
			self.puck_artist.set_offsets(np.empty((0, 2)))
		else:
			puck_row = current_puck.iloc[0]
			self.puck_artist.set_offsets([
				[puck_row[self.x_col], puck_row[self.y_col]]
			])

		puck_history = self.puck_data
		puck_xy = self._trail(puck_history, frame_position)
		
		self._set_trail(self.puck_trail, puck_xy, "black")

		# Prefer the source timestamp if present
		# Otherwise show seconds to goal.
		time_parts = []
		
		if (
			"period" in current.columns
			and "time_in_period" in current.columns
			and current["period"].notna().any()
			and current["time_in_period"].notna().any()
		):
			period = int(current["period"].dropna().iloc[0])
			game_time = current["time_in_period"].dropna().iloc[0]
			time_parts.append(f"P{period} {game_time}")
		
		if "seconds_to_goal" in current.columns and current["seconds_to_goal"].notna().any():
			seconds = current["seconds_to_goal"].dropna().iloc[0]
			time_parts.append(f"Goal in {-seconds:.1f} s")
		
		time_text = " | ".join(time_parts)
					
		self.clock_text.set_text(
			f"Frame {frame_idx}  |  {time_text}"
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
		self.timer = self.fig.canvas.new_timer(
			interval=self.interval
		)
		self.timer.add_callback(self._advance_frame)

		# Reserve space below the rink for controls.
		self.fig.subplots_adjust(bottom=0.23)

		# Slider — full-width row
		slider_ax = self.fig.add_axes(
		    [0.15, 0.12, 0.70, 0.03]
		)
		
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
		previous_ax = self.fig.add_axes(
		    [0.20, 0.025, 0.12, 0.06]
		)
		play_ax = self.fig.add_axes(
		    [0.36, 0.025, 0.12, 0.06]
		)
		pause_ax = self.fig.add_axes(
		    [0.52, 0.025, 0.12, 0.06]
		)
		next_ax = self.fig.add_axes(
		    [0.68, 0.025, 0.12, 0.06]
		)
		
		self.previous_button = Button(
			previous_ax, "Previous"
		)
		self.play_button = Button(
			play_ax, "Play"
		)
		self.pause_button = Button(
			pause_ax, "Pause"
		)
		self.next_button = Button(
			next_ax, "Next"
		)

		self.previous_button.on_clicked(self.previous_frame)
		self.play_button.on_clicked(self.play)
		self.pause_button.on_clicked(self.pause)
		self.next_button.on_clicked(self.next_frame)

		# Draw initial frame.
		self.replay.update(0)
		


		# In __init__, after the initial replay.update(0):
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
		
		

	def _set_frame(self, position):
		"""Set and display a frame by its position."""
		self.position = int(np.clip(
			position, 0, self.n_frames - 1
		))

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
