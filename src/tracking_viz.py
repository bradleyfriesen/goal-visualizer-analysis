"""Replay visualizer for player and puck tracking data.

ReplayVisualizer draws one event as a frame-by-frame replay on a rink.
ReplayController adds a slider, play/pause and previous/next buttons.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
from matplotlib.widgets import Slider, Button


class ReplayVisualizer:
    """Visualize player and puck tracking data as a frame-by-frame replay.

    How the drawing works (matplotlib "artists")
    --------------------------------------------
    Everything visible on the rink is an artist. We create each artist ONCE
    (in the _create_*_artists methods) and then, for every frame, update()
    only changes the artists' data: positions, sizes and colours. Creating
    new artists every frame would be far slower.

    To keep the artist count low, all players share ONE scatter artist
    (one dot per player), all player trails share another, and so on. Only
    the jersey numbers are separate artists, because text can't be batched.

    Missing data is stored as NaN. Matplotlib doesn't draw dots whose
    position is NaN, so a player can drop out of a frame (marker, trail and
    all) without any special-casing.

    Array layout
    ------------
    Positions are arrays shaped (frame, entity, 2), where an "entity" is one
    player, or the puck (the puck array has a single entity). Trail styling
    is shaped (trail age, entity, ...), ordered oldest -> newest.
    """

    # Drawing order: higher numbers are drawn on top of lower ones.
    Z_CLOCK = 20
    Z_PLAYER_TRAIL = 101
    Z_SCORER_RING = 102
    Z_PUCK_TRAIL = 102
    Z_PLAYER = 103
    Z_PLAYER_LABEL = 104
    Z_PUCK = 106

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
        if trail_frames < 1:
            raise ValueError("trail_frames must be at least 1.")

        self.data = data.copy()
        self.frames = np.asarray(frames)
        if self.frames.size == 0:
            raise ValueError("Replay contains no frames.")
        self.x_col = x_col
        self.y_col = y_col
        self.highlight_scorer = highlight_scorer

        # Trail appearance by age, oldest -> newest.
        self.trail_frames = int(trail_frames)
        self.trail_alpha = np.linspace(0.12, 0.65, self.trail_frames)
        self.trail_sizes = np.linspace(12, 35, self.trail_frames)

        # Reusable "no points" value for artists that are currently hidden.
        self.empty_offsets = np.empty((0, 2))

        # Data preparation.
        self._setup_teams(home_team_colour, away_team_colour)
        self._precompute_positions()
        self._precompute_clock_strings()

        # Drawing.
        self._setup_figure(rink, figsize)
        self._create_player_artists()
        self._create_puck_artists()

        self.update(0)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _setup_teams(self, home_colour, away_colour):
        self.home_team_id = int(self.data["home_team_id"].dropna().iloc[0])
        self.away_team_id = int(self.data["away_team_id"].dropna().iloc[0])

        self.team_colours = {
            self.home_team_id: home_colour,
            self.away_team_id: away_colour,
        }

    def _team_abbrev(self, team_id):
        return self.data.loc[
            self.data["team_id"].eq(team_id), "team_abbrev"
        ].iloc[0]

    def _xy_by_frame(self, rows):
        """Return one entity's (n_frames, 2) positions, NaN where it has no row."""
        rows = (
            rows.drop_duplicates("frame_idx")
            .set_index("frame_idx")
            .reindex(self.frames)
        )
        return rows[[self.x_col, self.y_col]].to_numpy(dtype=float)

    def _precompute_positions(self):
        """Build player_xy (frame, player, 2) and puck_xy (frame, 1, 2)."""
        entity_type = self.data["entity_type"].str.lower()
        self.player_data = self.data.loc[entity_type.eq("player")]
        self.puck_data = self.data.loc[entity_type.eq("puck")]

        # One record per player. The order here is the player order
        # (the second axis) of every per-player array in this class.
        self.player_keys = (
            self.player_data[["team_id", "sweater_number", "player_id", "team_abbrev"]]
            .drop_duplicates()
            .sort_values(["team_id", "sweater_number"])
            .to_dict("records")
        )

        unexpected = (
            {int(p["team_id"]) for p in self.player_keys} - set(self.team_colours)
        )
        if unexpected:
            raise ValueError(f"Unexpected player team ID(s): {unexpected}")

        # (team, number) pairs in player order; shared by the helpers below.
        player_order = [(p["team_id"], p["sweater_number"]) for p in self.player_keys]

        self.player_xy = self._player_positions(player_order)
        self.player_valid = np.isfinite(self.player_xy).all(axis=2)  # (frame, player)

        # Puck (given a length-1 "entity" axis so it matches the player layout)
        self.puck_xy = self._xy_by_frame(self.puck_data)[:, None, :]

        self.scorer_j = self._find_scorer(player_order)

    def _player_positions(self, player_order):
        """Return (n_frames, n_players, 2) positions, NaN where a player has no row."""
        id_cols = ["team_id", "sweater_number"]
        rows = self.player_data.drop_duplicates(["frame_idx", *id_cols])

        # Wide table: one row per frame, one column per (team, number),
        # with a separate block of columns for each coordinate.
        wide = rows.pivot(
            index="frame_idx", columns=id_cols, values=[self.x_col, self.y_col]
        )

        # Line the rows up with the replay frames and the columns with player order.
        columns = pd.MultiIndex.from_tuples(player_order)
        x = wide[self.x_col].reindex(index=self.frames, columns=columns)
        y = wide[self.y_col].reindex(index=self.frames, columns=columns)

        return np.stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)], axis=2)

    def _find_scorer(self, player_order):
        """Return the scorer's index in player order, or None if there isn't one."""
        is_scorer = self.player_data["is_scorer"].astype("boolean").fillna(False)
        scorer_rows = self.player_data.loc[is_scorer]

        if scorer_rows.empty:
            return None

        row = scorer_rows.iloc[0]
        return player_order.index((row["team_id"], row["sweater_number"]))

    @staticmethod
    def _format_clock(period, seconds_to_goal, goal_time):
        """Return e.g. 'P2 14:32', or '' if any input is missing."""
        if pd.isna(period) or pd.isna(seconds_to_goal) or pd.isna(goal_time):
            return ""

        minutes, seconds = map(int, goal_time.split(":"))
        clock_seconds = minutes * 60 + seconds + seconds_to_goal

        minutes, seconds = divmod(int(round(clock_seconds)), 60)
        return f"P{int(period)} {minutes:02d}:{seconds:02d}"

    def _precompute_clock_strings(self):
        """Build one clock string per replay frame."""
        clock = (
            self.data.groupby("frame_idx", sort=False)
            .first()
            .reindex(self.frames)
        )

        self.clock_strings = [
            self._format_clock(period, seconds_to_goal, goal_time)
            for period, seconds_to_goal, goal_time in zip(
                clock["period"].to_numpy(),
                clock["seconds_to_goal"].to_numpy(dtype=float),
                clock["time_in_period"].to_numpy(),
            )
        ]

    # ------------------------------------------------------------------
    # Creating artists (done once)
    # ------------------------------------------------------------------

    def _setup_figure(self, rink, figsize):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        rink.draw(ax=self.ax)
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_title(
            f"{self._team_abbrev(self.away_team_id)} at "
            f"{self._team_abbrev(self.home_team_id)}"
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
            zorder=self.Z_CLOCK,
        )

    def _trail_style(self, base_rgba):
        """Per-age trail styling for a set of entities.

        base_rgba: (n_entities, 4) colour of each entity.
        Returns (rgba, sizes) shaped (trail_frames, n_entities, 4) and
        (trail_frames, n_entities), oldest age first.
        """
        rgba = np.repeat(base_rgba[None], self.trail_frames, axis=0)
        rgba[:, :, 3] = self.trail_alpha[:, None]

        sizes = np.repeat(self.trail_sizes[:, None], len(base_rgba), axis=1)
        return rgba, sizes

    def _create_trail_artist(self, zorder):
        # Deliberately no colour argument. Passing c=[] here makes matplotlib
        # treat the artist as colormapped, which overrides the colours we set
        # every frame in _draw_trail and makes the trail invisible.
        return self.ax.scatter([], [], linewidths=0, zorder=zorder)

    def _create_player_artists(self):
        n_players = len(self.player_keys)
        team_ids = np.array([int(p["team_id"]) for p in self.player_keys])
        is_home = (team_ids == self.home_team_id)[:, None]  # (n_players, 1)

        team_rgba = np.array([to_rgba(self.team_colours[t]) for t in team_ids])
        white = np.array(to_rgba("white"))

        # Home: filled with team colour, white outline.
        # Away: filled white, team-colour outline.
        fill = np.where(is_home, team_rgba, white)
        outline = np.where(is_home, white, team_rgba)

        # One scatter holding every player. Created at dummy finite positions
        # because scatter() drops NaN points at creation, which would leave
        # the colour arrays out of step with the points. update() then moves
        # them to their real positions each frame.
        self.player_markers = self.ax.scatter(
            np.zeros(n_players),
            np.zeros(n_players),
            s=230,
            facecolors=fill,
            edgecolors=outline,
            linewidths=1.5,
            zorder=self.Z_PLAYER,
        )
        self.player_markers.set_offsets(np.full((n_players, 2), np.nan))

        # Jersey numbers: one Text artist per player, in player order.
        # The number uses the same colour as the marker outline.
        self.player_labels = [
            self.ax.text(
                0,
                0,
                str(int(player["sweater_number"])),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=tuple(outline[j]),
                zorder=self.Z_PLAYER_LABEL,
                visible=False,
            )
            for j, player in enumerate(self.player_keys)
        ]

        # One scatter holding every player's trail dots.
        self.player_trails = self._create_trail_artist(self.Z_PLAYER_TRAIL)
        self.player_trail_rgba, self.player_trail_sizes = self._trail_style(team_rgba)

        # Ring around the scorer (stays empty if there is no scorer).
        ring_colour = (
            tuple(team_rgba[self.scorer_j]) if self.scorer_j is not None else "none"
        )
        self.scorer_ring = self.ax.scatter(
            [],
            [],
            s=450,
            facecolors="none",
            edgecolors=ring_colour,
            linewidths=2.5,
            zorder=self.Z_SCORER_RING,
        )

    def _create_puck_artists(self):
        self.puck_marker = self.ax.scatter(
            [],
            [],
            s=45,
            c="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=self.Z_PUCK,
        )

        self.puck_trail = self._create_trail_artist(self.Z_PUCK_TRAIL)
        self.puck_trail_rgba, self.puck_trail_sizes = self._trail_style(
            np.array([to_rgba("black")])
        )

    # ------------------------------------------------------------------
    # Per-frame updates
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_trail(artist, history_xy, rgba, sizes):
        """Show recent positions as fading dots.

        history_xy: (k, n_entities, 2) positions, oldest first, k <= trail_frames.
        rgba, sizes: per-age styling; only the newest k ages are used, so the
            trail still fades correctly near the start of the replay.
        """
        k = len(history_xy)
        artist.set_offsets(history_xy.reshape(-1, 2))
        artist.set_sizes(sizes[-k:].ravel())
        artist.set_facecolors(rgba[-k:].reshape(-1, 4))

    def update(self, frame_position):
        """Show the frame at this position (0 .. len(frames) - 1)."""
        pos = int(np.clip(frame_position, 0, len(self.frames) - 1))
        trail_start = max(0, pos - self.trail_frames + 1)

        # Players
        xy = self.player_xy[pos]  # (n_players, 2), NaN where missing
        self.player_markers.set_offsets(xy)
        self._draw_trail(
            self.player_trails,
            self.player_xy[trail_start:pos + 1],
            self.player_trail_rgba,
            self.player_trail_sizes,
        )

        for j, label in enumerate(self.player_labels):
            is_present = self.player_valid[pos, j]
            label.set_visible(bool(is_present))
            if is_present:
                label.set_position(xy[j])

        if self.highlight_scorer and self.scorer_j is not None:
            self.scorer_ring.set_offsets(xy[self.scorer_j][None])

        # Puck
        self.puck_marker.set_offsets(self.puck_xy[pos])
        self._draw_trail(
            self.puck_trail,
            self.puck_xy[trail_start:pos + 1],
            self.puck_trail_rgba,
            self.puck_trail_sizes,
        )

        # Clock
        self.clock_text.set_text(
            f"Frame {self.frames[pos]}  |  {self.clock_strings[pos]}"
        )

        self.fig.canvas.draw_idle()

    def benchmark(self, n_frames=100):
        """Print the average time to update and fully render a frame."""
        n = min(n_frames, len(self.frames))
        self.fig.canvas.draw()  # warm-up

        start = time.perf_counter()
        for i in range(n):
            self.update(i)
            self.fig.canvas.draw()
        elapsed = time.perf_counter() - start

        print(f"Average frame rendering time: {1000 * elapsed / n:.1f} ms")

    def show(self):
        plt.show()


class ReplayController:
    """Interactive controller for a ReplayVisualizer.

    Provides a frame slider, playback controls, and single-frame stepping.

    Terminology: a "position" is an index into replay.frames (0 .. n_frames - 1).
    It is not the same as a "frame_idx", the frame number stored in the data,
    which is what the on-screen clock shows.

    interval is the time between frames during playback, in milliseconds.

    Keyboard: left/right arrows step one frame (and pause playback);
    space toggles play/pause.
    """

    def __init__(self, replay, interval=100):
        self.replay = replay
        self.interval = interval

        self.n_frames = len(replay.frames)
        self.position = 0

        # Figure and animation timer
        self.is_playing = False
        self.fig = replay.fig
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._advance_frame)

        # Reserve space below the rink for controls.
        self.fig.subplots_adjust(bottom=0.23)

        # Slider: full-width row
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

        # Buttons: separate row below the slider. We keep them in a list
        # because matplotlib widgets stop responding if they are garbage collected.
        button_specs = [
            ("Previous", self.previous_frame),
            ("Play", self.play),
            ("Pause", self.pause),
            ("Next", self.next_frame),
        ]

        self.buttons = []
        for i, (label, callback) in enumerate(button_specs):
            button_ax = self.fig.add_axes([0.20 + 0.16 * i, 0.025, 0.12, 0.06])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

        # Keyboard controls. Matplotlib already binds the left/right arrows to
        # "back/forward" through the zoom history, so free those two keys up.
        for name in ("keymap.back", "keymap.forward"):
            plt.rcParams[name] = [
                key for key in plt.rcParams[name] if key not in ("left", "right")
            ]

        self._key_actions = {
            "left": self.previous_frame,
            "right": self.next_frame,
            " ": self.toggle_play,  # the space bar
        }
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Make the display match our starting position, even if the
        # replay was left on a different frame beforehand.
        self._set_frame(0)

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

        self.is_playing = True
        self.timer.start()

    def pause(self, event=None):
        """Pause playback."""
        self.is_playing = False
        self.timer.stop()

    def toggle_play(self, event=None):
        """Pause if playing, play if paused."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _on_key_press(self, event):
        """Run the action bound to the pressed key, if there is one."""
        action = self._key_actions.get(event.key)
        if action is not None:
            action()

    def _step(self, delta):
        """Pause playback and move by delta frames."""
        self.pause()
        self._set_frame(self.position + delta)

    def previous_frame(self, event=None):
        """Step backward one frame."""
        self._step(-1)

    def next_frame(self, event=None):
        """Step forward one frame."""
        self._step(+1)

    def show(self):
        """Display the interactive replay."""
        plt.show()
