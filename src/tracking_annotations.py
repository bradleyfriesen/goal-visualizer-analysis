
"""Keyboard-driven puck possession annotation for the replay visualizer.
 
Annotations are stored as sparse *keyframes* that carry forward: whatever you
enter at frame_idx k stays in effect for k+1, k+2, ... until the next
keyframe. Saving writes one row per frame, with carried-forward values filled in.
 
Keys
----
    1  possessed      then type the possessor, Enter     e.g.  TOR 34
    2  free
    3  contested      then type 2+ players, Enter        e.g.  TOR 34, BOS 88
    4  undetermined
    5  faceoff
    6  post_goal	  some goal_frame_idx are not precise
    z          undo the last change (and jump to the frame it affected)
    backspace  delete the keyframe at this frame (reverts to the carried value)
    [  ]       jump to the previous / next keyframe
    ctrl+s     save
 
    Left/right arrows and space still step/play as usual. While you are typing
    players, every key goes to the entry box: Enter commits, Escape cancels.
    A bare number (e.g. 34) works if only one player during the goal wears it.
 
Usage
-----
    frames = frames_through_goal(event_data, frames)   # stop at goal_frame_idx
    replay = ReplayVisualizer(event_data, frames, rink, home, away, ...)
    annotator = PossessionAnnotator(replay, event_data, goal_number)
    controller = AnnotatingController(replay, annotator)
    controller.show()
"""
 
from __future__ import annotations
 
import re
import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
 
import matplotlib as mpl
import numpy as np
import pandas as pd
 
from src.paths import REPO_ROOT
from src.tracking_viz import ReplayController
 
 
DEFAULT_OUT_DIR = REPO_ROOT / "data"
 
# Number key -> puck state. The saved parquet stores the state NAME (a plain
# string), so a file stays readable and can't be misread if codes ever change.
STATE_KEYS = {
    "1": "possessed",
    "2": "free",
    "3": "contested",
    "4": "undetermined",
    "5": "faceoff",
    "6": "post_goal",
}
STATES = tuple(STATE_KEYS.values())
 
COMMAND_KEYS = {"z", "backspace", "delete", "[", "]", "ctrl+s"}
 
POSSESSOR_COLOUR = "gold"
CONTESTOR_COLOUR = "red"
 
 
# ----------------------------------------------------------------------
# Goal helpers (use these in annotate_possession.py)
# ----------------------------------------------------------------------
 
def _single_value(df, column):
    """Return the one value a column holds in a single-event dataframe."""
    values = df[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {column} in this event, found {list(values)}."
        )
    return int(values[0])
 
 
def frames_through_goal(event_data, frames):
    """Keep only frames up to and including goal_frame_idx.
 
    Pass the result to ReplayVisualizer so the replay (and therefore the
    annotation) ends at the goal.
    """
    goal_frame = _single_value(event_data, "goal_frame_idx")
    kept = np.asarray(frames)[np.asarray(frames) <= goal_frame]
    if kept.size == 0:
        raise ValueError(f"No frames at or before goal_frame_idx {goal_frame}.")
    return kept
 
 
def goal_number_of(tracking, event_id):
    """Chronological (one-based) goal number of an event in a game's data."""
    events = np.sort(tracking["event_id"].dropna().unique())
    matches = np.flatnonzero(events == event_id)
    if matches.size == 0:
        raise ValueError(f"Event ID {event_id} not found in the tracking data.")
    return int(matches[0]) + 1
 
 
# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------
 
@dataclass(frozen=True)
class Annotation:
    """What is true of the puck from some frame onward."""
 
    state: str
    possessor_id: int | None = None
    contestor_ids: tuple[int, ...] = ()
 
    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"Unknown state {self.state!r}.")
        if self.state == "possessed" and self.possessor_id is None:
            raise ValueError("Possessed needs a possessor.")
        if self.state != "possessed" and self.possessor_id is not None:
            raise ValueError(f"{self.state} can't have a possessor.")
        if self.state == "contested" and len(self.contestor_ids) < 2:
            raise ValueError("Contested needs at least 2 players.")
        if self.state != "contested" and self.contestor_ids:
            raise ValueError(f"{self.state} can't have contestors.")
 
 
# team letters (optional) followed by a sweater number
_PLAYER_TOKEN = re.compile(r"([A-Za-z]*)\s*(\d+)")
 
 
class Roster:
    """Maps (team_abbrev, sweater_number) <-> player_id for one goal."""
 
    def __init__(self, player_keys):
        self._by_tag = {}
        skipped = []
 
        for player in player_keys:
            abbrev = player["team_abbrev"]
            number = player["sweater_number"]
            player_id = player["player_id"]
            if pd.isna(abbrev) or pd.isna(number) or pd.isna(player_id):
                skipped.append(player)
                continue
 
            tag = (str(abbrev).upper(), int(number))
            if self._by_tag.get(tag, int(player_id)) != int(player_id):
                raise ValueError(f"{tag} maps to more than one player_id.")
            self._by_tag[tag] = int(player_id)
 
        self._by_id = {pid: tag for tag, pid in self._by_tag.items()}
 
        if skipped:
            warnings.warn(
                f"{len(skipped)} player(s) lack a team, number or player_id and "
                "can't be annotated.",
                stacklevel=2,
            )
 
    def __contains__(self, player_id):
        return player_id in self._by_id
 
    def tag(self, player_id):
        """(team_abbrev, sweater_number) for a player_id."""
        return self._by_id[player_id]
 
    def label(self, player_id):
        team, number = self.tag(player_id)
        return f"{team} {number}"
 
    def parse(self, text):
        """Turn 'TOR 34, BOS 88' into [player_id, player_id]. Raises ValueError."""
        text = text.strip()
        tokens = _PLAYER_TOKEN.findall(text)
        leftover = _PLAYER_TOKEN.sub("", text).strip(" ,;")
        if not tokens:
            raise ValueError("type team + number, e.g. TOR 34")
        if leftover:
            raise ValueError(f"can't parse {leftover!r}")
 
        ids = []
        for team, digits in tokens:
            number = int(digits)
            if team:
                player_id = self._by_tag.get((team.upper(), number))
                if player_id is None:
                    raise ValueError(f"no {team.upper()} {number} in this goal")
            else:
                matches = [tag for tag in self._by_tag if tag[1] == number]
                if not matches:
                    raise ValueError(f"no player #{number} in this goal")
                if len(matches) > 1:
                    teams = " / ".join(tag[0] for tag in matches)
                    raise ValueError(f"#{number} is on {teams}: add the team")
                player_id = self._by_tag[matches[0]]
 
            if player_id in ids:
                raise ValueError(f"{self.label(player_id)} entered twice")
            ids.append(player_id)
        return ids
 
 
class AnnotationStore:
    """Keyframes with carry-forward lookup and undo."""
 
    def __init__(self, frames):
        self.frames = [int(f) for f in frames]
        self._frame_set = set(self.frames)
        self._keys = {}        # frame_idx -> Annotation
        self._sorted = []      # keyframe frame_idx values, ascending
        self._undo = []        # (frame_idx, previous Annotation or None)
 
    @property
    def n_keyframes(self):
        return len(self._keys)
 
    def _reindex(self):
        self._sorted = sorted(self._keys)
 
    def set(self, frame, ann, record=True):
        if frame not in self._frame_set:
            raise ValueError(f"frame_idx {frame} is not in this replay.")
        if record:
            self._undo.append((frame, self._keys.get(frame)))
        self._keys[frame] = ann
        self._reindex()
 
    def clear(self, frame):
        """Delete the keyframe at this frame. Returns False if there isn't one."""
        if frame not in self._keys:
            return False
        self._undo.append((frame, self._keys.pop(frame)))
        self._reindex()
        return True
 
    def undo(self):
        """Undo the last change. Returns the frame it affected, or None."""
        if not self._undo:
            return None
        frame, previous = self._undo.pop()
        if previous is None:
            self._keys.pop(frame, None)
        else:
            self._keys[frame] = previous
        self._reindex()
        return frame
 
    def at(self, frame):
        """(annotation in effect at frame, frame of the keyframe it came from)."""
        i = bisect_right(self._sorted, frame) - 1
        if i < 0:
            return None, None
        return self._keys[self._sorted[i]], self._sorted[i]
 
    def prev_keyframe(self, frame):
        i = bisect_left(self._sorted, frame) - 1
        return self._sorted[i] if i >= 0 else None
 
    def next_keyframe(self, frame):
        i = bisect_right(self._sorted, frame)
        return self._sorted[i] if i < len(self._sorted) else None
 
    def n_unannotated(self):
        return sum(self.at(f)[0] is None for f in self.frames)
 
 
# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------
 
def _write_parquet(df, path):
    """Write with an explicit schema so every goal's file has identical types.
 
    (Without it, a goal with no contested frames would store contestor_ids as
    list<null>, and files would fail to combine.)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
 
    schema = pa.schema([
        ("game_id", pa.int64()),
        ("goal_number", pa.int64()),
        ("event_id", pa.int64()),
        ("frame_idx", pa.int64()),
        ("puck_state", pa.string()),
        ("possessor_id", pa.int64()),
        ("possessor_team_abbrev", pa.string()),
        ("possessor_sweater_number", pa.int64()),
        ("contestor_ids", pa.list_(pa.int64())),
        ("contestor_team_abbrevs", pa.list_(pa.string())),
        ("contestor_sweater_numbers", pa.list_(pa.int64())),
        ("is_keyframe", pa.bool_()),
    ])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)
 
 
def _read_parquet(path):
    return pd.read_parquet(path)
 
 
def annotations_to_dataframe(store, roster, game_id, goal_number, event_id):
    """One row per frame, with carried-forward annotations filled in."""
    rows = []
    for frame in store.frames:
        ann, _ = store.at(frame)
 
        possessor = {}
        if ann is not None and ann.possessor_id is not None:
            team, number = roster.tag(ann.possessor_id)
            possessor = dict(
                possessor_id=ann.possessor_id,
                possessor_team_abbrev=team,
                possessor_sweater_number=number,
            )
 
        contestor_tags = (
            [roster.tag(pid) for pid in ann.contestor_ids] if ann else []
        )
        rows.append({
            "game_id": game_id,
            "goal_number": goal_number,
            "event_id": event_id,
            "frame_idx": frame,
            "puck_state": ann.state if ann else None,
            "possessor_id": possessor.get("possessor_id"),
            "possessor_team_abbrev": possessor.get("possessor_team_abbrev"),
            "possessor_sweater_number": possessor.get("possessor_sweater_number"),
            "contestor_ids": list(ann.contestor_ids) if ann else [],
            "contestor_team_abbrevs": [team for team, _ in contestor_tags],
            "contestor_sweater_numbers": [number for _, number in contestor_tags],
            "is_keyframe": frame in store._keys,
        })
 
    return pd.DataFrame(rows).astype({
        "puck_state": "string",
        "possessor_id": "Int64",
        "possessor_team_abbrev": "string",
        "possessor_sweater_number": "Int64",
    })
 
 
def load_keyframes(store, roster, saved, game_id, event_id):
    """Restore keyframes from a saved file so annotation can resume."""
    if not (saved["game_id"].eq(game_id).all() and saved["event_id"].eq(event_id).all()):
        raise ValueError("Saved annotations are for a different game or event.")
 
    for row in saved.loc[saved["is_keyframe"]].itertuples(index=False):
        possessor = None if pd.isna(row.possessor_id) else int(row.possessor_id)
        contestors = tuple(int(pid) for pid in row.contestor_ids)
        unknown = [p for p in (possessor, *contestors) if p is not None and p not in roster]
        if unknown:
            raise ValueError(f"Saved annotations use unknown player_id(s): {unknown}")
        store.set(
            int(row.frame_idx),
            Annotation(row.puck_state, possessor, contestors),
            record=False,
        )
 
 
# ----------------------------------------------------------------------
# Annotator and controller
# ----------------------------------------------------------------------
 
def _free_default_keymaps():
    """Stop matplotlib's built-in shortcuts from firing while we type.
 
    Matplotlib's own handler sees every key press too, so typing 'tor' would
    otherwise trigger zoom ('o'), home ('r'), and so on. Annotating doesn't
    need them. Shortcuts with a modifier (ctrl+w) are kept for quitting.
    """
    for name in list(mpl.rcParams):
        if not name.startswith("keymap."):
            continue
        keep = []
        if name == "keymap.quit":
            keep = [key for key in mpl.rcParams[name] if "+" in key]
        mpl.rcParams[name] = keep
 
 
class PossessionAnnotator:
    """Holds the annotations for one goal and draws their status on the replay.
 
    Create it, then hand it to AnnotatingController, which routes key presses
    to handle_key() and calls refresh() whenever the frame changes.
    """
 
    def __init__(self, replay, event_data, goal_number, out_dir=DEFAULT_OUT_DIR):
        self.replay = replay
        self.controller = None  # set by AnnotatingController
        self.goal_number = int(goal_number)
        self.game_id = _single_value(event_data, "game_id")
        self.event_id = _single_value(event_data, "event_id")
 
        goal_frame = _single_value(event_data, "goal_frame_idx")
        if replay.frames.max() > goal_frame:
            raise ValueError(
                f"The replay runs past goal_frame_idx {goal_frame}. Build it "
                "with frames_through_goal(event_data, frames)."
            )
 
        self.path = Path(out_dir) / f"{self.game_id}_{self.goal_number:02d}.parquet"
        self.roster = Roster(replay.player_keys)
        self.store = AnnotationStore(replay.frames)
 
        # player_id -> index along the second axis of replay.player_xy
        self._player_index = {
            int(p["player_id"]): j
            for j, p in enumerate(replay.player_keys)
            if pd.notna(p["player_id"])
        }
 
        self._mode = None       # None, "possessed" or "contested" (typing players)
        self._buffer = ""
        self._message = ""
        self._dirty = False
 
        if self.path.exists():
            saved = _read_parquet(self.path)
            load_keyframes(self.store, self.roster, saved, self.game_id, self.event_id)
            self._message = f"resumed {self.store.n_keyframes} keyframes"
 
        self._create_artists()
        _free_default_keymaps()
        replay.fig.canvas.mpl_connect("close_event", self._on_close)
 
    @property
    def is_typing(self):
        return self._mode is not None
 
    @property
    def current_frame(self):
        return int(self.replay.frames[self.controller.position])
 
    # -- keys --------------------------------------------------------
 
    def handle_key(self, key):
        """Process a key press. Returns True if the annotator used the key."""
        if key is None:
            return False
 
        if self.is_typing:
            self._message = ""
            self._typing_key(key)
        elif key in STATE_KEYS or key in COMMAND_KEYS:
            self._message = ""
            self.controller.pause()
            self._command_key(key)
        else:
            return False  # arrows, space, ...: left to the replay controller
 
        self.refresh()
        return True
 
    def _command_key(self, key):
        frame = self.current_frame
 
        if key in STATE_KEYS:
            state = STATE_KEYS[key]
            if state in ("possessed", "contested"):
                self._mode, self._buffer = state, ""
            else:
                self._commit(frame, Annotation(state))
 
        elif key == "z":
            undone = self.store.undo()
            if undone is None:
                self._message = "nothing to undo"
            else:
                self._dirty = True
                self.controller.go_to_frame(undone)
                self._message = f"undid change at frame {undone}"
 
        elif key in ("backspace", "delete"):
            if self.store.clear(frame):
                self._dirty = True
            else:
                self._message = "no keyframe at this frame"
 
        elif key in ("[", "]"):
            if key == "[":
                target = self.store.prev_keyframe(frame)
            else:
                target = self.store.next_keyframe(frame)
            if target is None:
                self._message = "no more keyframes that way"
            else:
                self.controller.go_to_frame(target)
 
        elif key == "ctrl+s":
            self.save()
 
    def _typing_key(self, key):
        if key == "escape":
            self._mode, self._buffer = None, ""
        elif key == "enter":
            self._commit_typed()
        elif key == "backspace":
            self._buffer = self._buffer[:-1]
        elif key in (" ", "space"):
            self._buffer += " "
        elif len(key) == 1:
            self._buffer += key.upper()
        # anything else (arrows, shift, ...) is ignored while typing
 
    def _commit_typed(self):
        try:
            ids = self.roster.parse(self._buffer)
            self._check_tracked(ids)
            if self._mode == "possessed":
                if len(ids) != 1:
                    raise ValueError("possessed takes exactly one player")
                ann = Annotation("possessed", possessor_id=ids[0])
            else:
                ann = Annotation("contested", contestor_ids=tuple(ids))
        except ValueError as err:
            self._message = f"! {err}"  # stay in entry mode so it can be fixed
            return
 
        self._mode, self._buffer = None, ""
        self._commit(self.current_frame, ann)
 
    def _check_tracked(self, player_ids):
        """Refuse players with no tracking at this frame (the model can't see them)."""
        position = self.controller.position
        for pid in player_ids:
            if not self.replay.player_valid[position, self._player_index[pid]]:
                raise ValueError(
                    f"{self.roster.label(pid)} has no tracking at frame {self.current_frame}"
                )
 
    def _commit(self, frame, ann):
        self.store.set(frame, ann)
        self._dirty = True
 
    # -- saving ------------------------------------------------------
 
    def _n_untracked_frames(self):
        """Frames whose annotated player(s) have no tracking (e.g. left the ice)."""
        count = 0
        for position, frame in enumerate(self.replay.frames):
            ann, _ = self.store.at(int(frame))
            if ann is None:
                continue
            ids = list(ann.contestor_ids)
            if ann.possessor_id is not None:
                ids.append(ann.possessor_id)
            if any(not self.replay.player_valid[position, self._player_index[p]] for p in ids):
                count += 1
        return count
 
    def save(self):
        df = annotations_to_dataframe(
            self.store, self.roster, self.game_id, self.goal_number, self.event_id
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, self.path)
        self._dirty = False
 
        notes = []
        n_unannotated = self.store.n_unannotated()
        n_untracked = self._n_untracked_frames()
        if n_unannotated:
            notes.append(f"{n_unannotated} frames unannotated")
        if n_untracked:
            notes.append(f"{n_untracked} frames where an annotated player has no tracking")
        self._message = f"saved {self.path.name}" + (f" ({'; '.join(notes)})" if notes else "")
 
    def _on_close(self, _event):
        if self._dirty:
            self.save()
            print(self._message)
 
    # -- display -----------------------------------------------------
 
    def _create_artists(self):
        ax = self.replay.ax
 
        self.status_text = ax.text(
            0.985, 0.985, "",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4),
            zorder=self.replay.Z_CLOCK,
        )
        self.replay.fig.text(
            0.5, 0.004,
            "1 possessed  2 free  3 contested  4 undetermined  5 faceoff  6 post goal"
            "   z undo   backspace clear   [ ] keyframes   ctrl+s save",
            ha="center", va="bottom", fontsize=7, color="gray",
        )
 
        # Rings around the annotated players, so you can check what you typed.
        self.rings = ax.scatter(
            [], [], s=620, facecolors="none", linewidths=3,
            zorder=self.replay.Z_PLAYER_LABEL + 1,
        )
 
    def _describe(self, ann):
        if ann is None:
            return "UNANNOTATED"
        if ann.state == "possessed":
            return f"POSSESSED  {self.roster.label(ann.possessor_id)}"
        if ann.state == "contested":
            return "CONTESTED  " + ", ".join(self.roster.label(p) for p in ann.contestor_ids)
        return ann.state.upper()
 
    def _update_rings(self, position, ann):
        targets = []
        if ann is not None:
            if ann.possessor_id is not None:
                targets.append((ann.possessor_id, POSSESSOR_COLOUR))
            targets += [(pid, CONTESTOR_COLOUR) for pid in ann.contestor_ids]
 
        offsets, colours = [], []
        for pid, colour in targets:
            j = self._player_index[pid]
            if self.replay.player_valid[position, j]:
                offsets.append(self.replay.player_xy[position, j])
                colours.append(colour)
 
        self.rings.set_offsets(np.array(offsets).reshape(-1, 2))
        self.rings.set_edgecolor(colours if colours else "none")
 
    def refresh(self):
        """Redraw the status box and rings for the current frame."""
        frame = self.current_frame
        ann, source = self.store.at(frame)
        self._update_rings(self.controller.position, ann)
 
        if self.is_typing:
            text = f"{self._mode}> {self._buffer}_"
        else:
            text = self._describe(ann)
            if ann is not None:
                text += "  [set here]" if source == frame else f"  [from {source}]"
        if self._message:
            text += f"\n{self._message}"
 
        self.status_text.set_text(text)
        self.replay.fig.canvas.draw_idle()
 
 
class AnnotatingController(ReplayController):
    """ReplayController that also routes key presses to a PossessionAnnotator."""
 
    def __init__(self, replay, annotator, interval=100):
        # The base class calls _set_frame() while it is being constructed,
        # so the annotator must be attached first.
        self.annotator = annotator
        annotator.controller = self
        super().__init__(replay, interval)
 
    def _set_frame(self, position):
        super()._set_frame(position)
        self.annotator.refresh()
 
    def _on_key_press(self, event):
        if not self.annotator.handle_key(event.key):
            super()._on_key_press(event)
 
    def go_to_frame(self, frame_idx):
        """Pause and show the frame with this frame_idx."""
        self.pause()
        self._set_frame(int(np.searchsorted(self.replay.frames, frame_idx)))
 
