#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_fitness_tracker.py - AI Fitness Studio (100% local / offline, single file)

Exercise modes (keyboard OR mouse):
    1  Bicep Curls      elbow angle, reps, elbow-swing check
    2  Squats           knee angle, reps, torso-lean check
    3  Push-ups         elbow angle, reps, body-line check (sag / pike)
    4  Shoulder Press   both arms overhead, reps, arm-symmetry and elbow-drop check
    5  Plank Hold       isometric timer - only counts while the body line is correct
    R  Reset the current exercise (reps, hold timer, calories)
    M  Mute / unmute the voice coach
    Q  Quit (saves fitness_log.csv and shows the 3-second summary dashboard)

Stack : Python 3.10-3.12, OpenCV, MediaPipe Pose (legacy `solutions` API), NumPy,
        pyttsx3 (optional offline text-to-speech).

Install:
    pip install opencv-python numpy mediapipe==0.10.14 pyttsx3

Usage:
    python vision_fitness_tracker.py
    python vision_fitness_tracker.py --camera 1 --goal 10 --weight 80
    python vision_fitness_tracker.py --no-voice --no-mirror
    python vision_fitness_tracker.py --list-cameras
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
except (ImportError, AttributeError) as _exc:  # pragma: no cover
    sys.exit(
        "[fatal] MediaPipe with the legacy 'solutions' API is required.\n"
        "        Fix:  pip install mediapipe==0.10.14   (Python 3.10-3.12)\n"
        f"        Details: {_exc}"
    )

try:
    import pyttsx3
except ImportError:  # voice is optional - the app still runs without it
    pyttsx3 = None

Point = Tuple[float, float]
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


# =============================================================================
# CONFIG  (tweak these to taste)
# =============================================================================
DISPLAY_WIDTH = 960                 # frames are scaled to this width (aspect kept)
DOCK_HEIGHT = 96                    # height of the clickable control dock under the video
CAMERA_REQUEST_SIZE = (1280, 720)
VISIBILITY_MIN = 0.5                # landmark visibility needed to trust a joint
SKELETON_VISIBILITY_MIN = 0.4       # looser threshold, only used for drawing the skeleton
SMOOTHING_ALPHA = 0.35              # EMA on angles: 1.0 = no smoothing, lower = smoother
SIDE_SWITCH_MARGIN = 0.15           # how much more visible the other body side must be to switch
LOST_RESET_FRAMES = 15              # frames without tracking before the angle filter resets
POSTURE_DEBOUNCE_FRAMES = 8         # consecutive bad frames before a warning appears
VOICE_COOLDOWN_S = 4.0              # minimum seconds between spoken form warnings
MAX_FRAME_DT = 0.25                 # clamp frame time so pauses never inflate timers
DEFAULT_GOAL = 12                   # reps per set
DEFAULT_WEIGHT_KG = 70.0            # used for calorie estimates (MET x kg x hours)
HOLD_GOAL_S = 60                    # plank goal shown in the HUD
PLANK_MILESTONES = (15, 30, 45, 60, 90, 120, 180, 300)

CURL_ELBOW_SWING_MAX_DEG = 40.0     # hip-shoulder-elbow angle above this = elbow swinging
SQUAT_TORSO_LEAN_MAX_DEG = 60.0     # torso lean from vertical above this = slouching
PUSHUP_BODY_LINE_MIN_DEG = 150.0    # shoulder-hip-ankle angle below this = broken body line
PLANK_BODY_LINE_MIN_DEG = 170.0     # 170-180 = straight line (good band); below = sag / pike
PRESS_ARM_SYMMETRY_MAX_DEG = 25.0   # max difference between left and right elbow angles
PRESS_ELBOW_DROP_MAX_RATIO = 0.30   # elbow may sit at most 30% of torso length below shoulder

LOG_FILENAME = "fitness_log.csv"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILENAME)
LOG_HEADER = ["Date", "Exercise", "Reps", "Plank_Time_s", "Calories_kcal",
              "Exercise_Duration_s", "Session_Duration_s"]

# Colours (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (165, 165, 165)
DIM = (80, 80, 80)
PANEL = (18, 18, 22)
NEON_GREEN = (80, 255, 57)
YELLOW = (0, 235, 255)
CRIMSON = (60, 40, 235)
CYAN = (255, 240, 0)
ELECTRIC_BLUE = (255, 115, 20)
ORANGE = (0, 150, 255)
SKELETON_COLORS = {"GOOD": NEON_GREEN, "TRANSITION": YELLOW, "BAD": CRIMSON, "LOST": GREY}

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


# =============================================================================
# Errors & small helpers
# =============================================================================
class LandmarkNotVisible(Exception):
    """Raised when a landmark we need is hidden or outside the camera frame."""


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def calculate_angle(a, b, c) -> float:
    """Angle (0-180 degrees) at vertex b formed by a-b-c. Stable for any pose."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(float(np.degrees(radians)))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


class AngleSmoother:
    """Exponential moving average that removes landmark jitter from angles."""

    def __init__(self, alpha: float = SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class FPSCounter:
    """Rolling-average frames-per-second counter."""

    def __init__(self, window: int = 30) -> None:
        self.dts: deque = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self.dts.append(now - self.last)
        self.last = now
        avg = sum(self.dts) / len(self.dts)
        return 1.0 / avg if avg > 0 else 0.0


# =============================================================================
# Exercise definitions
# =============================================================================
@dataclass(frozen=True)
class Exercise:
    key: str                          # internal id
    name: str                         # shown in the HUD
    spoken: str                       # spoken on mode change
    joints: Tuple[str, str, str]      # (first, vertex, last) joint names for the rep angle
    met: float                        # metabolic equivalent (calorie estimate)
    rest_angle: float = 0.0           # angle at the rest position of a rep
    peak_angle: float = 0.0           # angle at the peak of a rep (rep is counted here)
    rest_high: bool = True            # True: rest = large angle (curls, squats...)
    rest_stage: str = "DOWN"          # stage label while at rest
    peak_stage: str = "UP"            # stage label at the peak
    bilateral: bool = False           # True: both arms are tracked and averaged
    is_hold: bool = False             # True: isometric timer instead of reps


CURL = Exercise(key="curl", name="BICEP CURLS", spoken="Bicep curls",
                joints=("shoulder", "elbow", "wrist"), met=3.5,
                rest_angle=160.0, peak_angle=35.0, rest_high=True,
                rest_stage="DOWN", peak_stage="UP")
SQUAT = Exercise(key="squat", name="SQUATS", spoken="Squats",
                 joints=("hip", "knee", "ankle"), met=5.0,
                 rest_angle=165.0, peak_angle=90.0, rest_high=True,
                 rest_stage="UP", peak_stage="DOWN")
PUSHUP = Exercise(key="pushup", name="PUSH-UPS", spoken="Push ups",
                  joints=("shoulder", "elbow", "wrist"), met=3.8,
                  rest_angle=160.0, peak_angle=90.0, rest_high=True,
                  rest_stage="UP", peak_stage="DOWN")
PRESS = Exercise(key="press", name="SHOULDER PRESS", spoken="Shoulder press",
                 joints=("shoulder", "elbow", "wrist"), met=4.0,
                 rest_angle=95.0, peak_angle=160.0, rest_high=False,
                 rest_stage="DOWN", peak_stage="UP", bilateral=True)
PLANK = Exercise(key="plank", name="PLANK HOLD", spoken="Plank hold",
                 joints=("shoulder", "hip", "ankle"), met=3.8, is_hold=True)

EXERCISES: Tuple[Exercise, ...] = (CURL, SQUAT, PUSHUP, PRESS, PLANK)
EXERCISE_BY_KEY: Dict[str, Exercise] = {ex.key: ex for ex in EXERCISES}
KEY_BINDINGS: Dict[str, Exercise] = {"1": CURL, "2": SQUAT, "3": PUSHUP, "4": PRESS, "5": PLANK}

_PL = mp_pose.PoseLandmark
SIDE_LANDMARKS: Dict[str, Dict[str, "mp_pose.PoseLandmark"]] = {
    "LEFT": {
        "shoulder": _PL.LEFT_SHOULDER, "elbow": _PL.LEFT_ELBOW, "wrist": _PL.LEFT_WRIST,
        "hip": _PL.LEFT_HIP, "knee": _PL.LEFT_KNEE, "ankle": _PL.LEFT_ANKLE,
    },
    "RIGHT": {
        "shoulder": _PL.RIGHT_SHOULDER, "elbow": _PL.RIGHT_ELBOW, "wrist": _PL.RIGHT_WRIST,
        "hip": _PL.RIGHT_HIP, "knee": _PL.RIGHT_KNEE, "ankle": _PL.RIGHT_ANKLE,
    },
}


# =============================================================================
# Repetition counter (two-stage state machine, works in both directions)
# =============================================================================
class RepCounter:
    """
    Counts a rep only after a full range of motion:
      at rest  -> rest_stage
      at peak while in rest_stage -> peak_stage, reps += 1
    """

    def __init__(self, exercise: Exercise) -> None:
        self.exercise = exercise
        self.reps = 0
        self.stage: Optional[str] = None

    def reset(self) -> None:
        self.reps = 0
        self.stage = None

    def _at_rest(self, angle: float) -> bool:
        ex = self.exercise
        return angle > ex.rest_angle if ex.rest_high else angle < ex.rest_angle

    def _at_peak(self, angle: float) -> bool:
        ex = self.exercise
        return angle < ex.peak_angle if ex.rest_high else angle > ex.peak_angle

    def update(self, angle: float) -> bool:
        """Feed the latest angle. Returns True on the frame a rep is completed."""
        ex = self.exercise
        if self._at_rest(angle):
            self.stage = ex.rest_stage
            return False
        if self._at_peak(angle) and self.stage == ex.rest_stage:
            self.stage = ex.peak_stage
            self.reps += 1
            return True
        return False

    def progress(self, angle: float) -> float:
        """0.0 at rest -> 1.0 at the peak of the movement."""
        ex = self.exercise
        span = ex.peak_angle - ex.rest_angle
        return float(clamp((angle - ex.rest_angle) / span, 0.0, 1.0))


# =============================================================================
# Form / posture analysis
# =============================================================================
@dataclass
class PostureResult:
    bad: Optional[bool]               # None = could not be evaluated
    value: Optional[float] = None     # the measured angle / lean / asymmetry
    message: str = ""                 # on-screen warning text
    voice: str = ""                   # spoken warning text


def _have(pts: Optional[Dict[str, Optional[Point]]], *names: str) -> bool:
    return pts is not None and all(pts.get(n) is not None for n in names)


def _body_line(p: Dict[str, Optional[Point]]) -> Tuple[float, float]:
    """
    Returns (shoulder-hip-ankle angle, hip deviation).
    Angle ~180 = straight. Deviation > 0 = hip sits BELOW the shoulder-ankle line (sagging),
    deviation < 0 = hip sits ABOVE it (piking / hips too high).
    """
    sh, hip, ankle = p["shoulder"], p["hip"], p["ankle"]
    line = calculate_angle(sh, hip, ankle)
    dx = ankle[0] - sh[0]
    dev = 0.0
    if abs(dx) > 1e-3:
        y_line = sh[1] + (ankle[1] - sh[1]) * (hip[0] - sh[0]) / dx
        dev = hip[1] - y_line
    return line, dev


def _line_feedback(dev: float) -> Tuple[str, str]:
    if dev > 0:
        return ("HIPS SAGGING - SQUEEZE GLUTES", "Your hips are sagging. Squeeze your glutes.")
    return ("HIPS TOO HIGH - LOWER YOUR HIPS", "Lower your hips into a straight line.")


def check_form(ex: Exercise, sides: Dict[str, Dict[str, Optional[Point]]], primary: str) -> PostureResult:
    """Evaluate form for the active exercise. Never raises; unknown -> bad=None."""
    try:
        if ex.key == "press":
            left, right = sides.get("LEFT"), sides.get("RIGHT")
            if not (_have(left, *ex.joints) and _have(right, *ex.joints)):
                return PostureResult(None)
            ang_l = calculate_angle(left["shoulder"], left["elbow"], left["wrist"])
            ang_r = calculate_angle(right["shoulder"], right["elbow"], right["wrist"])
            asym = abs(ang_l - ang_r)
            if asym > PRESS_ARM_SYMMETRY_MAX_DEG:
                return PostureResult(True, asym, "PRESS BOTH ARMS EVENLY",
                                     "Press both arms evenly")
            if _have(left, "hip") and _have(right, "hip"):
                torso = (math.dist(left["shoulder"], left["hip"]) +
                         math.dist(right["shoulder"], right["hip"])) / 2.0
                drop = ((left["elbow"][1] - left["shoulder"][1]) +
                        (right["elbow"][1] - right["shoulder"][1])) / 2.0
                if torso > 1e-3 and drop > PRESS_ELBOW_DROP_MAX_RATIO * torso:
                    return PostureResult(True, drop, "ELBOWS TOO LOW - PRESS FROM SHOULDER HEIGHT",
                                         "Keep your elbows at shoulder height")
            return PostureResult(False, (ang_l + ang_r) / 2.0)

        p = sides.get(primary) or {}

        if ex.key == "curl":
            if not _have(p, "hip", "shoulder", "elbow"):
                return PostureResult(None)
            swing = calculate_angle(p["hip"], p["shoulder"], p["elbow"])
            return PostureResult(swing > CURL_ELBOW_SWING_MAX_DEG, swing,
                                 "PIN YOUR ELBOW TO YOUR SIDE!",
                                 "Keep your elbow close to your body")

        if ex.key == "squat":
            if not _have(p, "shoulder", "hip"):
                return PostureResult(None)
            sx, sy = p["shoulder"]
            hx, hy = p["hip"]
            lean = float(np.degrees(np.arctan2(abs(sx - hx), hy - sy)))  # 0 = upright
            return PostureResult(lean > SQUAT_TORSO_LEAN_MAX_DEG, lean,
                                 "FIX BACK POSTURE - CHEST UP!",
                                 "Keep your back straight and chest up")

        if ex.key == "pushup":
            if not _have(p, "shoulder", "hip", "ankle"):
                return PostureResult(None)
            line, dev = _body_line(p)
            msg, voice = _line_feedback(dev)
            return PostureResult(line < PUSHUP_BODY_LINE_MIN_DEG, line, msg, voice)

        if ex.key == "plank":
            if not _have(p, "shoulder", "hip", "ankle"):
                return PostureResult(None)
            line, dev = _body_line(p)
            msg, voice = _line_feedback(dev)
            return PostureResult(line < PLANK_BODY_LINE_MIN_DEG, line, msg, voice)
    except Exception:  # posture must never break rep counting
        return PostureResult(None)
    return PostureResult(None)


# =============================================================================
# Non-blocking offline voice coach (pyttsx3 in its own daemon thread)
# =============================================================================
class VoiceAssistant:
    """
    pyttsx3 lives in its own daemon thread, so speech can never stall the video loop.

    * alert(text)    - form warnings; throttled by a cooldown.
    * announce(text) - high priority (rep counts, milestones, mode changes); bypasses the
                       cooldown and drops queued speech so it is never stale.
    * set_muted()    - muting drops anything queued and silences all further speech.
    """

    def __init__(self, enabled: bool = True, cooldown: float = VOICE_COOLDOWN_S,
                 rate: int = 175, volume: float = 1.0) -> None:
        self.cooldown = cooldown
        self._rate = rate
        self._volume = volume
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=3)
        self._last_spoken = -1e9
        self._available = False
        self._thread: Optional[threading.Thread] = None
        self.muted = False

        if not enabled:
            return
        if pyttsx3 is None:
            print("[voice] pyttsx3 is not installed - voice alerts disabled.")
            return
        self._available = True
        self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()

    # ---- public API (main/video thread; never blocks) ------------------------
    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            self._drain()

    def alert(self, text: str) -> bool:
        now = time.monotonic()
        if not self._available or self.muted or (now - self._last_spoken) < self.cooldown:
            return False
        self._last_spoken = now
        self._enqueue(text)
        return True

    def announce(self, text: str) -> None:
        if not self._available or self.muted:
            return
        self._last_spoken = time.monotonic()
        self._drain()
        self._enqueue(text)

    def shutdown(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.5)
            except queue.Full:
                pass
            self._thread.join(timeout=5.0)

    # ---- internals -------------------------------------------------------------
    def _enqueue(self, item: Optional[str]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass  # dropping speech is always better than blocking the video

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _worker(self) -> None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
        except Exception as exc:
            print(f"[voice] Could not start the speech engine ({exc}). Voice disabled.")
            self._available = False
            return

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[voice] Speech error: {exc}")
        try:
            engine.stop()
        except Exception:
            pass


# =============================================================================
# Drawing primitives: text, glass panels, gradients
# =============================================================================
def put_text(img, text: str, org: Tuple[int, int], scale: float = 0.6,
             color=WHITE, thickness: int = 1) -> None:
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, FONT, scale, BLACK, thickness + 3, AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, AA)


def put_text_centered(img, text: str, cx: int, baseline_y: int, scale: float,
                      color=WHITE, thickness: int = 2, max_width: Optional[int] = None) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    while max_width is not None and tw > max_width and scale > 0.35:
        scale -= 0.05
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    put_text(img, text, (int(cx - tw / 2), baseline_y), scale, color, thickness)


@lru_cache(maxsize=64)
def _rounded_alpha(width: int, height: int, r: int) -> np.ndarray:
    """Cached float mask (H, W, 1) with 1.0 inside a rounded rectangle."""
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (r, 0), (width - r - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (width - 1, height - r - 1), 255, -1)
    for cx, cy in ((r, r), (width - r - 1, r), (r, height - r - 1), (width - r - 1, height - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return (mask.astype(np.float32) / 255.0)[..., None]


def _outline(img, x1, y1, x2, y2, r, color, t) -> None:
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, t, AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, t, AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, t, AA)


def rounded_outline(img, x1, y1, x2, y2, r, color, thickness=2, glow=True) -> None:
    """Neon border: a dimmed wide stroke underneath gives a cheap glow."""
    if glow:
        dim = tuple(int(c * 0.35) for c in color)
        _outline(img, x1, y1, x2, y2, r, dim, thickness + 5)
    _outline(img, x1, y1, x2, y2, r, color, thickness)


def rounded_panel(img, x1: int, y1: int, x2: int, y2: int, radius: int = 14,
                  color=PANEL, alpha: float = 0.6, border=None, glow: bool = True,
                  border_thickness: int = 2) -> None:
    """Glassmorphism panel: semi-transparent rounded rectangle with optional neon border."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    rw, rh = x2 - x1, y2 - y1
    if rw < 4 or rh < 4:
        return
    r = max(1, min(radius, rw // 2, rh // 2))
    m = _rounded_alpha(rw, rh, r) * alpha
    roi = img[y1:y2, x1:x2].astype(np.float32)
    col = np.array(color, np.float32)
    img[y1:y2, x1:x2] = (roi * (1.0 - m) + col * m).astype(np.uint8)
    if border is not None:
        rounded_outline(img, x1, y1, x2 - 1, y2 - 1, r, border, border_thickness, glow)


def progress_color(pct: float) -> Tuple[int, int, int]:
    """Red -> yellow -> green gradient (BGR)."""
    pct = float(clamp(pct, 0.0, 1.0))
    if pct < 0.5:
        return (0, int(255 * pct * 2), 255)
    return (0, 255, int(255 * (1.0 - (pct - 0.5) * 2)))


@lru_cache(maxsize=8)
def gradient_column(height: int) -> np.ndarray:
    """Per-row colours for a vertical bar: red at the bottom, green at the top. Shape (H, 1, 3)."""
    col = np.zeros((height, 1, 3), np.uint8)
    for i in range(height):
        value = 1.0 - i / max(1, height - 1)     # 1.0 at top, 0.0 at bottom
        col[i, 0] = progress_color(value)
    return col


def _dock_background(w: int, h: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array((46, 30, 22), np.float32)
    bottom = np.array((12, 8, 8), np.float32)
    col = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    bg = np.repeat(col, w, axis=1)
    cv2.line(bg, (0, 0), (w, 0), CYAN, 1, AA)
    return bg


# =============================================================================
# Programmatic vector icons (OpenCV geometry only - no image files)
# =============================================================================
def _pt(cx, cy, s, dx, dy) -> Tuple[int, int]:
    return (int(round(cx + dx * s)), int(round(cy + dy * s)))


def _poly(cx, cy, s, coords) -> np.ndarray:
    return np.array([[int(round(cx + x * s)), int(round(cy + y * s))] for x, y in coords],
                    np.int32).reshape((-1, 1, 2))


def icon_dumbbell(img, cx, cy, s, color, t=2) -> None:
    cv2.line(img, _pt(cx, cy, s, -0.5, 0), _pt(cx, cy, s, 0.5, 0), color, t, AA)
    for sign in (-1, 1):
        for dx, pw, ph in ((0.30, 0.10, 0.56), (0.42, 0.08, 0.38)):
            x = cx + sign * dx * s
            cv2.rectangle(img, (int(x - pw * s / 2), int(cy - ph * s / 2)),
                          (int(x + pw * s / 2), int(cy + ph * s / 2)), color, -1, AA)


def icon_flame(img, cx, cy, s, color, inner=WHITE) -> None:
    outer = [(0.0, -0.55), (0.20, -0.22), (0.34, 0.02), (0.30, 0.30), (0.12, 0.50),
             (-0.12, 0.50), (-0.30, 0.30), (-0.34, 0.02), (-0.20, -0.22), (-0.06, -0.08)]
    core = [(0.0, -0.02), (0.13, 0.22), (0.08, 0.40), (-0.08, 0.40), (-0.13, 0.22)]
    cv2.fillPoly(img, [_poly(cx, cy, s, outer)], color, AA)
    cv2.fillPoly(img, [_poly(cx, cy, s, core)], inner, AA)


def icon_stopwatch(img, cx, cy, s, color, t=2) -> None:
    cy2 = cy + int(0.06 * s)
    r = int(0.38 * s)
    cv2.circle(img, (cx, cy2), r, color, t, AA)
    cv2.line(img, _pt(cx, cy, s, -0.12, -0.46), _pt(cx, cy, s, 0.12, -0.46), color, t, AA)
    cv2.line(img, _pt(cx, cy, s, 0.0, -0.46), (cx, cy2 - r), color, t, AA)
    cv2.line(img, (cx, cy2), _pt(cx, cy, s, 0.18, -0.10), color, t, AA)
    cv2.circle(img, (cx, cy2), max(2, int(0.05 * s)), color, -1, AA)


def icon_target(img, cx, cy, s, color, t=2) -> None:
    for f in (0.44, 0.28):
        cv2.circle(img, (cx, cy), int(f * s), color, t, AA)
    cv2.circle(img, (cx, cy), max(2, int(0.08 * s)), color, -1, AA)


def icon_speaker(img, cx, cy, s, color, muted: bool = False, t=2) -> None:
    body = _poly(cx, cy, s, [(-0.42, -0.16), (-0.20, -0.16), (0.02, -0.40),
                             (0.02, 0.40), (-0.20, 0.16), (-0.42, 0.16)])
    cv2.fillPoly(img, [body], color, AA)
    if muted:
        cv2.line(img, _pt(cx, cy, s, 0.22, -0.22), _pt(cx, cy, s, 0.50, 0.22), CRIMSON, t + 1, AA)
        cv2.line(img, _pt(cx, cy, s, 0.22, 0.22), _pt(cx, cy, s, 0.50, -0.22), CRIMSON, t + 1, AA)
    else:
        center = _pt(cx, cy, s, 0.02, 0)
        cv2.ellipse(img, center, (int(0.22 * s), int(0.22 * s)), 0, -55, 55, color, t, AA)
        cv2.ellipse(img, center, (int(0.42 * s), int(0.42 * s)), 0, -55, 55, color, t, AA)


def icon_reset(img, cx, cy, s, color, t=2) -> None:
    r = int(0.36 * s)
    cv2.ellipse(img, (cx, cy), (r, r), 0, -60, 240, color, t, AA)
    ang = math.radians(240)
    px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
    tx, ty = -math.sin(ang), math.cos(ang)       # direction of travel
    nx, ny = math.cos(ang), math.sin(ang)        # radial direction
    tip = (px + tx * 0.22 * s, py + ty * 0.22 * s)
    b1 = (px + nx * 0.16 * s, py + ny * 0.16 * s)
    b2 = (px - nx * 0.16 * s, py - ny * 0.16 * s)
    cv2.fillPoly(img, [np.array([tip, b1, b2], np.int32)], color, AA)


def icon_power(img, cx, cy, s, color, t=2) -> None:
    r = int(0.38 * s)
    cv2.ellipse(img, (cx, cy), (r, r), 0, -60, 240, color, t, AA)
    cv2.line(img, (cx, int(cy - 0.50 * s)), (cx, int(cy - 0.02 * s)), color, t, AA)


def icon_exercise(img, key: str, cx, cy, s, color, t=2) -> None:
    """Stick-figure glyph for each exercise (curls use the dumbbell)."""
    if key == "curl":
        icon_dumbbell(img, cx, cy, s, color, t)
        return

    def P(dx, dy):
        return _pt(cx, cy, s, dx, dy)

    if key == "squat":
        cv2.circle(img, P(-0.10, -0.42), max(2, int(0.11 * s)), color, t, AA)
        cv2.line(img, P(-0.10, -0.28), P(-0.05, 0.05), color, t, AA)
        cv2.line(img, P(-0.05, 0.05), P(0.30, 0.20), color, t, AA)
        cv2.line(img, P(0.30, 0.20), P(0.28, 0.46), color, t, AA)
        cv2.line(img, P(-0.10, -0.22), P(0.30, -0.05), color, t, AA)
        cv2.line(img, P(-0.45, 0.48), P(0.45, 0.48), color, t, AA)
    elif key == "pushup":
        cv2.circle(img, P(-0.45, -0.12), max(2, int(0.10 * s)), color, t, AA)
        cv2.line(img, P(-0.32, -0.04), P(0.42, 0.10), color, t, AA)
        cv2.line(img, P(-0.25, -0.02), P(-0.25, 0.32), color, t, AA)
        cv2.line(img, P(-0.50, 0.36), P(0.50, 0.36), color, t, AA)
    elif key == "press":
        cv2.circle(img, P(0.0, -0.42), max(2, int(0.11 * s)), color, t, AA)
        cv2.line(img, P(0.0, -0.28), P(0.0, 0.10), color, t, AA)
        for sign in (-1, 1):
            cv2.line(img, P(0.0, -0.22), P(sign * 0.30, -0.20), color, t, AA)
            cv2.line(img, P(sign * 0.30, -0.20), P(sign * 0.22, -0.48), color, t, AA)
            cv2.line(img, P(0.0, 0.10), P(sign * 0.14, 0.46), color, t, AA)
    elif key == "plank":
        cv2.circle(img, P(-0.44, -0.14), max(2, int(0.10 * s)), color, t, AA)
        cv2.line(img, P(-0.32, -0.08), P(0.42, -0.04), color, t, AA)
        cv2.line(img, P(-0.30, -0.08), P(-0.30, 0.30), color, t, AA)
        cv2.line(img, P(-0.50, 0.34), P(0.50, 0.34), color, t, AA)


# =============================================================================
# Skeleton, joint arcs
# =============================================================================
def draw_skeleton(frame, landmarks, color, w: int, h: int) -> None:
    """Neon skeleton: blurred glow layer (half resolution for speed) plus crisp core lines."""
    pts: List[Optional[Tuple[int, int]]] = [None] * len(landmarks)
    for i, lm in enumerate(landmarks):
        if lm.visibility >= SKELETON_VISIBILITY_MIN:
            pts[i] = (int(lm.x * w), int(lm.y * h))

    glow = np.zeros((h // 2, w // 2, 3), np.uint8)
    for a, b in POSE_CONNECTIONS:
        p, q = pts[a], pts[b]
        if p is not None and q is not None:
            cv2.line(glow, (p[0] // 2, p[1] // 2), (q[0] // 2, q[1] // 2), color, 5, AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 3)
    glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
    np.copyto(frame, cv2.add(frame, glow))

    for a, b in POSE_CONNECTIONS:
        p, q = pts[a], pts[b]
        if p is not None and q is not None:
            cv2.line(frame, p, q, color, 3, AA)
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 4, WHITE, -1, AA)


def draw_joint_arc(frame, p1: Point, vertex: Point, p2: Point, color, radius: int = 32) -> float:
    """Glowing arc across the joint's interior angle, with the live degree value beside it."""
    angle = calculate_angle(p1, vertex, p2)
    b = (int(round(vertex[0])), int(round(vertex[1])))
    a1 = math.degrees(math.atan2(p1[1] - vertex[1], p1[0] - vertex[0]))
    a2 = math.degrees(math.atan2(p2[1] - vertex[1], p2[0] - vertex[0]))
    d = (a2 - a1 + 180.0) % 360.0 - 180.0          # shortest signed sweep
    start = a1 if d >= 0 else a1 + d
    span = abs(d)
    dim = tuple(int(c * 0.4) for c in color)
    cv2.ellipse(frame, b, (radius, radius), 0, start, start + span, dim, 6, AA)
    cv2.ellipse(frame, b, (radius, radius), 0, start, start + span, color, 2, AA)
    mid = math.radians(start + span / 2.0)
    lx = b[0] + (radius + 18) * math.cos(mid)
    ly = b[1] + (radius + 18) * math.sin(mid)
    label = str(int(round(angle)))
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 2)
    put_text(frame, label, (int(lx - tw / 2), int(ly + th / 2)), 0.5, color, 2)
    cv2.circle(frame, b, 4, color, -1, AA)
    return angle


# =============================================================================
# Session data
# =============================================================================
@dataclass
class ExerciseStats:
    active_s: float = 0.0       # seconds the athlete was tracked on this exercise
    kcal: float = 0.0           # estimated calories burned (MET x kg x hours)
    hold_s: float = 0.0         # plank: time held with correct posture


@dataclass
class FrameState:
    """Everything the renderer needs to know about the current frame."""
    tracking_ok: bool = False
    message: str = ""
    angle: Optional[float] = None
    progress: float = 0.0
    sides: Dict[str, Dict[str, Optional[Point]]] = field(default_factory=dict)
    primary: str = "LEFT"
    posture: Optional[PostureResult] = None
    posture_status: str = "N/A"          # "GOOD" | "FIX" | "N/A"
    skeleton: str = "LOST"               # "GOOD" | "TRANSITION" | "BAD" | "LOST"
    pose_landmarks: object = None


@dataclass
class Button:
    action: str                          # "ex:<key>" | "reset" | "voice" | "quit"
    hotkey: str
    label: str
    icon: str
    x1: int
    y1: int
    x2: int
    y2: int


# =============================================================================
# Post-workout logging and summary dashboard
# =============================================================================
def append_log(rows: List[Dict]) -> None:
    new_file = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(LOG_HEADER)
        for r in rows:
            writer.writerow([r["date"], r["name"], r["reps"], r["hold_s"],
                             r["kcal"], r["active_s"], r["session_s"]])


def build_summary(rows: List[Dict], session_s: float, total_kcal: float,
                  total_active: float, saved: bool) -> np.ndarray:
    W, H = 1000, 620
    t = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    top = np.array((40, 24, 18), np.float32)
    bottom = np.array((10, 7, 8), np.float32)
    img = np.repeat((top * (1.0 - t) + bottom * t).astype(np.uint8), W, axis=1)

    put_text(img, "WORKOUT SUMMARY", (40, 72), 1.2, CYAN, 2)
    put_text(img, datetime.now().strftime("%A, %d %B %Y   %H:%M"), (40, 102), 0.55, GREY, 1)

    cards = [
        ("CALORIES", f"{total_kcal:.1f} kcal", ORANGE, "flame"),
        ("ACTIVE TIME", fmt_clock(total_active), ELECTRIC_BLUE, "watch"),
        ("SESSION", fmt_clock(session_s), NEON_GREEN, "watch"),
    ]
    for i, (label, value, color, icon) in enumerate(cards):
        x1 = 40 + i * 310
        x2 = x1 + 290
        rounded_panel(img, x1, 124, x2, 224, 18, PANEL, 0.7, color, True, 2)
        if icon == "flame":
            icon_flame(img, x1 + 42, 174, 30, color)
        else:
            icon_stopwatch(img, x1 + 42, 174, 30, color, 2)
        put_text(img, label, (x1 + 80, 152), 0.5, GREY, 1)
        put_text(img, value, (x1 + 80, 200), 0.9, WHITE, 2)

    rounded_panel(img, 40, 246, 960, 556, 18, PANEL, 0.7, DIM, False)
    for text, x in (("EXERCISE", 86), ("REPS", 440), ("HOLD (s)", 550), ("KCAL", 690), ("TIME", 820)):
        put_text(img, text, (x, 282), 0.45, GREY, 1)
    cv2.line(img, (60, 294), (940, 294), DIM, 1, AA)

    if not rows:
        put_text(img, "No activity recorded this session - get moving!", (70, 350), 0.8, YELLOW, 2)
    for i, r in enumerate(rows[:6]):
        y = 332 + i * 40
        icon_exercise(img, r["key"], 68, y - 6, 22, CYAN, 2)
        put_text(img, r["name"], (88, y), 0.55, WHITE, 1)
        put_text(img, str(r["reps"]) if r["reps"] else "-", (440, y), 0.55, WHITE, 1)
        put_text(img, f"{r['hold_s']:.1f}" if r["hold_s"] else "-", (550, y), 0.55, WHITE, 1)
        put_text(img, f"{r['kcal']:.1f}", (690, y), 0.55, ORANGE, 1)
        put_text(img, fmt_clock(r["active_s"]), (820, y), 0.55, ELECTRIC_BLUE, 1)

    footer = (f"Saved to {LOG_FILENAME}" if saved else "Nothing to save") + "   |   closing in 3 s"
    put_text(img, footer, (40, H - 22), 0.5, GREY, 1)
    return img


# =============================================================================
# Camera helpers
# =============================================================================
def _backend() -> int:
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def open_camera(preferred_index: int) -> cv2.VideoCapture:
    """Open the requested webcam; if that fails, auto-scan indices 0-4."""
    candidates = [preferred_index] + [i for i in range(5) if i != preferred_index]
    for idx in candidates:
        cap = cv2.VideoCapture(idx, _backend())
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_REQUEST_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_REQUEST_SIZE[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if ok:
            if idx != preferred_index:
                print(f"[camera] Index {preferred_index} unavailable - using camera {idx} instead.")
            else:
                print(f"[camera] Using camera index {idx}.")
            return cap
        cap.release()
    raise RuntimeError(
        "No working webcam found. Close other apps using the camera, check OS camera "
        "permissions, or try --camera 1 / --list-cameras."
    )


def list_cameras(max_index: int = 6) -> None:
    print("Probing camera indices...")
    found = False
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, _backend())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                found = True
                print(f"  index {idx}: OK  ({frame.shape[1]}x{frame.shape[0]})")
        cap.release()
    if not found:
        print("  No cameras returned frames.")


# =============================================================================
# Main application
# =============================================================================
class FitnessTracker:
    WINDOW = "AI Fitness Studio"

    def __init__(self, camera_index: int = 0, mirror: bool = True, voice: bool = True,
                 model_complexity: int = 1, goal: int = DEFAULT_GOAL,
                 weight_kg: float = DEFAULT_WEIGHT_KG) -> None:
        self.mirror = mirror
        self.goal = max(1, goal)
        self.weight_kg = weight_kg
        self.cap = open_camera(camera_index)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.voice = VoiceAssistant(enabled=voice)
        self.fps_counter = FPSCounter()

        self.exercise: Exercise = CURL
        self.counters: Dict[str, RepCounter] = {ex.key: RepCounter(ex) for ex in EXERCISES}
        self.stats: Dict[str, ExerciseStats] = {ex.key: ExerciseStats() for ex in EXERCISES}
        self._hold_done: set = set()
        self.smoother = AngleSmoother()
        self.side: Optional[str] = None
        self.bad_frames = 0
        self.missed_frames = 0
        self.display_pct = 0.0
        self._last_error = ""
        self._dt = 0.0
        self._t_prev = time.perf_counter()
        self._session_t0 = time.perf_counter()
        self._quit = False
        self._hover: Optional[str] = None
        self.buttons: List[Button] = []
        self._canvas: Optional[np.ndarray] = None
        self._dock_bg: Optional[np.ndarray] = None

    @property
    def counter(self) -> RepCounter:
        return self.counters[self.exercise.key]

    def total_active(self) -> float:
        return sum(st.active_s for st in self.stats.values())

    def total_kcal(self) -> float:
        return sum(st.kcal for st in self.stats.values())

    # ---- mode / reset / voice --------------------------------------------------
    def set_exercise(self, exercise: Exercise) -> None:
        if exercise.key == self.exercise.key:
            return
        self.exercise = exercise
        self.counter.stage = None
        self.smoother.reset()
        self.side = None
        self.bad_frames = 0
        self.voice.announce(f"{exercise.spoken} mode")
        print(f"[mode] {exercise.name}")

    def reset_counter(self) -> None:
        key = self.exercise.key
        self.counters[key].reset()
        self.stats[key] = ExerciseStats()
        if self.exercise.is_hold:
            self._hold_done.clear()
        self.smoother.reset()
        self.bad_frames = 0
        self.voice.announce("Stats reset")
        print(f"[reset] {self.exercise.name} cleared")

    def toggle_voice(self) -> None:
        self.voice.set_muted(not self.voice.muted)
        if not self.voice.muted:
            self.voice.announce("Voice coach on")
        print(f"[voice] {'muted' if self.voice.muted else 'unmuted'}")

    # ---- input -------------------------------------------------------------------
    def _on_key(self, ch: str) -> None:
        if ch == "q":
            self._quit = True
        elif ch == "r":
            self.reset_counter()
        elif ch == "m":
            self.toggle_voice()
        elif ch in KEY_BINDINGS:
            self.set_exercise(KEY_BINDINGS[ch])

    def _layout_dock(self, w: int, h: int) -> None:
        specs = [
            ("ex:curl", "1", "CURLS", "curl"),
            ("ex:squat", "2", "SQUATS", "squat"),
            ("ex:pushup", "3", "PUSH-UPS", "pushup"),
            ("ex:press", "4", "PRESS", "press"),
            ("ex:plank", "5", "PLANK", "plank"),
            ("reset", "R", "RESET", "reset"),
            ("voice", "M", "VOICE", "voice"),
            ("quit", "Q", "QUIT", "quit"),
        ]
        margin, gap = 14, 8
        bw = (w - 2 * margin - (len(specs) - 1) * gap) // len(specs)
        y1, y2 = h + 10, h + DOCK_HEIGHT - 10
        self.buttons = []
        for i, (action, hotkey, label, icon) in enumerate(specs):
            x1 = margin + i * (bw + gap)
            self.buttons.append(Button(action, hotkey, label, icon, x1, y1, x1 + bw, y2))

    def _button_at(self, x: int, y: int) -> Optional[str]:
        for b in self.buttons:
            if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                return b.action
        return None

    def _activate(self, action: str) -> None:
        if action.startswith("ex:"):
            self.set_exercise(EXERCISE_BY_KEY[action[3:]])
        elif action == "reset":
            self.reset_counter()
        elif action == "voice":
            self.toggle_voice()
        elif action == "quit":
            self._quit = True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover = self._button_at(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            action = self._button_at(x, y)
            if action is not None:
                self._activate(action)

    # ---- per-frame analysis ----------------------------------------------------
    def _select_side(self, lms) -> str:
        """Track the body side the camera sees best (sticky to avoid flicker)."""
        scores = {
            side: float(np.mean([lms[SIDE_LANDMARKS[side][n].value].visibility
                                 for n in self.exercise.joints]))
            for side in SIDE_LANDMARKS
        }
        best = max(scores, key=scores.get)
        if self.side is None:
            self.side = best
        elif best != self.side and scores[best] - scores[self.side] > SIDE_SWITCH_MARGIN:
            self.side = best
            self.smoother.reset()
        return self.side

    @staticmethod
    def _collect_points(lms, side: str, w: int, h: int) -> Dict[str, Optional[Point]]:
        """Pixel-space coordinates so angles are not distorted by the aspect ratio."""
        pts: Dict[str, Optional[Point]] = {}
        for name, idx in SIDE_LANDMARKS[side].items():
            lm = lms[idx.value]
            inside = -0.02 <= lm.x <= 1.02 and -0.02 <= lm.y <= 1.02
            pts[name] = (lm.x * w, lm.y * h) if (lm.visibility >= VISIBILITY_MIN and inside) else None
        return pts

    def _on_tracking_lost(self) -> None:
        self.missed_frames += 1
        self.bad_frames = 0
        if self.missed_frames > LOST_RESET_FRAMES:
            self.smoother.reset()

    def _analyze(self, frame) -> FrameState:
        state = FrameState()
        h, w = frame.shape[:2]
        ex = self.exercise
        stats = self.stats[ex.key]
        dt = self._dt

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        state.pose_landmarks = results.pose_landmarks

        try:
            if results.pose_landmarks is None:
                raise LandmarkNotVisible("No person detected - step into the frame")
            lms = results.pose_landmarks.landmark

            if ex.bilateral:
                sides = {s: self._collect_points(lms, s, w, h) for s in ("LEFT", "RIGHT")}
                for s in ("LEFT", "RIGHT"):
                    if any(sides[s][n] is None for n in ex.joints):
                        raise LandmarkNotVisible("Show both arms to the camera")
                primary = "LEFT"
                raw = (calculate_angle(*(sides["LEFT"][n] for n in ex.joints)) +
                       calculate_angle(*(sides["RIGHT"][n] for n in ex.joints))) / 2.0
            else:
                side = self._select_side(lms)
                pts = self._collect_points(lms, side, w, h)
                if ex.key in ("pushup", "plank") and pts.get("ankle") is None and pts.get("knee"):
                    pts["ankle"] = pts["knee"]          # knee push-up / kneeling fallback
                missing = [n for n in ex.joints if pts[n] is None]
                if missing:
                    raise LandmarkNotVisible(f"Cannot see your {side.lower()} {', '.join(missing)}")
                sides = {side: pts}
                primary = side
                raw = calculate_angle(*(pts[n] for n in ex.joints))

            angle = self.smoother.update(raw)
            self.missed_frames = 0
            state.angle = angle
            state.sides = sides
            state.primary = primary
            state.tracking_ok = True

            # ---- reps or hold stability -----------------------------------------
            if ex.is_hold:
                state.progress = float(clamp(1.0 - max(0.0, PLANK_BODY_LINE_MIN_DEG - angle) / 30.0, 0.0, 1.0))
            else:
                if self.counter.update(angle):
                    reps = self.counter.reps
                    if reps % self.goal == 0:
                        self.voice.announce(f"Set complete. {reps} reps!")
                    else:
                        self.voice.announce(f"Rep {reps}")
                    print(f"[rep] {ex.name}: {reps}")
                state.progress = self.counter.progress(angle)

            # ---- form -------------------------------------------------------------
            posture = check_form(ex, sides, primary)
            state.posture = posture
            if posture.bad is None:
                self.bad_frames = 0
                state.posture_status = "N/A"
            else:
                self.bad_frames = self.bad_frames + 1 if posture.bad else 0
                state.posture_status = "FIX" if self.bad_frames >= POSTURE_DEBOUNCE_FRAMES else "GOOD"
            if state.posture_status == "FIX" and posture.voice:
                self.voice.alert(posture.voice)

            # ---- skeleton colour state -----------------------------------------
            transition = self.bad_frames > 0 or (not ex.is_hold and 0.12 < state.progress < 0.88)
            if state.posture_status == "FIX":
                state.skeleton = "BAD"
            elif transition:
                state.skeleton = "TRANSITION"
            else:
                state.skeleton = "GOOD"

            # ---- statistics ---------------------------------------------------------
            stats.active_s += dt
            stats.kcal += ex.met * self.weight_kg * dt / 3600.0

            if ex.is_hold and state.posture_status != "FIX":
                stats.hold_s += dt
                for m in PLANK_MILESTONES:
                    if stats.hold_s >= m and m not in self._hold_done:
                        self._hold_done.add(m)
                        self.voice.announce(f"{m} seconds held")
                        print(f"[hold] {m} seconds")
                        break

        except LandmarkNotVisible as exc:
            state.tracking_ok = False
            state.message = str(exc)
            state.skeleton = "LOST"
            self._on_tracking_lost()

        return state

    # ---- rendering ----------------------------------------------------------------
    def _arc_specs(self, state: FrameState) -> List[Tuple[Point, Point, Point]]:
        ex = self.exercise
        sides = ["LEFT", "RIGHT"] if ex.bilateral else [state.primary]
        specs: List[Tuple[Point, Point, Point]] = []
        for side in sides:
            p = state.sides.get(side) or {}
            if ex.key == "curl" and _have(p, "shoulder", "elbow", "wrist"):
                specs.append((p["shoulder"], p["elbow"], p["wrist"]))
                if _have(p, "hip"):
                    specs.append((p["hip"], p["shoulder"], p["elbow"]))
            elif ex.key == "squat" and _have(p, "hip", "knee", "ankle"):
                specs.append((p["hip"], p["knee"], p["ankle"]))
                if _have(p, "shoulder"):
                    specs.append((p["shoulder"], p["hip"], p["knee"]))
            elif ex.key == "pushup" and _have(p, "shoulder", "elbow", "wrist"):
                specs.append((p["shoulder"], p["elbow"], p["wrist"]))
                if _have(p, "hip", "ankle"):
                    specs.append((p["shoulder"], p["hip"], p["ankle"]))
            elif ex.key == "press" and _have(p, "shoulder", "elbow", "wrist"):
                specs.append((p["shoulder"], p["elbow"], p["wrist"]))
                if _have(p, "hip"):
                    specs.append((p["hip"], p["shoulder"], p["elbow"]))
            elif ex.key == "plank" and _have(p, "shoulder", "hip", "ankle"):
                specs.append((p["shoulder"], p["hip"], p["ankle"]))
        return specs

    def _render(self, frame, state: FrameState, fps: float) -> None:
        h, w = frame.shape[:2]
        accent = SKELETON_COLORS.get(state.skeleton, GREY)

        if state.pose_landmarks is not None:
            draw_skeleton(frame, state.pose_landmarks.landmark, accent, w, h)
        if state.tracking_ok and state.angle is not None:
            for p1, vertex, p2 in self._arc_specs(state):
                draw_joint_arc(frame, p1, vertex, p2, accent)

        target = state.progress if state.tracking_ok else 0.0
        self.display_pct += (target - self.display_pct) * 0.35
        self._draw_hud(frame, state, fps)

    def _draw_hud(self, frame, state: FrameState, fps: float) -> None:
        h, w = frame.shape[:2]
        ex = self.exercise
        counter = self.counter
        stats = self.stats[ex.key]
        voice_on = not self.voice.muted

        # ---- left info panel (glass, neon cyan border) -----------------------
        rounded_panel(frame, 14, 14, 344, 296, 18, PANEL, 0.62, CYAN, True)
        icon_dumbbell(frame, 36, 42, 26, CYAN, 2)
        put_text(frame, ex.name, (58, 52), 0.72, CYAN, 2)
        cv2.line(frame, (30, 68), (328, 68), DIM, 1, AA)

        if ex.is_hold:
            put_text(frame, "HOLD TIME (SS.ms)", (30, 92), 0.42, GREY, 1)
            value_color = CRIMSON if state.posture_status == "FIX" else NEON_GREEN
            put_text(frame, f"{stats.hold_s:05.2f}", (30, 160), 1.9, value_color, 4)
            stage_text = "PAUSED" if (state.posture_status == "FIX" or not state.tracking_ok) else "HOLDING"
            stage_color = CRIMSON if stage_text == "PAUSED" else NEON_GREEN
        else:
            put_text(frame, "REPS", (30, 92), 0.42, GREY, 1)
            put_text(frame, str(counter.reps), (30, 160), 2.0, WHITE, 4)
            stage_text = counter.stage or "--"
            stage_color = {"UP": NEON_GREEN, "DOWN": ORANGE}.get(stage_text, GREY)
        put_text(frame, "STAGE", (200, 92), 0.42, GREY, 1)
        put_text(frame, stage_text, (200, 140), 0.9, stage_color, 2)

        # goal (trophy / target icon)
        if ex.is_hold:
            pct_goal = min(100, int(stats.hold_s / HOLD_GOAL_S * 100))
            goal_text = f"GOAL {HOLD_GOAL_S}s  |  {pct_goal}%"
        else:
            goal_text = f"SET {counter.reps // self.goal + 1}   {counter.reps % self.goal}/{self.goal}"
        icon_target(frame, 42, 196, 24, ELECTRIC_BLUE, 2)
        put_text(frame, goal_text, (64, 203), 0.56, WHITE, 1)

        # active workout timer (stopwatch) and live calorie burn (flame)
        icon_stopwatch(frame, 42, 240, 24, ELECTRIC_BLUE, 2)
        put_text(frame, f"ACTIVE {fmt_clock(self.total_active())}", (64, 247), 0.56, WHITE, 1)
        icon_flame(frame, 226, 240, 24, ORANGE)
        put_text(frame, f"{self.total_kcal():.1f} kcal", (244, 247), 0.5, ORANGE, 1)

        cv2.line(frame, (30, 266), (328, 266), DIM, 1, AA)
        status_text, status_color = {
            "GOOD": ("GOOD", NEON_GREEN),
            "FIX": ("FIX POSTURE", CRIMSON),
        }.get(state.posture_status, ("N/A", GREY))
        if not state.tracking_ok:
            status_text, status_color = ("WAITING", YELLOW)
        put_text(frame, "FORM", (30, 286), 0.42, GREY, 1)
        put_text(frame, status_text, (100, 288), 0.62, status_color, 2)

        # ---- top-right pills: voice indicator and FPS -------------------------
        voice_color = NEON_GREEN if voice_on else CRIMSON
        rounded_panel(frame, w - 300, 14, w - 144, 50, 12, PANEL, 0.62, voice_color, False)
        icon_speaker(frame, w - 276, 32, 22, voice_color, muted=not voice_on, t=2)
        put_text(frame, "VOICE ON" if voice_on else "MUTED", (w - 254, 39), 0.5, voice_color, 1)

        fps_color = NEON_GREEN if fps >= 20 else (YELLOW if fps >= 12 else CRIMSON)
        rounded_panel(frame, w - 132, 14, w - 14, 50, 12, PANEL, 0.62, fps_color, False)
        put_text(frame, f"FPS {fps:4.1f}", (w - 122, 39), 0.55, fps_color, 2)

        # ---- vertical gradient progress bar (right edge) ----------------------
        bar_x2 = w - 30
        bar_x1 = bar_x2 - 34
        bar_y1 = 120
        bar_y2 = h - 124
        if bar_y2 - bar_y1 > 80:
            pct = clamp(self.display_pct, 0.0, 1.0)
            pct_color = progress_color(pct) if state.tracking_ok else GREY
            rounded_panel(frame, bar_x1 - 16, bar_y1 - 44, bar_x2 + 16, bar_y2 + 14, 16, PANEL, 0.62, DIM, False)
            label = f"{int(round(pct * 100))}%" if state.tracking_ok else "--"
            put_text_centered(frame, label, (bar_x1 + bar_x2) // 2, bar_y1 - 16, 0.52, pct_color, 2)

            inner_top = bar_y1 + 2
            inner_bottom = bar_y2 - 2
            inner_h = inner_bottom - inner_top
            fill_h = int(round(inner_h * pct))
            if fill_h > 0:
                grad = gradient_column(inner_h)
                frame[inner_bottom - fill_h:inner_bottom, bar_x1 + 3:bar_x2 - 2] = grad[inner_h - fill_h:]
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (230, 230, 230), 2, AA)
            for frac in (0.25, 0.5, 0.75):
                ty = int(bar_y2 - frac * (bar_y2 - bar_y1))
                cv2.line(frame, (bar_x1 - 8, ty), (bar_x1, ty), WHITE, 1, AA)

        # ---- tracking message (centre) ----------------------------------------
        if state.message:
            (tw, th), _ = cv2.getTextSize(state.message, FONT, 0.7, 2)
            cx, cy = w // 2, h // 2
            rounded_panel(frame, cx - tw // 2 - 22, cy - th - 18, cx + tw // 2 + 22, cy + 18,
                          14, PANEL, 0.78, YELLOW, False)
            put_text(frame, state.message, (cx - tw // 2, cy), 0.7, YELLOW, 2)

        # ---- blinking red form warning banner ---------------------------------
        if state.posture_status == "FIX" and state.posture is not None and state.posture.message:
            blink_on = int(time.time() * 3) % 2 == 0
            fill = CRIMSON if blink_on else (40, 20, 120)
            rounded_panel(frame, 20, h - 106, w - 20, h - 50, 16, fill, 0.9,
                          WHITE if blink_on else CRIMSON, True, 2)
            put_text_centered(frame, f"WARNING: {state.posture.message}", w // 2, h - 74,
                              0.8, WHITE, 2, max_width=w - 80)

    def _draw_dock(self, canvas, h: int) -> None:
        """Clickable control dock. Active exercise glows; hovered buttons light up."""
        active_action = f"ex:{self.exercise.key}"
        voice_on = not self.voice.muted
        for b in self.buttons:
            is_active = b.action == active_action
            hovered = b.action == self._hover
            if b.action == "voice":
                accent = NEON_GREEN if voice_on else CRIMSON
            elif b.action == "quit":
                accent = CRIMSON
            else:
                accent = (215, 215, 215)

            if is_active:
                rounded_panel(canvas, b.x1, b.y1, b.x2, b.y2, 14, ELECTRIC_BLUE, 0.30, CYAN, True, 2)
            else:
                border = accent if hovered else DIM
                rounded_panel(canvas, b.x1, b.y1, b.x2, b.y2, 14, PANEL,
                              0.85 if hovered else 0.7, border, hovered, 2)

            cx = (b.x1 + b.x2) // 2
            icon_cy = b.y1 + 30
            icon_color = CYAN if is_active else accent
            if b.icon in EXERCISE_BY_KEY:
                icon_exercise(canvas, b.icon, cx, icon_cy, 26, icon_color, 2)
            elif b.icon == "reset":
                icon_reset(canvas, cx, icon_cy, 26, icon_color, 2)
            elif b.icon == "voice":
                icon_speaker(canvas, cx, icon_cy, 26, icon_color, muted=not voice_on, t=2)
            elif b.icon == "quit":
                icon_power(canvas, cx, icon_cy, 26, icon_color, 2)

            label_color = CYAN if is_active else WHITE
            put_text_centered(canvas, b.label, cx, b.y2 - 14, 0.42, label_color, 1,
                              max_width=b.x2 - b.x1 - 10)
            put_text(canvas, b.hotkey, (b.x1 + 9, b.y1 + 18), 0.38, GREY, 1)

    def _compose(self, frame) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (h + DOCK_HEIGHT, w):
            self._canvas = np.zeros((h + DOCK_HEIGHT, w, 3), np.uint8)
            self._dock_bg = _dock_background(w, DOCK_HEIGHT)
            self._layout_dock(w, h)
        canvas = self._canvas
        canvas[:h] = frame
        canvas[h:] = self._dock_bg
        self._draw_dock(canvas, h)
        return canvas

    def _prepare_frame(self, frame):
        if self.mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if w != DISPLAY_WIDTH:
            new_h = int(round(h * DISPLAY_WIDTH / w))
            interp = cv2.INTER_AREA if w > DISPLAY_WIDTH else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (DISPLAY_WIDTH, new_h), interpolation=interp)
        return frame

    # ---- session end ------------------------------------------------------------
    def _session_rows(self) -> List[Dict]:
        rows: List[Dict] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_s = time.perf_counter() - self._session_t0
        for ex in EXERCISES:
            st = self.stats[ex.key]
            reps = self.counters[ex.key].reps
            if reps == 0 and st.active_s < 1.0 and st.hold_s < 0.5:
                continue
            rows.append({
                "date": now, "name": ex.name, "key": ex.key,
                "reps": reps, "hold_s": round(st.hold_s, 2),
                "kcal": round(st.kcal, 2), "active_s": round(st.active_s, 1),
                "session_s": round(session_s, 1),
            })
        return rows

    def _show_summary(self, image: np.ndarray, seconds: float = 3.0) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            cv2.imshow(self.WINDOW, image)
            if cv2.waitKey(30) & 0xFF != 255:     # any key skips the dashboard
                break

    def finish(self) -> None:
        """Save the session to fitness_log.csv and show the 3-second summary dashboard."""
        rows = self._session_rows()
        session_s = time.perf_counter() - self._session_t0
        total_kcal = self.total_kcal()
        total_active = self.total_active()
        saved = False
        if rows:
            try:
                append_log(rows)
                saved = True
                print(f"[log] Saved {len(rows)} row(s) to {LOG_PATH}")
            except OSError as exc:
                print(f"[log] Could not write {LOG_PATH}: {exc}")
            self.voice.announce("Workout complete. Great job!")
        summary = build_summary(rows, session_s, total_kcal, total_active, saved)
        self._show_summary(summary)

    # ---- main loop ------------------------------------------------------------------
    def run(self) -> None:
        print("\nControls:  [1] Curls  [2] Squats  [3] Push-ups  [4] Press  [5] Plank")
        print("           [R] Reset  [M] Mute  [Q] Quit   (or click the dock)\n")
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        self.voice.announce(f"{self.exercise.spoken} mode")
        self._t_prev = time.perf_counter()

        read_failures = 0
        shown_once = False
        while not self._quit:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures > 30:
                    print("[camera] The webcam stopped returning frames. Exiting.")
                    break
                time.sleep(0.01)
                continue
            read_failures = 0

            frame = self._prepare_frame(frame)
            now = time.perf_counter()
            self._dt = min(MAX_FRAME_DT, now - self._t_prev)
            self._t_prev = now

            try:
                state = self._analyze(frame)
            except Exception as exc:  # one bad frame must never crash the app
                msg = f"{type(exc).__name__}: {exc}"
                if msg != self._last_error:
                    print(f"[warn] Frame analysis failed - {msg}")
                    self._last_error = msg
                state = FrameState(message="Tracking error - adjust position")
                self._on_tracking_lost()

            self._render(frame, state, self.fps_counter.tick())
            cv2.imshow(self.WINDOW, self._compose(frame))
            shown_once = True

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                          # ESC
                self._quit = True
            elif key != 255:
                self._on_key(chr(key).lower())

            if shown_once:
                try:
                    if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        self._quit = True         # window closed with the [X] button
                except cv2.error:
                    self._quit = True

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.pose.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.voice.shutdown()
        print("Goodbye - stay strong!")


# =============================================================================
# Entry point
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Fitness Studio - offline rep counter and form coach.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--no-voice", action="store_true", help="start with voice coach disabled")
    p.add_argument("--no-mirror", action="store_true", help="do not flip the image horizontally")
    p.add_argument("--complexity", type=int, choices=(0, 1, 2), default=1,
                   help="MediaPipe model: 0 = fastest, 1 = balanced (default), 2 = most accurate")
    p.add_argument("--goal", type=int, default=DEFAULT_GOAL, help="reps per set (default 12)")
    p.add_argument("--weight", type=float, default=DEFAULT_WEIGHT_KG,
                   help="body weight in kg for calorie estimates (default 70)")
    p.add_argument("--list-cameras", action="store_true", help="list working camera indices and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cameras:
        list_cameras()
        return 0
    try:
        tracker = FitnessTracker(
            camera_index=args.camera,
            mirror=not args.no_mirror,
            voice=not args.no_voice,
            model_complexity=args.complexity,
            goal=args.goal,
            weight_kg=args.weight,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1
    try:
        tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            tracker.finish()
        except Exception as exc:  # the summary must never block a clean exit
            print(f"[summary] {exc}")
        tracker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
