#!/usr/bin/env python3
"""
Check video-analysis dependencies before running the CVS video pipeline.

The turn-aligned video workflow currently uses the MediaPipe legacy Solutions
API for Pose and Hands. That API is available in Python 3.10/3.11-compatible
MediaPipe builds, but may be absent from newer Python 3.13 installs.
"""
from __future__ import annotations

import importlib
import sys


def main() -> int:
    print(f"python {sys.version.split()[0]}")
    if sys.version_info >= (3, 12):
        print("FAIL python_version: use Python 3.10 or 3.11 for MediaPipe Solutions.")
        return 1

    try:
        import cv2
        print(f"opencv {cv2.__version__}")
    except Exception as exc:
        print(f"FAIL opencv: {exc}")
        return 1

    try:
        import mediapipe as mp
        print(f"mediapipe {getattr(mp, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"FAIL mediapipe import: {exc}")
        return 1

    try:
        pose = getattr(getattr(mp, "solutions"), "pose")
        hands = getattr(getattr(mp, "solutions"), "hands")
    except Exception:
        try:
            pose = importlib.import_module("mediapipe.python.solutions.pose")
            hands = importlib.import_module("mediapipe.python.solutions.hands")
        except Exception as exc:
            print(f"FAIL mediapipe_solutions: {exc}")
            print("Install in a Python 3.10/3.11 env, for example:")
            print("  conda env create -f environment.yml")
            print("  conda activate cvs-conversation")
            return 1

    print(f"mediapipe pose api OK: {pose.Pose}")
    print(f"mediapipe hands api OK: {hands.Hands}")
    print("video dependencies OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
