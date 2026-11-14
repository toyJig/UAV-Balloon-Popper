"""Shared data types for the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DetectionSource(str, Enum):
    ML = "ml"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class Detection:
    """A single candidate target in one frame.

    ``x``, ``y``, ``w``, ``h`` are in pixel coordinates (top-left origin,
    same convention as OpenCV). ``confidence`` is in [0, 1].
    """

    x: float
    y: float
    w: float
    h: float
    confidence: float
    source: DetectionSource
    class_name: str = "target"

    @property
    def center(self) -> "tuple[float, float]":
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return self.w * self.h
