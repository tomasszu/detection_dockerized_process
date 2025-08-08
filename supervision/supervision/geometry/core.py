from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
import math
from typing import Tuple


class Position(Enum):
    """
    Enum representing the position of an anchor point.
    """

    CENTER = "CENTER"
    CENTER_LEFT = "CENTER_LEFT"
    CENTER_RIGHT = "CENTER_RIGHT"
    TOP_CENTER = "TOP_CENTER"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    CENTER_OF_MASS = "CENTER_OF_MASS"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class Vector:
    start: Point
    end: Point

    @property
    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return sqrt(dx**2 + dy**2)

    @property
    def center(self) -> Point:
        """
        Calculate the center point of the vector.

        Returns:
            Point: The center point of the vector.
        """
        return Point(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def cross_product(self, point: Point) -> float:
        """
        Calculate the 2D cross product (also known as the vector product or outer
        product) of the vector and a point, treated as vectors in 2D space.

        Args:
            point (Point): The point to be evaluated, treated as the endpoint of a
                vector originating from the 'start' of the main vector.

        Returns:
            float: The scalar value of the cross product. It is positive if 'point'
                lies to the left of the vector (when moving from 'start' to 'end'),
                negative if it lies to the right, and 0 if it is collinear with the
                vector.
        """
        dx_vector = self.end.x - self.start.x
        dy_vector = self.end.y - self.start.y
        dx_point = point.x - self.start.x
        dy_point = point.y - self.start.y
        return (dx_vector * dy_point) - (dy_vector * dx_point)
    
    #MODIFICATION-----------------------------------------------------------
    #Function to determine the distance of a point from this vector    
    def distance_point_to_line(self, point : Point):
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        x0, y0 = point.x, point.y

        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        return numerator / denominator
    
    #-------------------------------------------------------------------------
    #Function to determine the closest anchor out of a set of bbox anchors passed
    def furthest_anchor_from_vector(self, box_anchors):
        max_distance = -1
        furthest_anchor = None

        for anchor in box_anchors:
            distance = self.distance_point_to_line(anchor)
            if distance > max_distance:
                max_distance = distance
                furthest_anchor = anchor

        return furthest_anchor
    #--------------------------------------------------------------------------
    #Function to determine the closest anchor ID out of a set of bbox anchors passed
    def furthest_anchor_id_from_vector(self, box_anchors):
        max_distance = -1
        furthest_anchor = None
        furthest_anchor_id = None

        for i, anchor in enumerate(box_anchors):
            distance = self.distance_point_to_line(anchor)
            if distance > max_distance:
                max_distance = distance
                furthest_anchor = anchor
                furthest_anchor_id = i

        return furthest_anchor_id
    #--------------------------------------------------------------------------


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )
