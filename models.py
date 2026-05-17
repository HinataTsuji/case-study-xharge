"""
Data models for Solar PV Layout Optimizer.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Point:
    """A 2D point in pixel coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Obstacle:
    """A rectangular obstacle on the roof."""
    x: float
    y: float
    width: float
    height: float


@dataclass
class PlacedPanel:
    """A solar panel placed on the roof."""
    x: float          # top-left x (pixels)
    y: float          # top-left y (pixels)
    width: float      # width in pixels
    height: float     # height in pixels
    rotation: float   # 0 = landscape, 90 = portrait
    row: int
    col: int


@dataclass
class PanelConfig:
    """Solar panel specifications."""
    width_mm: float = 1134.0    # mm
    height_mm: float = 2278.0   # mm
    watt_peak: float = 620.0    # Wp
    name: str = "620Wp Panel (2278×1134mm)"

    @property
    def width_m(self) -> float:
        return self.width_mm / 1000.0

    @property
    def height_m(self) -> float:
        return self.height_mm / 1000.0


@dataclass
class LocationData:
    """Malaysian location solar irradiance data."""
    name: str
    state: str
    latitude: float
    longitude: float
    psh: float               # Peak Sun Hours (kWh/m²/day)
    annual_irradiance: float  # kWh/m²/year
    avg_temp: float           # °C


@dataclass
class OptimizationResult:
    """Results from panel placement optimization."""
    panels: List[PlacedPanel]
    total_panels: int
    total_capacity_kwp: float
    annual_yield_kwh: float
    specific_yield: float       # kWh/kWp/year
    performance_ratio: float
    co2_savings_tons: float
    annual_savings_rm: float
    roof_area_m2: float
    panel_area_m2: float
    coverage_percent: float
    orientation: str            # landscape / portrait / mixed
