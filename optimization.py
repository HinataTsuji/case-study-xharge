"""
Panel placement optimization algorithm.
Grid-based placement with offset trials for maximum packing.
"""
from typing import List, Tuple, Dict, Any
from .models import PlacedPanel, Obstacle, OptimizationResult, LocationData
from .irradiance import DEFAULT_PANEL_SPEC, calculate_yield, GRID_EMISSION_FACTOR, AVG_TARIFF_RM
from .geometry import (
    rect_inside_polygon,
    rect_overlaps_rect,
    polygon_area,
    bounding_box,
    inset_polygon,
)


def optimize_layout(
    roof_points: List[Tuple[float, float]],
    obstacles: List[Dict[str, float]],
    pixels_per_meter: float,
    orientation: str,           # 'portrait' or 'landscape'
    tilt_angle: float,
    row_gap_m: float,
    col_gap_m: float,
    edge_setback_m: float,
    location: LocationData,
    panel_spec: dict | None = None,
    custom_psh: float | None = None,
) -> OptimizationResult:
    """
    Optimise solar panel placement within the roof polygon.

    Uses a grid search with multiple offset trials to maximise the number
    of panels that fit while respecting boundaries, obstacles, and gaps.

    Args:
        panel_spec: Custom panel specification dict. Uses DEFAULT_PANEL_SPEC if None.
        custom_psh: Override PSH value (kWh/m²/day). Uses location.psh if None.
    """
    spec = panel_spec or DEFAULT_PANEL_SPEC
    ppm = pixels_per_meter

    # Panel dimensions in metres
    panel_len_m = spec["length_mm"] / 1000
    panel_wid_m = spec["width_mm"] / 1000

    if orientation == "landscape":
        pw, ph = panel_len_m, panel_wid_m
    else:
        pw, ph = panel_wid_m, panel_len_m

    # Convert to pixels
    panel_w_px = pw * ppm
    panel_h_px = ph * ppm
    row_gap_px = row_gap_m * ppm
    col_gap_px = col_gap_m * ppm
    setback_px = edge_setback_m * ppm

    # Inset polygon for edge setback
    usable_poly = inset_polygon(roof_points, setback_px)
    bb = bounding_box(usable_poly)

    step_x = panel_w_px + col_gap_px
    step_y = panel_h_px + row_gap_px

    # Expand obstacles by gap so panels maintain clearance
    expanded_obs = []
    for o in obstacles:
        expanded_obs.append({
            "x": o["x"] - col_gap_px / 2,
            "y": o["y"] - row_gap_px / 2,
            "w": o["width"] + col_gap_px,
            "h": o["height"] + row_gap_px,
        })

    # Try several grid offsets for best packing
    best_panels: List[PlacedPanel] = []
    trials = 6

    for oy in range(trials):
        for ox in range(trials):
            candidate: List[PlacedPanel] = []
            sx = bb["min_x"] + (ox / trials) * step_x
            sy = bb["min_y"] + (oy / trials) * step_y

            y = sy
            row = 0
            while y + panel_h_px <= bb["max_y"] + 0.5:
                x = sx
                col = 0
                while x + panel_w_px <= bb["max_x"] + 0.5:
                    # Check inside polygon
                    if rect_inside_polygon(x, y, panel_w_px, panel_h_px, usable_poly):
                        # Check obstacles
                        blocked = False
                        for eo in expanded_obs:
                            if rect_overlaps_rect(
                                x, y, panel_w_px, panel_h_px,
                                eo["x"], eo["y"], eo["w"], eo["h"],
                            ):
                                blocked = True
                                break
                        if not blocked:
                            candidate.append(PlacedPanel(
                                x=x, y=y,
                                width=panel_w_px, height=panel_h_px,
                                rotation=0 if orientation == "landscape" else 90,
                                row=row, col=col,
                            ))
                    x += step_x
                    col += 1
                y += step_y
                row += 1

            if len(candidate) > len(best_panels):
                best_panels = candidate

    # ── Results ──────────────────────────────────────────────────────────────
    total_panels = len(best_panels)
    total_capacity_kwp = round(total_panels * spec["wattage"] / 1000, 2)

    roof_area_px = polygon_area(roof_points)
    roof_area_m2 = roof_area_px / (ppm * ppm)
    panel_area_m2 = total_panels * panel_len_m * panel_wid_m
    coverage_pct = round(panel_area_m2 / roof_area_m2 * 100, 1) if roof_area_m2 > 0 else 0.0

    yield_data = calculate_yield(
        total_capacity_kwp, location, tilt_angle,
        panel_spec=spec, custom_psh=custom_psh,
    )

    co2_savings = round(yield_data["annual_yield_kwh"] / 1000 * GRID_EMISSION_FACTOR, 2)
    annual_savings_rm = round(yield_data["annual_yield_kwh"] * AVG_TARIFF_RM)

    return OptimizationResult(
        panels=best_panels,
        total_panels=total_panels,
        total_capacity_kwp=total_capacity_kwp,
        annual_yield_kwh=yield_data["annual_yield_kwh"],
        specific_yield=yield_data["specific_yield"],
        performance_ratio=yield_data["performance_ratio"],
        co2_savings_tons=co2_savings,
        annual_savings_rm=annual_savings_rm,
        roof_area_m2=round(roof_area_m2, 1),
        panel_area_m2=round(panel_area_m2, 1),
        coverage_percent=coverage_pct,
        orientation=orientation,
    )
