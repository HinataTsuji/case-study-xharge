"""
Malaysian solar irradiance data and energy yield calculations.
Contains Peak Sun Hours (PSH) for 17 locations and performance ratio modelling.
"""
import math
from .models import LocationData

# ── Malaysian Locations with PSH Data ────────────────────────────────────────
MALAYSIAN_LOCATIONS: list[LocationData] = [
    LocationData("Kuala Lumpur",       "W.P. KL",          3.14, 101.69, 4.6, 4.6*365, 28.5),
    LocationData("Petaling Jaya",      "Selangor",          3.11, 101.65, 4.6, 4.6*365, 28.5),
    LocationData("Shah Alam",          "Selangor",          3.07, 101.52, 4.7, 4.7*365, 28.3),
    LocationData("George Town",        "Penang",            5.41, 100.34, 4.7, 4.7*365, 28.0),
    LocationData("Johor Bahru",        "Johor",             1.49, 103.74, 4.9, 4.9*365, 27.5),
    LocationData("Ipoh",               "Perak",             4.60, 101.07, 4.8, 4.8*365, 28.0),
    LocationData("Melaka",             "Melaka",            2.19, 102.25, 4.9, 4.9*365, 27.8),
    LocationData("Kuantan",            "Pahang",            3.81, 103.33, 4.5, 4.5*365, 27.5),
    LocationData("Kota Bharu",         "Kelantan",          6.12, 102.24, 4.7, 4.7*365, 27.5),
    LocationData("Kuala Terengganu",   "Terengganu",        5.31, 103.13, 4.6, 4.6*365, 27.3),
    LocationData("Alor Setar",         "Kedah",             6.12, 100.37, 4.8, 4.8*365, 28.0),
    LocationData("Seremban",           "N. Sembilan",       2.73, 101.94, 4.7, 4.7*365, 27.8),
    LocationData("Kota Kinabalu",      "Sabah",             5.98, 116.07, 5.2, 5.2*365, 27.5),
    LocationData("Kuching",            "Sarawak",           1.55, 110.35, 4.5, 4.5*365, 27.3),
    LocationData("Putrajaya",          "W.P. Putrajaya",    2.93, 101.69, 4.7, 4.7*365, 28.3),
    LocationData("Miri",               "Sarawak",           4.40, 114.01, 4.8, 4.8*365, 27.5),
    LocationData("Sandakan",           "Sabah",             5.84, 118.12, 5.0, 5.0*365, 27.3),
]

# ── Default Panel Specification ─────────────────────────────────────────────
DEFAULT_PANEL_SPEC = {
    "name":       "Generic 620Wp Panel",
    "wattage":    620,      # Wp
    "length_mm":  2278,     # mm
    "width_mm":   1134,     # mm
    "efficiency": 21.3,     # %
    "temp_coeff": -0.35,    # %/°C
}

# Keep backward-compatible alias
PANEL_SPEC = DEFAULT_PANEL_SPEC

# Malaysia grid emission factor (tCO₂/MWh)
GRID_EMISSION_FACTOR = 0.585

# Average residential tariff (RM/kWh)
AVG_TARIFF_RM = 0.571


def calculate_yield(
    total_capacity_kwp: float,
    location: LocationData,
    tilt_angle: float,
    panel_spec: dict | None = None,
    custom_psh: float | None = None,
) -> dict:
    """
    Calculate annual energy yield with Malaysian performance ratio model.

    Args:
        total_capacity_kwp: Total installed capacity in kWp.
        location: LocationData for the selected city.
        tilt_angle: Tilt angle in degrees.
        panel_spec: Custom panel specification dict. Uses DEFAULT_PANEL_SPEC if None.
        custom_psh: Override PSH value (kWh/m²/day). Uses location.psh if None.

    Returns dict with:
        annual_yield_kwh, specific_yield, performance_ratio
    """
    spec = panel_spec or DEFAULT_PANEL_SPEC
    psh = custom_psh if custom_psh is not None else location.psh

    # Performance Ratio components
    soiling_loss   = 0.02
    wiring_loss    = 0.02
    inverter_eff   = 0.96
    degradation    = 0.005

    # Temperature derating
    cell_temp = location.avg_temp + 25  # simplified NOCT estimate
    temp_loss = abs(spec["temp_coeff"]) * (cell_temp - 25) / 100

    # Tilt correction (near-equatorial, optimal ≈ latitude)
    opt_tilt = location.latitude
    tilt_penalty = 0.005 * (tilt_angle - opt_tilt) ** 2 / 100
    tilt_factor = max(0.85, 1 - tilt_penalty)

    pr = (
        (1 - soiling_loss)
        * (1 - wiring_loss)
        * inverter_eff
        * (1 - degradation)
        * (1 - temp_loss)
        * tilt_factor
    )

    annual_irradiance = psh * 365
    annual_yield_kwh = total_capacity_kwp * annual_irradiance * pr
    specific_yield = annual_irradiance * pr

    return {
        "annual_yield_kwh": round(annual_yield_kwh),
        "specific_yield":   round(specific_yield),
        "performance_ratio": round(pr, 3),
    }
