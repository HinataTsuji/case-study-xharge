"""
Solar PV Layout Optimizer — Streamlit App
Malaysian rooftop PV design tool with interactive canvas drawing.

Run:  streamlit run app.py
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import json
import math
import pandas as pd

from utils.models import LocationData, Obstacle
from utils.irradiance import MALAYSIAN_LOCATIONS, DEFAULT_PANEL_SPEC, GRID_EMISSION_FACTOR, AVG_TARIFF_RM
from utils.optimization import optimize_layout


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar PV Layout Optimizer",
    page_icon="☀️",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #f97316, #eab308);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f97316;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .step-badge {
        display: inline-block;
        background: #f97316;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        margin-right: 8px;
    }
    .tool-hint {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .psh-slider-container {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
    }
    .yield-preview {
        background: linear-gradient(135deg, #065f46, #064e3b);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
    }
    .yield-preview .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #34d399;
    }
    .yield-preview .label {
        font-size: 0.75rem;
        color: #a7f3d0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,
        "image": None,
        "rotation": 0,
        "flip_h": False,
        "flip_v": False,
        "scale_mode": "two_point",
        "pixels_per_meter": None,
        "manual_ppm": 50.0,
        "scale_points": [],
        "scale_distance": 10.0,
        "roof_points": [],
        "obstacles": [],
        "result": None,
        "drawing_mode": "polygon",   # polygon | obstacle | scale
        "canvas_key": 0,             # bump to reset canvas strokes
        # Panel spec (editable)
        "panel_wattage": DEFAULT_PANEL_SPEC["wattage"],
        "panel_length_mm": DEFAULT_PANEL_SPEC["length_mm"],
        "panel_width_mm": DEFAULT_PANEL_SPEC["width_mm"],
        "panel_efficiency": DEFAULT_PANEL_SPEC["efficiency"],
        "panel_temp_coeff": DEFAULT_PANEL_SPEC["temp_coeff"],
        # Irradiance control
        "custom_psh": None,  # None means use location default
        "use_custom_psh": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def get_panel_spec() -> dict:
    """Build panel spec dict from session state."""
    return {
        "name": f"Custom {st.session_state['panel_wattage']}Wp Panel",
        "wattage": st.session_state["panel_wattage"],
        "length_mm": st.session_state["panel_length_mm"],
        "width_mm": st.session_state["panel_width_mm"],
        "efficiency": st.session_state["panel_efficiency"],
        "temp_coeff": st.session_state["panel_temp_coeff"],
    }


# ── Helper Functions ─────────────────────────────────────────────────────────
def get_adjusted_image() -> Image.Image:
    """Apply rotation and flip to the uploaded image."""
    img = st.session_state["image"].copy()
    if st.session_state["rotation"] != 0:
        img = img.rotate(-st.session_state["rotation"], expand=True)
    if st.session_state["flip_h"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if st.session_state["flip_v"]:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def bump_canvas():
    """Reset the canvas by bumping its key."""
    st.session_state["canvas_key"] += 1


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="main-header">☀️ Solar PV Optimizer</p>', unsafe_allow_html=True)
    st.caption("Malaysian Rooftop PV Design Tool")
    st.divider()

    # ── Step 1: Upload ───────────────────────────────────────────────────────
    st.markdown('<span class="step-badge">1</span> **Upload Roof Image**', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a rooftop image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if uploaded is not None and st.session_state["image"] is None:
        img = Image.open(uploaded).convert("RGB")
        # Resize large images for performance
        max_dim = 1200
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        st.session_state["image"] = img
        st.session_state["step"] = 2
        st.rerun()

    st.divider()

    # ── Step 2: Orientation ──────────────────────────────────────────────────
    st.markdown('<span class="step-badge">2</span> **Adjust Orientation**', unsafe_allow_html=True)
    if st.session_state["image"] is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↻ 90° CW"):
                st.session_state["rotation"] = (st.session_state["rotation"] + 90) % 360
                st.rerun()
            if st.button("🔄 Flip H"):
                st.session_state["flip_h"] = not st.session_state["flip_h"]
                st.rerun()
        with col2:
            if st.button("↺ 90° CCW"):
                st.session_state["rotation"] = (st.session_state["rotation"] - 90) % 360
                st.rerun()
            if st.button("🔃 Flip V"):
                st.session_state["flip_v"] = not st.session_state["flip_v"]
                st.rerun()

        st.session_state["rotation"] = st.slider(
            "Fine rotation (°)", -180, 180,
            st.session_state["rotation"], step=1
        )

        if st.button("✅ Confirm Orientation", type="primary"):
            # Bake the transform
            st.session_state["image"] = get_adjusted_image()
            st.session_state["rotation"] = 0
            st.session_state["flip_h"] = False
            st.session_state["flip_v"] = False
            st.session_state["step"] = 3
            st.rerun()
    else:
        st.info("Upload an image first")

    st.divider()

    # ── Step 3: Scale ────────────────────────────────────────────────────────
    st.markdown('<span class="step-badge">3</span> **Set Scale**', unsafe_allow_html=True)
    scale_mode = st.radio(
        "Scale method",
        ["Two-Point", "Manual Ratio"],
        horizontal=True,
        index=0 if st.session_state["scale_mode"] == "two_point" else 1,
        label_visibility="collapsed",
    )
    st.session_state["scale_mode"] = "two_point" if scale_mode == "Two-Point" else "manual"

    if st.session_state["scale_mode"] == "manual":
        st.session_state["manual_ppm"] = st.number_input(
            "Pixels per meter",
            min_value=1.0,
            max_value=1000.0,
            value=st.session_state["manual_ppm"],
            step=1.0,
        )
        if st.button("✅ Apply Manual Scale", type="primary"):
            st.session_state["pixels_per_meter"] = st.session_state["manual_ppm"]
            st.session_state["step"] = max(st.session_state["step"], 4)
            st.rerun()
        if st.session_state["pixels_per_meter"]:
            st.success(f"Scale: {st.session_state['pixels_per_meter']:.1f} px/m")
    else:
        st.session_state["scale_distance"] = st.number_input(
            "Known distance (m)",
            min_value=0.1,
            value=st.session_state["scale_distance"],
            step=0.5,
        )
        if len(st.session_state["scale_points"]) >= 2:
            p1, p2 = st.session_state["scale_points"][:2]
            pixel_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            ppm = pixel_dist / st.session_state["scale_distance"]
            st.session_state["pixels_per_meter"] = ppm
            st.success(f"Scale: {ppm:.1f} px/m ({pixel_dist:.0f}px = {st.session_state['scale_distance']}m)")
        else:
            st.info("Select **🔵 Scale Points** tool, then click 2 points on the image")

    st.divider()

    # ── Step 4: Drawing Mode ─────────────────────────────────────────────────
    st.markdown('<span class="step-badge">4</span> **Draw on Image**', unsafe_allow_html=True)
    drawing_mode = st.radio(
        "Drawing tool",
        ["🔵 Scale Points", "🟢 Roof Boundary", "🔴 Obstacle"],
        horizontal=False,
    )
    if "Scale" in drawing_mode:
        st.session_state["drawing_mode"] = "scale"
    elif "Roof" in drawing_mode:
        st.session_state["drawing_mode"] = "polygon"
    else:
        st.session_state["drawing_mode"] = "obstacle"

    st.caption(f"Boundary points: {len(st.session_state['roof_points'])}")
    st.caption(f"Obstacles: {len(st.session_state['obstacles'])}")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑 Clear Boundary"):
            st.session_state["roof_points"] = []
            st.session_state["result"] = None
            bump_canvas()
            st.rerun()
    with col_c2:
        if st.button("🗑 Clear Obstacles"):
            st.session_state["obstacles"] = []
            st.session_state["result"] = None
            bump_canvas()
            st.rerun()

    if st.button("↩ Undo Last Point"):
        if st.session_state["drawing_mode"] == "polygon" and st.session_state["roof_points"]:
            st.session_state["roof_points"].pop()
            bump_canvas()
            st.rerun()
        elif st.session_state["drawing_mode"] == "scale" and st.session_state["scale_points"]:
            st.session_state["scale_points"].pop()
            bump_canvas()
            st.rerun()
        elif st.session_state["drawing_mode"] == "obstacle" and st.session_state["obstacles"]:
            st.session_state["obstacles"].pop()
            bump_canvas()
            st.rerun()

    st.divider()

    # ── Step 5: Panel Specification (EDITABLE) ──────────────────────────────
    st.markdown('<span class="step-badge">5</span> **Panel Specification**', unsafe_allow_html=True)

    with st.expander("🔧 Edit Panel Specs", expanded=True):
        st.session_state["panel_wattage"] = st.number_input(
            "Wattage (Wp)",
            min_value=100,
            max_value=1000,
            value=st.session_state["panel_wattage"],
            step=5,
            help="Rated power output of one panel in Watt-peak",
        )

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.session_state["panel_length_mm"] = st.number_input(
                "Length (mm)",
                min_value=500,
                max_value=3000,
                value=st.session_state["panel_length_mm"],
                step=10,
                help="Panel length in millimeters",
            )
        with pcol2:
            st.session_state["panel_width_mm"] = st.number_input(
                "Width (mm)",
                min_value=500,
                max_value=2000,
                value=st.session_state["panel_width_mm"],
                step=10,
                help="Panel width in millimeters",
            )

        pcol3, pcol4 = st.columns(2)
        with pcol3:
            st.session_state["panel_efficiency"] = st.number_input(
                "Efficiency (%)",
                min_value=10.0,
                max_value=30.0,
                value=st.session_state["panel_efficiency"],
                step=0.1,
                format="%.1f",
                help="Panel conversion efficiency",
            )
        with pcol4:
            st.session_state["panel_temp_coeff"] = st.number_input(
                "Temp Coeff (%/°C)",
                min_value=-0.60,
                max_value=-0.10,
                value=st.session_state["panel_temp_coeff"],
                step=0.01,
                format="%.2f",
                help="Power temperature coefficient (negative value)",
            )

        # Show panel area
        area_m2 = (st.session_state["panel_length_mm"] / 1000) * (st.session_state["panel_width_mm"] / 1000)
        st.caption(f"📐 Panel area: {area_m2:.2f} m² | "
                   f"{st.session_state['panel_length_mm']} × {st.session_state['panel_width_mm']} mm")

        if st.button("🔄 Reset to Default"):
            st.session_state["panel_wattage"] = DEFAULT_PANEL_SPEC["wattage"]
            st.session_state["panel_length_mm"] = DEFAULT_PANEL_SPEC["length_mm"]
            st.session_state["panel_width_mm"] = DEFAULT_PANEL_SPEC["width_mm"]
            st.session_state["panel_efficiency"] = DEFAULT_PANEL_SPEC["efficiency"]
            st.session_state["panel_temp_coeff"] = DEFAULT_PANEL_SPEC["temp_coeff"]
            st.rerun()

    st.divider()

    # ── Step 6: Configuration & Irradiance ───────────────────────────────────
    st.markdown('<span class="step-badge">6</span> **Configuration**', unsafe_allow_html=True)

    location_names = [f"{loc.name} ({loc.state})" for loc in MALAYSIAN_LOCATIONS]
    loc_idx = st.selectbox("Location", range(len(location_names)), format_func=lambda i: location_names[i])
    selected_location = MALAYSIAN_LOCATIONS[loc_idx]

    # ── PSH / Irradiance Slider ──────────────────────────────────────────────
    st.markdown('<div class="psh-slider-container">', unsafe_allow_html=True)
    st.markdown("**☀️ Solar Irradiance (PSH)**")
    st.caption(f"📍 {selected_location.name} default: **{selected_location.psh}** kWh/m²/day")

    use_custom = st.checkbox(
        "Override with custom PSH",
        value=st.session_state["use_custom_psh"],
        key="use_custom_psh_cb",
    )
    st.session_state["use_custom_psh"] = use_custom

    if use_custom:
        custom_psh_val = st.slider(
            "Peak Sun Hours (kWh/m²/day)",
            min_value=3.0,
            max_value=7.0,
            value=st.session_state["custom_psh"] or selected_location.psh,
            step=0.1,
            format="%.1f",
            help="Adjust to see how irradiance affects energy yield",
        )
        st.session_state["custom_psh"] = custom_psh_val
        active_psh = custom_psh_val
    else:
        st.session_state["custom_psh"] = None
        active_psh = selected_location.psh

    # Live yield preview based on PSH
    annual_irr = active_psh * 365
    st.markdown(f"""
    <div class="yield-preview">
        <div class="value">{active_psh:.1f} kWh/m²/day</div>
        <div class="label">Annual Irradiance: {annual_irr:,.0f} kWh/m²/yr</div>
    </div>
    """, unsafe_allow_html=True)

    # If we have results, show live yield estimate
    if st.session_state["result"]:
        spec = get_panel_spec()
        capacity = st.session_state["result"].total_capacity_kwp
        from utils.irradiance import calculate_yield as calc_yield
        live_yield = calc_yield(capacity, selected_location, tilt_angle=5.0,
                                panel_spec=spec, custom_psh=active_psh)
        st.markdown(f"""
        <div class="yield-preview">
            <div class="value">{live_yield['annual_yield_kwh']:,.0f} kWh/yr</div>
            <div class="label">Est. yield for {capacity} kWp @ PSH {active_psh}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    orientation = st.radio("Panel Orientation", ["Landscape", "Portrait"], horizontal=True)
    tilt_angle = st.slider("Tilt Angle (°)", 0.0, 30.0, 5.0, step=0.5)
    row_gap = st.number_input("Row Gap (m)", 0.0, 2.0, 0.02, step=0.01)
    col_gap = st.number_input("Column Gap (m)", 0.0, 2.0, 0.02, step=0.01)
    edge_setback = st.number_input("Edge Setback (m)", 0.0, 3.0, 0.5, step=0.1)

    st.divider()

    # ── Step 7: Optimize ─────────────────────────────────────────────────────
    st.markdown('<span class="step-badge">7</span> **Optimize**', unsafe_allow_html=True)

    can_optimize = (
        st.session_state["image"] is not None
        and len(st.session_state["roof_points"]) >= 3
        and st.session_state["pixels_per_meter"] is not None
        and st.session_state["pixels_per_meter"] > 0
    )

    if st.button("⚡ Run Optimization", type="primary", disabled=not can_optimize):
        with st.spinner("Optimizing panel layout..."):
            panel_spec = get_panel_spec()
            custom_psh = st.session_state["custom_psh"] if st.session_state["use_custom_psh"] else None

            result = optimize_layout(
                roof_points=st.session_state["roof_points"],
                obstacles=[
                    {"x": o["x"], "y": o["y"], "width": o["width"], "height": o["height"]}
                    for o in st.session_state["obstacles"]
                ],
                pixels_per_meter=st.session_state["pixels_per_meter"],
                orientation=orientation.lower(),
                tilt_angle=tilt_angle,
                row_gap_m=row_gap,
                col_gap_m=col_gap,
                edge_setback_m=edge_setback,
                location=selected_location,
                panel_spec=panel_spec,
                custom_psh=custom_psh,
            )
            st.session_state["result"] = result
            st.rerun()

    if not can_optimize:
        missing = []
        if st.session_state["image"] is None:
            missing.append("Upload image")
        if len(st.session_state["roof_points"]) < 3:
            missing.append(f"Draw boundary ({len(st.session_state['roof_points'])}/3+ points)")
        if not st.session_state["pixels_per_meter"]:
            missing.append("Set scale")
        st.warning("Missing: " + ", ".join(missing))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CANVAS — Interactive Click-to-Draw
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["image"] is None:
    st.markdown("## ☀️ Solar PV Layout Optimizer")
    st.markdown("### Malaysian Rooftop PV Design Tool")
    st.info("👈 Upload a rooftop image in the sidebar to get started!")

    spec = get_panel_spec()
    st.markdown(f"""
    #### Features
    - 📸 **Image Upload** with rotation & flip adjustment
    - 📏 **Scale Setting** — two-point or manual pixels/meter
    - 🔷 **Roof Boundary** — click on image to draw polygon vertices
    - 🚫 **Obstacle Marking** — drag rectangles on obstacles
    - ⚡ **Auto Optimization** — maximise panel placement
    - 📊 **Yield Estimation** — 17 Malaysian locations with real irradiance data
    - ☀️ **PSH Slider** — adjust irradiance to see yield impact in real-time
    - 🔧 **Editable Panel Specs** — customize wattage, dimensions & more
    - 💰 **Financial & CO₂** savings calculations

    #### Current Panel Specification
    | Spec | Value |
    |------|-------|
    | Wattage | {spec['wattage']} Wp |
    | Dimensions | {spec['length_mm']} × {spec['width_mm']} mm |
    | Efficiency | {spec['efficiency']}% |
    | Temp Coefficient | {spec['temp_coeff']} %/°C |

    *Edit panel specs in the sidebar under Step 5* ⬅️
    """)
else:
    # Prepare the display image
    display_img = get_adjusted_image() if st.session_state["step"] <= 2 else st.session_state["image"]

    # ── Create annotated background with existing points ────────────────────
    annotated = display_img.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    # Draw roof boundary polygon
    roof_pts = st.session_state["roof_points"]
    if len(roof_pts) >= 2:
        for i in range(len(roof_pts)):
            p1 = roof_pts[i]
            p2 = roof_pts[(i + 1) % len(roof_pts)] if i < len(roof_pts) - 1 else roof_pts[0]
            if i < len(roof_pts) - 1 or len(roof_pts) >= 3:
                draw.line([p1[0], p1[1], p2[0], p2[1]], fill=(59, 130, 246, 200), width=3)

    if len(roof_pts) >= 3:
        poly_tuples = [(p[0], p[1]) for p in roof_pts]
        draw.polygon(poly_tuples, fill=(59, 130, 246, 30))

    for i, pt in enumerate(roof_pts):
        r = 6
        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=(59, 130, 246, 255))
        draw.text((pt[0]+10, pt[1]-10), str(i+1), fill=(255, 255, 255, 255))

    # Draw obstacles
    for obs in st.session_state["obstacles"]:
        draw.rectangle(
            [obs["x"], obs["y"], obs["x"] + obs["width"], obs["y"] + obs["height"]],
            fill=(239, 68, 68, 60),
            outline=(239, 68, 68, 200),
            width=2,
        )

    # Draw scale points
    for pt in st.session_state["scale_points"]:
        r = 7
        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=(234, 179, 8, 255))

    if len(st.session_state["scale_points"]) >= 2:
        p1, p2 = st.session_state["scale_points"][:2]
        draw.line([p1[0], p1[1], p2[0], p2[1]], fill=(234, 179, 8, 200), width=3)

    # Draw optimized panels
    if st.session_state["result"]:
        for panel in st.session_state["result"].panels:
            draw.rectangle(
                [panel.x, panel.y, panel.x + panel.width, panel.y + panel.height],
                fill=(34, 197, 94, 80),
                outline=(34, 197, 94, 220),
                width=2,
            )

    # ── Active tool hint ────────────────────────────────────────────────────
    mode = st.session_state["drawing_mode"]
    mode_config = {
        "scale": {
            "label": "📏 Scale Points — Click 2 points on a known distance",
            "canvas_mode": "point",
            "stroke": "#eab308",
            "point_radius": 6,
        },
        "polygon": {
            "label": "🟢 Roof Boundary — Click to place polygon vertices",
            "canvas_mode": "point",
            "stroke": "#3b82f6",
            "point_radius": 6,
        },
        "obstacle": {
            "label": "🔴 Obstacles — Click and drag to draw rectangles",
            "canvas_mode": "rect",
            "stroke": "#ef4444",
            "point_radius": 3,
        },
    }
    cfg = mode_config[mode]

    st.markdown(f'<div class="tool-hint">🎯 <strong>Active Tool:</strong> {cfg["label"]}</div>', unsafe_allow_html=True)

    # ── Interactive Canvas ──────────────────────────────────────────────────
    # Calculate canvas display width (fit to column, max 900px)
    img_w, img_h = display_img.size
    canvas_display_width = min(img_w, 900)
    canvas_display_height = int(img_h * (canvas_display_width / img_w))

    # Resize annotated image to canvas display size for background
    bg_image = annotated.resize((canvas_display_width, canvas_display_height), Image.LANCZOS)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.15)",
        stroke_width=3 if mode == "obstacle" else 1,
        stroke_color=cfg["stroke"],
        background_image=bg_image,
        drawing_mode=cfg["canvas_mode"],
        point_display_radius=cfg["point_radius"],
        width=canvas_display_width,
        height=canvas_display_height,
        key=f"canvas_{st.session_state['canvas_key']}",
    )

    # ── Process canvas clicks ───────────────────────────────────────────────
    if canvas_result is not None and canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        scale_factor = img_w / canvas_display_width

        if objects:
            if mode == "polygon":
                new_points = []
                for obj in objects:
                    if obj.get("type") == "circle":
                        cx = (obj.get("left", 0) + obj.get("radius", 0)) * scale_factor
                        cy = (obj.get("top", 0) + obj.get("radius", 0)) * scale_factor
                        new_points.append((int(cx), int(cy)))
                if new_points:
                    st.session_state["roof_points"].extend(new_points)
                    st.session_state["result"] = None
                    bump_canvas()
                    st.rerun()

            elif mode == "scale":
                new_points = []
                for obj in objects:
                    if obj.get("type") == "circle":
                        cx = (obj.get("left", 0) + obj.get("radius", 0)) * scale_factor
                        cy = (obj.get("top", 0) + obj.get("radius", 0)) * scale_factor
                        new_points.append((int(cx), int(cy)))
                remaining = 2 - len(st.session_state["scale_points"])
                if new_points and remaining > 0:
                    st.session_state["scale_points"].extend(new_points[:remaining])
                    bump_canvas()
                    st.rerun()

            elif mode == "obstacle":
                new_obstacles = []
                for obj in objects:
                    if obj.get("type") == "rect":
                        ox = int(obj.get("left", 0) * scale_factor)
                        oy = int(obj.get("top", 0) * scale_factor)
                        ow = int(obj.get("width", 0) * obj.get("scaleX", 1) * scale_factor)
                        oh = int(obj.get("height", 0) * obj.get("scaleY", 1) * scale_factor)
                        if ow > 5 and oh > 5:
                            new_obstacles.append({"x": ox, "y": oy, "width": ow, "height": oh})
                if new_obstacles:
                    st.session_state["obstacles"].extend(new_obstacles)
                    st.session_state["result"] = None
                    bump_canvas()
                    st.rerun()

    # ── Manual coordinate fallback (collapsed) ──────────────────────────────
    with st.expander("⌨️ Manual Coordinate Entry (optional)"):
        st.caption("Use this if clicking on the canvas isn't working for you.")
        col_x, col_y, col_btn = st.columns([2, 2, 1])
        with col_x:
            click_x = st.number_input("X", min_value=0, max_value=display_img.width, value=0, key="click_x")
        with col_y:
            click_y = st.number_input("Y", min_value=0, max_value=display_img.height, value=0, key="click_y")
        with col_btn:
            st.write("")
            st.write("")
            add_point = st.button("➕ Add")

        if add_point and (click_x > 0 or click_y > 0):
            if mode == "polygon":
                st.session_state["roof_points"].append((click_x, click_y))
                st.session_state["result"] = None
                bump_canvas()
                st.rerun()
            elif mode == "scale":
                if len(st.session_state["scale_points"]) < 2:
                    st.session_state["scale_points"].append((click_x, click_y))
                    bump_canvas()
                    st.rerun()

        if mode == "obstacle":
            st.markdown("**Add Obstacle (rectangle):**")
            oc1, oc2, oc3, oc4, oc5 = st.columns([1, 1, 1, 1, 1])
            with oc1:
                obs_x = st.number_input("Obs X", 0, display_img.width, 0, key="obs_x")
            with oc2:
                obs_y = st.number_input("Obs Y", 0, display_img.height, 0, key="obs_y")
            with oc3:
                obs_w = st.number_input("Width", 1, display_img.width, 50, key="obs_w")
            with oc4:
                obs_h = st.number_input("Height", 1, display_img.height, 50, key="obs_h")
            with oc5:
                st.write("")
                st.write("")
                if st.button("➕ Add Obstacle"):
                    st.session_state["obstacles"].append({
                        "x": obs_x, "y": obs_y,
                        "width": obs_w, "height": obs_h,
                    })
                    st.session_state["result"] = None
                    bump_canvas()
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["result"]:
    r = st.session_state["result"]
    loc = MALAYSIAN_LOCATIONS[loc_idx]
    spec = get_panel_spec()
    active_psh = st.session_state["custom_psh"] if st.session_state["use_custom_psh"] else loc.psh

    st.markdown("---")
    st.markdown("## 📊 Optimization Results")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔢 Total Panels", f"{r.total_panels}")
    with c2:
        st.metric("⚡ Capacity", f"{r.total_capacity_kwp} kWp")
    with c3:
        st.metric("🌤 Annual Yield", f"{r.annual_yield_kwh:,.0f} kWh")
    with c4:
        st.metric("📈 Specific Yield", f"{r.specific_yield:,} kWh/kWp")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("🏠 Roof Area", f"{r.roof_area_m2} m²")
    with c6:
        st.metric("📐 Panel Area", f"{r.panel_area_m2} m²")
    with c7:
        st.metric("🌿 CO₂ Saved", f"{r.co2_savings_tons} t/yr")
    with c8:
        st.metric("💰 Savings", f"RM {r.annual_savings_rm:,}/yr")

    # Detailed table
    st.markdown("### 📋 Detailed Results")
    details = {
        "Parameter": [
            "Location", "PSH Used", "Panel Orientation", "Performance Ratio",
            "Roof Utilization", "Grid Emission Factor", "Electricity Tariff",
            "Panel Model", "Panel Dimensions", "Panel Wattage",
            "Panel Efficiency", "Temp Coefficient",
        ],
        "Value": [
            f"{loc.name}, {loc.state} (Default PSH: {loc.psh} kWh/m²/day)",
            f"{active_psh:.1f} kWh/m²/day" + (" (custom)" if st.session_state["use_custom_psh"] else " (location default)"),
            r.orientation.capitalize(),
            f"{r.performance_ratio:.1%}",
            f"{r.coverage_percent}%",
            f"{GRID_EMISSION_FACTOR} tCO₂/MWh",
            f"RM {AVG_TARIFF_RM}/kWh",
            spec["name"],
            f"{spec['length_mm']} × {spec['width_mm']} mm",
            f"{spec['wattage']} Wp",
            f"{spec['efficiency']}%",
            f"{spec['temp_coeff']} %/°C",
        ],
    }
    st.table(details)

    # Export results
    st.markdown("### 📥 Export")
    export_data = {
        "location": loc.name,
        "state": loc.state,
        "psh_used": active_psh,
        "total_panels": r.total_panels,
        "capacity_kwp": r.total_capacity_kwp,
        "annual_yield_kwh": r.annual_yield_kwh,
        "specific_yield": r.specific_yield,
        "performance_ratio": r.performance_ratio,
        "roof_area_m2": r.roof_area_m2,
        "panel_area_m2": r.panel_area_m2,
        "coverage_percent": r.coverage_percent,
        "co2_savings_tons": r.co2_savings_tons,
        "annual_savings_rm": r.annual_savings_rm,
        "orientation": r.orientation,
        "panel_spec": spec,
    }
    st.download_button(
        "📥 Download Results (JSON)",
        json.dumps(export_data, indent=2),
        file_name="solar_pv_results.json",
        mime="application/json",
    )
