"""
Streamlit frontend for automated solar panel layout engine.
Integrates with FastAPI backend endpoints:
- POST /analyze-roof
- POST /estimate-panels
"""

import base64
import io
import json
import math
from urllib.parse import unquote_plus

import requests
import streamlit as st
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError

API_BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/analyze-roof"
ESTIMATE_ENDPOINT = f"{API_BASE_URL}/estimate-panels"


st.set_page_config(page_title="Solar Layout Frontend", page_icon="☀️", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        .stButton > button {
            background-color: #f97316;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: #ea580c;
            color: #ffffff;
        }
        .card {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "uploaded_file_token": None,
        "image_bytes": None,
        "image_name": None,
        "image_type": "image/png",
        "image_size": None,
        "image_data_uri": None,
        "roof_coords": [],
        "roof_confidence": None,
        "analyze_message": None,
        "estimate_result": None,
        "obstacles_json": "[]",
        "panel_width_m": 1.134,
        "panel_height_m": 2.278,
        "panel_wattage": 620,
        "panel_orientation": "landscape",
        "planar_meters_per_pixel": 0.25,
        "roof_pitch_deg": 30,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def to_data_uri(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def normalize_polygon(points) -> list[list[float]]:
    normalized = []
    if not isinstance(points, list):
        return normalized
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                x = float(p[0])
                y = float(p[1])
                normalized.append([x, y])
            except (TypeError, ValueError):
                continue
    return normalized


def apply_locked_boundary_from_query() -> None:
    params = st.experimental_get_query_params()
    raw = params.get("locked_roof_coords", [None])[0]
    if not raw:
        return

    try:
        decoded = unquote_plus(raw)
        parsed = json.loads(decoded)
        coords = normalize_polygon(parsed)
        if len(coords) >= 3:
            st.session_state["roof_coords"] = coords
            st.session_state["estimate_result"] = None
    except (json.JSONDecodeError, TypeError, ValueError):
        st.warning("Unable to parse locked boundary coordinates from browser payload.")
    finally:
        st.experimental_set_query_params()


apply_locked_boundary_from_query()


with st.sidebar:
    st.markdown("## ☀️ Solar Layout Engine")
    st.caption("FastAPI backend: http://localhost:8000")

    uploaded = st.file_uploader("Upload rooftop image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        token = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        if token != st.session_state["uploaded_file_token"]:
            image_bytes = uploaded.getvalue()
            mime_type = uploaded.type or "image/png"

            st.session_state["uploaded_file_token"] = token
            st.session_state["image_bytes"] = image_bytes
            st.session_state["image_name"] = uploaded.name
            st.session_state["image_type"] = mime_type
            st.session_state["image_data_uri"] = to_data_uri(image_bytes, mime_type)
            st.session_state["roof_coords"] = []
            st.session_state["roof_confidence"] = None
            st.session_state["analyze_message"] = None
            st.session_state["estimate_result"] = None

            try:
                files = {
                    "file": (uploaded.name, image_bytes, mime_type),
                }
                data = {"backend": "qwen_vl"}
                response = requests.post(ANALYZE_ENDPOINT, files=files, data=data, timeout=90)
                response.raise_for_status()
                payload = response.json()

                if payload.get("status") != "success":
                    st.session_state["analyze_message"] = "Roof analysis failed: backend returned non-success status."
                else:
                    data_obj = payload.get("data", {})
                    coords = normalize_polygon(data_obj.get("roof_polygon", []))
                    st.session_state["roof_coords"] = coords
                    st.session_state["roof_confidence"] = data_obj.get("roof_confidence")
                    st.session_state["analyze_message"] = "Roof boundary detected successfully."
            except RequestsConnectionError:
                st.session_state["analyze_message"] = (
                    "Cannot connect to FastAPI backend. Please ensure the server is running at "
                    "http://localhost:8000."
                )
            except requests.RequestException as exc:
                st.session_state["analyze_message"] = f"Roof analysis request failed: {exc}"
            except ValueError:
                st.session_state["analyze_message"] = "Roof analysis returned invalid JSON response."

    st.markdown("---")
    st.session_state["roof_pitch_deg"] = st.number_input(
        "Roof Pitch / Inclination Angle (Degrees)",
        min_value=0,
        max_value=60,
        value=int(st.session_state["roof_pitch_deg"]),
        step=5,
    )

    st.session_state["planar_meters_per_pixel"] = st.number_input(
        "Base Planar Scale (meters/pixel)",
        min_value=0.001,
        max_value=10.0,
        value=float(st.session_state["planar_meters_per_pixel"]),
        step=0.01,
        format="%.4f",
    )

    alpha_rad = math.radians(float(st.session_state["roof_pitch_deg"]))
    cos_alpha = max(math.cos(alpha_rad), 1e-6)
    compensated_mpp = float(st.session_state["planar_meters_per_pixel"]) / math.sqrt(cos_alpha)

    st.info(f"Compensated scale (S_skewed): **{compensated_mpp:.4f} m/pixel**")

    st.markdown("---")
    st.markdown("### Panel Parameters")
    st.session_state["panel_width_m"] = st.number_input(
        "Panel Width (m)",
        min_value=0.1,
        max_value=5.0,
        value=float(st.session_state["panel_width_m"]),
        step=0.01,
        format="%.3f",
    )
    st.session_state["panel_height_m"] = st.number_input(
        "Panel Height (m)",
        min_value=0.1,
        max_value=5.0,
        value=float(st.session_state["panel_height_m"]),
        step=0.01,
        format="%.3f",
    )
    st.session_state["panel_wattage"] = st.number_input(
        "Panel Wattage (Wp)",
        min_value=50,
        max_value=1500,
        value=int(st.session_state["panel_wattage"]),
        step=10,
    )
    st.session_state["panel_orientation"] = st.selectbox(
        "Panel Orientation",
        options=["landscape", "portrait"],
        index=0 if st.session_state["panel_orientation"] == "landscape" else 1,
    )

    st.markdown("---")
    st.markdown("### Obstacles (JSON)")
    st.session_state["obstacles_json"] = st.text_area(
        "Obstacle rectangles as JSON list",
        value=st.session_state["obstacles_json"],
        help='Example: [{"x": 100, "y": 120, "width": 40, "height": 60}]',
        height=120,
    )


st.markdown("# Automated Solar Layout Frontend")
st.caption("Interactive Qwen-VL boundary correction with client-side polygon editing")

if st.session_state["analyze_message"]:
    if "successfully" in st.session_state["analyze_message"].lower():
        st.success(st.session_state["analyze_message"])
    elif "cannot connect" in st.session_state["analyze_message"].lower():
        st.error(st.session_state["analyze_message"])
    else:
        st.warning(st.session_state["analyze_message"])

if st.session_state["image_bytes"] is None:
    st.info("Upload a PNG/JPG image in the sidebar to start roof detection.")
    st.stop()

img = Image.open(io.BytesIO(st.session_state["image_bytes"])).convert("RGB")
img_w, img_h = img.size

a_col, b_col = st.columns([3, 2])
with a_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Interactive Roof Polygon Adjustment")
    st.caption("Drag vertices directly on the client-side canvas, then click 'Lock Custom Boundary'.")

    initial_coords = normalize_polygon(st.session_state["roof_coords"]) or [
        [img_w * 0.2, img_h * 0.2],
        [img_w * 0.8, img_h * 0.2],
        [img_w * 0.8, img_h * 0.8],
        [img_w * 0.2, img_h * 0.8],
    ]

    html_payload = f"""
    <div style=\"background:#0b1220;border:1px solid #334155;border-radius:10px;padding:10px;\">
      <canvas id=\"roofCanvas\" style=\"width:100%;max-width:100%;border:1px solid #334155;border-radius:8px;cursor:crosshair;\"></canvas>
      <div style=\"display:flex;gap:8px;margin-top:10px;\">
        <button id=\"lockBtn\" style=\"background:#f97316;color:white;border:none;padding:8px 12px;border-radius:8px;font-weight:600;\">Lock Custom Boundary</button>
        <span style=\"color:#94a3b8;font-size:12px;align-self:center;\">No Streamlit rerun while dragging points.</span>
      </div>
    </div>
    <script>
      const dataUri = {json.dumps(st.session_state["image_data_uri"])};
      let coords = {json.dumps(initial_coords)};
      const canvas = document.getElementById('roofCanvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      const HANDLE_R = 7;
      let draggingIdx = -1;

      function resizeCanvasToImage() {{
        const maxWidth = 980;
        const ratio = img.width > maxWidth ? (maxWidth / img.width) : 1;
        canvas.width = Math.round(img.width * ratio);
        canvas.height = Math.round(img.height * ratio);
      }}

      function sx(x) {{ return x * (canvas.width / img.width); }}
      function sy(y) {{ return y * (canvas.height / img.height); }}
      function ix(x) {{ return x * (img.width / canvas.width); }}
      function iy(y) {{ return y * (img.height / canvas.height); }}

      function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        if (coords.length > 1) {{
          ctx.beginPath();
          ctx.moveTo(sx(coords[0][0]), sy(coords[0][1]));
          for (let i = 1; i < coords.length; i++) {{
            ctx.lineTo(sx(coords[i][0]), sy(coords[i][1]));
          }}
          ctx.closePath();
          ctx.fillStyle = 'rgba(249, 115, 22, 0.2)';
          ctx.fill();
          ctx.strokeStyle = '#f97316';
          ctx.lineWidth = 2;
          ctx.stroke();
        }}

        coords.forEach((p, i) => {{
          const x = sx(p[0]);
          const y = sy(p[1]);
          ctx.beginPath();
          ctx.arc(x, y, HANDLE_R, 0, Math.PI * 2);
          ctx.fillStyle = '#22d3ee';
          ctx.fill();
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 2;
          ctx.stroke();

          ctx.fillStyle = '#ffffff';
          ctx.font = '12px sans-serif';
          ctx.fillText(String(i + 1), x + 10, y - 10);
        }});
      }}

      function pointerPos(evt) {{
        const rect = canvas.getBoundingClientRect();
        return {{ x: evt.clientX - rect.left, y: evt.clientY - rect.top }};
      }}

      function pickPoint(px, py) {{
        for (let i = 0; i < coords.length; i++) {{
          const dx = sx(coords[i][0]) - px;
          const dy = sy(coords[i][1]) - py;
          if (Math.sqrt(dx * dx + dy * dy) <= HANDLE_R + 4) return i;
        }}
        return -1;
      }}

      canvas.addEventListener('mousedown', (evt) => {{
        const p = pointerPos(evt);
        draggingIdx = pickPoint(p.x, p.y);
      }});

      canvas.addEventListener('mousemove', (evt) => {{
        if (draggingIdx === -1) return;
        const p = pointerPos(evt);
        const x = Math.max(0, Math.min(img.width, ix(p.x)));
        const y = Math.max(0, Math.min(img.height, iy(p.y)));
        coords[draggingIdx] = [x, y];
        draw();
      }});

      window.addEventListener('mouseup', () => {{ draggingIdx = -1; }});
      canvas.addEventListener('mouseleave', () => {{ draggingIdx = -1; }});

      document.getElementById('lockBtn').addEventListener('click', () => {{
        const parentWindow = window.parent;
        const url = new URL(parentWindow.location.href);
        url.searchParams.set('locked_roof_coords', JSON.stringify(coords));
        parentWindow.location.href = url.toString();
      }});

      img.onload = () => {{
        resizeCanvasToImage();
        draw();
      }};
      img.src = dataUri;
    </script>
    """

    st.components.v1.html(html_payload, height=min(max(img_h + 90, 460), 980), scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

with b_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Boundary & Analysis")
    st.write(f"**Image:** {st.session_state['image_name']}")
    st.write(f"**Detected vertices:** {len(st.session_state['roof_coords'])}")
    if st.session_state["roof_confidence"] is not None:
        st.write(f"**Qwen-VL confidence:** {st.session_state['roof_confidence']:.3f}")
    st.json({"roof_polygon": st.session_state["roof_coords"]})

    if st.button("Reset to Auto-Detected Boundary"):
        try:
            files = {
                "file": (
                    st.session_state["image_name"],
                    st.session_state["image_bytes"],
                    st.session_state["image_type"],
                )
            }
            data = {"backend": "qwen_vl"}
            response = requests.post(ANALYZE_ENDPOINT, files=files, data=data, timeout=90)
            response.raise_for_status()
            payload = response.json()
            coords = normalize_polygon(payload.get("data", {}).get("roof_polygon", []))
            if len(coords) >= 3:
                st.session_state["roof_coords"] = coords
                st.session_state["roof_confidence"] = payload.get("data", {}).get("roof_confidence")
                st.success("Boundary reset from backend detection.")
            else:
                st.warning("Backend returned an invalid boundary.")
        except RequestsConnectionError:
            st.error("Cannot connect to FastAPI backend at http://localhost:8000.")
        except requests.RequestException as exc:
            st.error(f"Failed to refresh boundary: {exc}")
        except ValueError:
            st.error("Invalid JSON received from backend.")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown("## Estimate Panel Packing")

if st.button("Run /estimate-panels", type="primary"):
    coords = normalize_polygon(st.session_state["roof_coords"])
    if len(coords) < 3:
        st.error("Please provide at least 3 roof polygon points before estimation.")
    else:
        try:
            raw_obstacles = json.loads(st.session_state["obstacles_json"] or "[]")
            if not isinstance(raw_obstacles, list):
                raise ValueError("Obstacles JSON must be a list.")

            payload = {
                "roof_polygon": coords,
                "obstacles": raw_obstacles,
                "panel_spec": {
                    "width_m": float(st.session_state["panel_width_m"]),
                    "height_m": float(st.session_state["panel_height_m"]),
                    "wattage": int(st.session_state["panel_wattage"]),
                },
                "orientation": st.session_state["panel_orientation"],
                "meters_per_pixel": compensated_mpp,
            }

            response = requests.post(ESTIMATE_ENDPOINT, json=payload, timeout=90)
            response.raise_for_status()
            st.session_state["estimate_result"] = response.json()
            st.success("Panel estimation completed successfully.")
        except RequestsConnectionError:
            st.error(
                "Cannot connect to FastAPI backend. "
                "Please ensure the server is running at http://localhost:8000."
            )
        except json.JSONDecodeError as exc:
            st.error(f"Invalid obstacles JSON: {exc}")
        except requests.RequestException as exc:
            st.error(f"Estimation request failed: {exc}")
        except ValueError as exc:
            st.error(str(exc))

if st.session_state["estimate_result"] is not None:
    st.markdown("### Backend Response")
    st.json(st.session_state["estimate_result"])
