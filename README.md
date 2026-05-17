# ☀️ Solar PV Layout Optimizer

**Malaysian Rooftop PV Design Tool** — A Streamlit web app for optimising solar panel placement on rooftops, built with real Malaysian irradiance data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

- 📸 **Image Upload** — Upload rooftop satellite/drone images
- 🔄 **Orientation Adjustment** — Rotate, flip, and align images
- 📏 **Scale Setting** — Two-point calibration or manual pixels-per-meter
- 🔷 **Roof Boundary Drawing** — Click-to-draw polygon vertices
- 🚫 **Obstacle Marking** — Mark AC units, skylights, vents as exclusion zones
- ⚡ **Automated Optimization** — Grid-based panel placement with offset trials
- 📊 **Energy Yield Estimation** — 17 Malaysian locations with real PSH data
- 💰 **Financial Analysis** — RM savings at RM 0.571/kWh tariff
- 🌿 **CO₂ Savings** — Based on Malaysia grid emission factor (0.585 tCO₂/MWh)
- 📥 **Export Results** — Download optimization results as JSON

## 📋 Panel Specification

| Parameter | Value |
|-----------|-------|
| Model | Generic 620Wp |
| Dimensions | 2278 × 1134 mm |
| Efficiency | 21.3% |
| Temp Coefficient | -0.35 %/°C |

## 🏙️ Supported Locations (17 cities)

Kuala Lumpur, Petaling Jaya, Shah Alam, George Town, Johor Bahru, Ipoh, Melaka, Kuantan, Kota Bharu, Kuala Terengganu, Alor Setar, Seremban, Kota Kinabalu, Kuching, Putrajaya, Miri, Sandakan

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/solar-pv-optimizer.git
cd solar-pv-optimizer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
solar-pv-optimizer/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
├── rooftop_3d_midas.py       # Bonus: MiDaS depth estimation script
└── utils/
    ├── __init__.py
    ├── models.py             # Data classes (Point, Panel, Result, etc.)
    ├── geometry.py           # Polygon math, bounding box, inset
    ├── irradiance.py         # Malaysian PSH data, yield calculation
    └── optimization.py       # Panel placement optimization algorithm
```

## 🔧 How It Works

### Optimization Algorithm
1. **Edge Setback** — Polygon is inset by the configured setback distance
2. **Grid Placement** — Panels are placed in a regular grid within the inset boundary
3. **Offset Trials** — 6×6 = 36 grid offset positions are tested to find maximum packing
4. **Collision Detection** — Each panel is checked against boundary (point-in-polygon) and obstacles (AABB)
5. **Best Result** — The offset yielding the most panels is selected

### Performance Ratio Model
```
PR = (1 - soiling) × (1 - wiring) × inverter_eff × (1 - degradation) × (1 - temp_loss) × tilt_factor
```

| Component | Value |
|-----------|-------|
| Soiling Loss | 2% |
| Wiring Loss | 2% |
| Inverter Efficiency | 96% |
| Degradation | 0.5% |
| Temperature Derating | Based on NOCT model |
| Tilt Factor | Quadratic penalty from optimal (latitude) |

### Annual Yield
```
Yield (kWh) = Capacity (kWp) × PSH × 365 × PR
```

## 🐍 Bonus: 3D Roof Mesh Generator

The `rooftop_3d_midas.py` script uses **MiDaS depth estimation** + **Open3D** to generate a 3D mesh from rooftop photos.

```bash
pip install torch torchvision opencv-python open3d
python rooftop_3d_midas.py --image path/to/roof.jpg --model DPT_Large
```

## 📄 License

MIT License — free for personal and commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
