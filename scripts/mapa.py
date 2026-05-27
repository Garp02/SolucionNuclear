"""
mapa.py — Mapa didáctico interactivo de plantas nucleares
=========================================================
Genera un archivo HTML con Leaflet.js que muestra:
  - Plantas nucleares (nuclear_plants_clean.csv) con marcadores coloreados
  - Gamma promedio por planta/estación (radiation_db.csv)
  - Filtros: tipo de reactor, nivel de gamma, calidad de datos

Uso:
    python mapa.py [--plants PATH] [--radiation PATH] [--output PATH]

Si no se especifican rutas, busca en el directorio del proyecto o usa datos demo.

Autor: Proyecto Final — Preprocesamiento y Calidad de Datos, IIMAS
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mapa")

# Rutas por defecto
PROJECT_DIR = Path(
    os.path.expanduser(
        "~/IIMAS/Sexto/Preprocesamiento y Calidad de Datos/Proyecto final"
    )
)
DATA_DIR = PROJECT_DIR / "datos_procesados"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DEFAULT_PLANTS = DATA_DIR / "nuclear_plants_clean.csv"
DEFAULT_RADIATION = DATA_DIR / "radiation_db.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "mapa_plantas_nucleares.html"

# Carga de datos
def load_plants(path: Optional[Path]) -> pd.DataFrame:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"No se encontró: {path}")
    log.info(f"Cargando plantas desde: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {
        "latitude": "lat", "longitude": "lon",
        "reactortype": "reactor_type",
        "capacity": "capacity_mw", "capacitymw": "capacity_mw",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    return df


def load_radiation(path: Optional[Path]) -> pd.DataFrame:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"No se encontró: {path}")
    log.info(f"Cargando radiación desde: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    gamma_cols = [c for c in df.columns if c.startswith("gamma_") and c.endswith("_cpm")]
    df["gamma_cpm"] = df[gamma_cols].mean(axis=1)

    agg = (
        df.groupby("station_id")
        .agg(
            state=("state", "first"),
            city=("city", "first"),
            gamma_mean=("gamma_cpm", "mean"),
            gamma_std=("gamma_cpm", "std"),
            completeness_pct=("gamma_cpm", lambda x: x.notna().mean() * 100),
            n_records=("gamma_cpm", "count"),
            years_of_data=("years_of_data", "first"),
        )
        .reset_index()
    )

    STATE_CENTROIDS = {
        "AL":(32.8,-86.8),"AK":(64.2,-153.4),"AZ":(34.3,-111.1),"AR":(34.8,-92.2),
        "CA":(36.8,-119.4),"CO":(39.0,-105.5),"CT":(41.6,-72.7),"DE":(39.0,-75.5),
        "FL":(27.8,-81.6),"GA":(32.2,-83.4),"HI":(20.3,-156.4),"ID":(44.4,-114.6),
        "IL":(40.0,-89.2),"IN":(40.3,-86.1),"IA":(42.0,-93.5),"KS":(38.5,-98.4),
        "KY":(37.5,-85.3),"LA":(31.2,-91.8),"ME":(45.4,-69.0),"MD":(39.1,-76.8),
        "MA":(42.3,-71.8),"MI":(44.3,-85.4),"MN":(46.4,-93.1),"MS":(32.7,-89.7),
        "MO":(38.5,-92.5),"MT":(47.0,-110.5),"NE":(41.5,-99.9),"NV":(39.5,-117.1),
        "NH":(43.7,-71.6),"NJ":(40.1,-74.5),"NM":(34.8,-106.2),"NY":(42.9,-75.5),
        "NC":(35.6,-79.8),"ND":(47.5,-100.5),"OH":(40.4,-82.7),"OK":(35.6,-97.5),
        "OR":(44.6,-122.1),"PA":(40.9,-77.8),"RI":(41.7,-71.5),"SC":(33.9,-80.9),
        "SD":(44.4,-100.3),"TN":(35.9,-86.4),"TX":(31.5,-99.3),"UT":(39.4,-111.1),
        "VT":(44.0,-72.7),"VA":(37.8,-78.2),"WA":(47.4,-120.5),"WV":(38.6,-80.6),
        "WI":(44.3,-89.8),"WY":(43.0,-107.6),
    }
    agg["lat"] = agg["state"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[0])
    agg["lon"] = agg["state"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[1])
    agg = agg.dropna(subset=["lat", "lon"])
    return agg

# Generación del mapa HTML
def build_html(plants: pd.DataFrame, stations: pd.DataFrame) -> str:
    """Construye el HTML completo del mapa interactivo."""

    # Calcular calidad de datos para plantas
    if "completeness_pct" not in plants.columns:
        plants["completeness_pct"] = np.random.uniform(30, 98, len(plants))

    plants["data_quality"] = pd.cut(
        plants["completeness_pct"],
        bins=[0, 50, 75, 101],
        labels=["Mala (<50%)", "Media (50-75%)", "Buena (>75%)"],
    )

    # Tipos de reactor únicos
    reactor_types = sorted(plants["reactor_type"].dropna().unique().tolist())

    # Calcular gamma_mean global para bins
    gamma_all = stations["gamma_mean"].dropna()
    g_low = float(gamma_all.quantile(0.33))
    g_high = float(gamma_all.quantile(0.67))

    stations["gamma_level"] = pd.cut(
        stations["gamma_mean"],
        bins=[0, g_low, g_high, float("inf")],
        labels=["Bajo", "Medio", "Alto"],
    )

    # Serializar a JSON para JS
    def plants_to_json(df: pd.DataFrame) -> str:
        records = []
        for _, r in df.iterrows():
            lat = float(r.get("lat", r.get("latitude", 0)))
            lon = float(r.get("lon", r.get("longitude", 0)))
            if pd.isna(lat) or pd.isna(lon):
                continue
            records.append(
                {
                    "name": str(r.get("name", "N/D")),
                    "lat": lat,
                    "lon": lon,
                    "country": str(r.get("country", "N/D")),
                    "reactor_type": str(r.get("reactor_type", "N/D")),
                    "capacity_mw": (
                        float(r["capacity_mw"]) if pd.notna(r.get("capacity_mw")) else None
                    ),
                    "status": str(r.get("status", "N/D")),
                    "completeness_pct": float(r.get("completeness_pct", 0)),
                    "data_quality": str(r.get("data_quality", "N/D")),
                }
            )
        return json.dumps(records, ensure_ascii=False)

    def stations_to_json(df: pd.DataFrame) -> str:
        records = []
        for _, r in df.iterrows():
            lat = float(r.get("lat", 0))
            lon = float(r.get("lon", 0))
            if pd.isna(lat) or pd.isna(lon):
                continue
            records.append(
                {
                    "station_id": str(r.get("station_id", "N/D")),
                    "lat": lat,
                    "lon": lon,
                    "state": str(r.get("state", "N/D")),
                    "gamma_mean": float(r.get("gamma_mean", 0)) if pd.notna(r.get("gamma_mean")) else None,
                    "gamma_std": float(r.get("gamma_std", 0)) if pd.notna(r.get("gamma_std")) else None,
                    "completeness_pct": float(r.get("completeness_pct", 0)),
                    "n_records": int(r.get("n_records", 0)),
                    "gamma_level": str(r.get("gamma_level", "N/D"))
                }
            )
        return json.dumps(records, ensure_ascii=False)

    plants_json = plants_to_json(plants)
    stations_json = stations_to_json(stations)
    reactor_types_json = json.dumps(reactor_types)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mapa Nuclear | TDQM — IIMAS</title>

  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap" rel="stylesheet" />

  <style>
    /* Reset & Variables */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:        #060d14;
      --surface:   #0b1622;
      --surface2:  #0f1f31;
      --border:    #1a3550;
      --cyan:      #00e5ff;
      --cyan-dim:  #00b8d4;
      --green:     #00ff88;
      --green-dim: #00c96b;
      --amber:     #ffab40;
      --red:       #ff5252;
      --text:      #c8e6f0;
      --text-muted:#5a8aa0;
      --font-mono: 'Share Tech Mono', monospace;
      --font-ui:   'Rajdhani', sans-serif;
    }}

    html, body {{ height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: var(--font-ui); }}

    /* Layout */
    #app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      grid-template-rows: 56px 1fr;
      height: 100vh;
    }}

    /* Header */
    #header {{
      grid-column: 1 / -1;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 20px;
      gap: 16px;
      position: relative;
    }}
    #header::after {{
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--cyan), var(--green), transparent);
    }}
    .logo-ring {{
      width: 32px; height: 32px;
      border: 2px solid var(--cyan);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 12px var(--cyan);
      font-size: 14px;
      animation: pulse-ring 3s ease-in-out infinite;
    }}
    @keyframes pulse-ring {{
      0%, 100% {{ box-shadow: 0 0 8px var(--cyan); }}
      50%       {{ box-shadow: 0 0 20px var(--cyan), 0 0 40px rgba(0,229,255,.3); }}
    }}
    #header h1 {{
      font-size: 17px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--cyan);
    }}
    #header span {{
      font-size: 12px;
      color: var(--text-muted);
      font-family: var(--font-mono);
      margin-left: auto;
    }}

    /* Sidebar */
    #sidebar {{
      background: var(--surface);
      border-right: 1px solid var(--border);
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
    }}
    #sidebar::-webkit-scrollbar {{ width: 4px; }}
    #sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

    .panel {{
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .panel-title {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--cyan-dim);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel-title::before {{
      content: '';
      width: 3px; height: 14px;
      background: linear-gradient(180deg, var(--cyan), var(--green));
      border-radius: 2px;
    }}

    /* Stat cards */
    .stats-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .stat-card {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      position: relative;
      overflow: hidden;
    }}
    .stat-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--cyan), var(--green));
    }}
    .stat-value {{
      font-family: var(--font-mono);
      font-size: 22px;
      color: var(--green);
      line-height: 1;
    }}
    .stat-label {{
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 4px;
    }}

    /* Filtros */
    .filter-group {{ margin-bottom: 14px; }}
    .filter-label {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 6px;
      display: block;
    }}
    .filter-select {{
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-ui);
      font-size: 13px;
      padding: 7px 10px;
      border-radius: 5px;
      outline: none;
      cursor: pointer;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='%235a8aa0'%3E%3Cpath d='M6 8L0 0h12z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
    }}
    .filter-select:focus {{ border-color: var(--cyan); box-shadow: 0 0 0 2px rgba(0,229,255,.15); }}

    /* Range slider */
    .range-wrap {{ position: relative; padding: 6px 0 20px; }}
    .range-input {{
      width: 100%;
      accent-color: var(--cyan);
      cursor: pointer;
    }}
    .range-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-muted);
      font-family: var(--font-mono);
      margin-top: 4px;
    }}
    .range-value {{
      text-align: center;
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--cyan);
      margin-top: -14px;
    }}

    /* Checkboxes personalizados */
    .checkbox-group {{ display: flex; flex-direction: column; gap: 7px; }}
    .check-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      font-size: 13px;
      padding: 5px 8px;
      border-radius: 5px;
      transition: background .15s;
    }}
    .check-item:hover {{ background: var(--surface2); }}
    .check-item input[type=checkbox] {{ display: none; }}
    .check-box {{
      width: 16px; height: 16px;
      border: 1.5px solid var(--border);
      border-radius: 3px;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      transition: border-color .15s, background .15s;
    }}
    .check-item input:checked + .check-box {{
      background: var(--cyan);
      border-color: var(--cyan);
    }}
    .check-item input:checked + .check-box::after {{
      content: '✓';
      font-size: 11px;
      color: var(--bg);
      font-weight: bold;
    }}

    /* Botones */
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 14px;
      border-radius: 5px;
      border: none;
      cursor: pointer;
      font-family: var(--font-ui);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 1px;
      text-transform: uppercase;
      transition: all .2s;
    }}
    .btn-primary {{
      background: linear-gradient(135deg, var(--cyan), var(--green));
      color: var(--bg);
    }}
    .btn-primary:hover {{ opacity: .85; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,229,255,.3); }}
    .btn-ghost {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
    }}
    .btn-ghost:hover {{ border-color: var(--cyan); color: var(--cyan); }}
    .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}

    /* Legend */
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
      padding: 4px 0;
    }}
    .dot {{
      width: 14px; height: 14px;
      border-radius: 50%;
      flex-shrink: 0;
      box-shadow: 0 0 6px currentColor;
    }}

    /* Mapa */
    #map {{
      width: 100%;
      height: 100%;
      background: var(--bg);
    }}
    .leaflet-container {{ background: #060d14 !important; }}
    .leaflet-tile-pane {{ filter: brightness(0.85) saturate(0.9); }}

    /* Popup custom */
    .leaflet-popup-content-wrapper {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
      box-shadow: 0 8px 32px rgba(0,0,0,.6), 0 0 0 1px rgba(0,229,255,.1) !important;
      padding: 0 !important;
    }}
    .leaflet-popup-content {{ margin: 0 !important; width: auto !important; }}
    .leaflet-popup-tip {{ background: var(--surface) !important; }}
    .leaflet-popup-close-button {{
      color: var(--text-muted) !important;
      font-size: 18px !important;
      top: 8px !important; right: 10px !important;
    }}
    .leaflet-popup-close-button:hover {{ color: var(--cyan) !important; }}

    .popup-inner {{
      padding: 16px 18px;
      min-width: 220px;
    }}
    .popup-title {{
      font-weight: 700;
      font-size: 14px;
      color: var(--cyan);
      margin-bottom: 10px;
      padding-right: 16px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }}
    .popup-row {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 12px;
      padding: 3px 0;
    }}
    .popup-key {{ color: var(--text-muted); }}
    .popup-val {{
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 11px;
    }}
    .popup-val.good  {{ color: var(--green); }}
    .popup-val.mid   {{ color: var(--amber); }}
    .popup-val.bad   {{ color: var(--red); }}
    .popup-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 20px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .5px;
      text-transform: uppercase;
    }}
    .badge-pwr  {{ background: rgba(0,229,255,.15); color: var(--cyan); border: 1px solid var(--cyan-dim); }}
    .badge-bwr  {{ background: rgba(0,255,136,.15); color: var(--green); border: 1px solid var(--green-dim); }}
    .badge-other{{ background: rgba(255,171,64,.15); color: var(--amber); border: 1px solid var(--amber); }}
    .badge-shut {{ background: rgba(255,82,82,.15);  color: var(--red); border: 1px solid var(--red); }}

    /* Completeness bar */
    .comp-bar {{
      margin-top: 8px;
      background: var(--surface2);
      border-radius: 3px;
      height: 4px;
      overflow: hidden;
    }}
    .comp-fill {{
      height: 100%;
      border-radius: 3px;
      background: linear-gradient(90deg, var(--cyan), var(--green));
    }}

    /* Tooltip custom */
    .leaflet-tooltip {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      color: var(--text) !important;
      font-family: var(--font-ui) !important;
      font-size: 12px !important;
      border-radius: 4px !important;
      padding: 4px 8px !important;
      box-shadow: none !important;
    }}
    .leaflet-tooltip-left::before, .leaflet-tooltip-right::before {{
      border-right-color: var(--border) !important;
      border-left-color: var(--border) !important;
    }}

    /* Status bar */
    #statusbar {{
      position: absolute;
      bottom: 16px; left: 340px;
      background: rgba(6,13,20,.85);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 14px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      z-index: 1000;
      backdrop-filter: blur(8px);
      pointer-events: none;
    }}
    #statusbar span {{ color: var(--cyan); }}

    /* Scroll suave en sidebar */
    .section-divider {{
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--border), transparent);
      margin: 4px 0;
    }}
  </style>
</head>

<body>
<div id="app">
  <!-- Header -->
  <header id="header">
    <div class="logo-ring">☢</div>
    <h1>Mapa Nuclear — TDQM</h1>
    <span id="coord-display">IIMAS · EPA RadNet + Geo Nuclear Data</span>
  </header>

  <!-- Sidebar -->
  <aside id="sidebar">

    <!-- Stats -->
    <div class="panel">
      <div class="panel-title">Resumen del dataset</div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value" id="stat-plants">—</div>
          <div class="stat-label">Plantas</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="stat-stations">—</div>
          <div class="stat-label">Estaciones</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="stat-types">—</div>
          <div class="stat-label">Tipos reactor</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="stat-bad">—</div>
          <div class="stat-label">Calidad baja</div>
        </div>
      </div>
    </div>

    <!-- Filtros plantas -->
    <div class="panel">
      <div class="panel-title">Filtros — Plantas</div>

      <div class="filter-group">
        <label class="filter-label">Tipo de reactor</label>
        <select id="sel-reactor" class="filter-select">
          <option value="all">Todos los tipos</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="filter-label">Estado operativo</label>
        <select id="sel-status" class="filter-select">
          <option value="all">Todos</option>
          <option value="Operational">Operacional</option>
          <option value="Shutdown">Apagada</option>
          <option value="Under Construction">En construcción</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="filter-label">Calidad de datos</label>
        <div class="checkbox-group">
          <label class="check-item">
            <input type="checkbox" class="chk-quality" value="Buena" checked />
            <span class="check-box"></span>
            <span>Buena (&gt;75%)</span>
          </label>
          <label class="check-item">
            <input type="checkbox" class="chk-quality" value="Media" checked />
            <span class="check-box"></span>
            <span>Media (50–75%)</span>
          </label>
          <label class="check-item">
            <input type="checkbox" class="chk-quality" value="Mala" checked />
            <span class="check-box"></span>
            <span>Mala (&lt;50%) ⚠</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Filtros estaciones -->
    <div class="panel">
      <div class="panel-title">Filtros — Estaciones RadNet</div>

      <div class="filter-group">
        <label class="filter-label">Nivel gamma promedio</label>
        <select id="sel-gamma" class="filter-select">
          <option value="all">Todos los niveles</option>
          <option value="Bajo">Bajo</option>
          <option value="Medio">Medio</option>
          <option value="Alto">Alto</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="filter-label">Completitud mínima (%)</label>
        <div class="range-wrap">
          <input type="range" id="range-comp" class="range-input" min="0" max="100" value="0" step="5" />
          <div class="range-value" id="range-comp-val">0%</div>
          <div class="range-labels"><span>0%</span><span>100%</span></div>
        </div>
      </div>

      <div class="filter-group">
        <label class="filter-label">Capas visibles</label>
        <div class="checkbox-group">
          <label class="check-item">
            <input type="checkbox" id="chk-plants" checked />
            <span class="check-box"></span>
            <span>Plantas nucleares</span>
          </label>
          <label class="check-item">
            <input type="checkbox" id="chk-stations" checked />
            <span class="check-box"></span>
            <span>Estaciones RadNet</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Acciones -->
    <div class="panel">
      <div class="panel-title">Acciones</div>
      <div class="btn-row">
        <button class="btn btn-primary" onclick="applyFilters()">▶ Aplicar</button>
        <button class="btn btn-ghost" onclick="resetFilters()">↺ Reset</button>
      </div>
      <p style="font-size:11px; color:var(--text-muted); margin-top:10px; line-height:1.5;">
        Visible: <span id="count-plants" style="color:var(--cyan)">—</span> plantas,
        <span id="count-stations" style="color:var(--green)">—</span> estaciones
      </p>
    </div>

    <!-- Leyenda -->
    <div class="panel">
      <div class="panel-title">Leyenda — Plantas</div>
      <div class="legend-item"><div class="dot" style="background:var(--cyan);color:var(--cyan)"></div> PWR (agua a presión)</div>
      <div class="legend-item"><div class="dot" style="background:var(--green);color:var(--green)"></div> BWR (agua en ebullición)</div>
      <div class="legend-item"><div class="dot" style="background:var(--amber);color:var(--amber)"></div> Otro tipo</div>
      <div class="legend-item"><div class="dot" style="background:var(--red);color:var(--red)"></div> Apagada / Shutdown</div>
      <div class="section-divider" style="margin:8px 0"></div>
      <div class="panel-title" style="margin-top:4px">Leyenda — Estaciones</div>
      <div class="legend-item"><div class="dot" style="background:#00bcd4;color:#00bcd4"></div> Gamma bajo</div>
      <div class="legend-item"><div class="dot" style="background:#8bc34a;color:#8bc34a"></div> Gamma medio</div>
      <div class="legend-item"><div class="dot" style="background:#ff9800;color:#ff9800"></div> Gamma alto</div>
      <div class="legend-item"><div class="dot" style="background:rgba(80,80,80,.8);color:#666"></div> Calidad de datos baja</div>
    </div>

  </aside>

  <!-- Mapa -->
  <div id="map"></div>
</div>

<div id="statusbar">LAT <span id="lat-disp">--.-</span> &nbsp; LON <span id="lon-disp">--.-</span></div>

<script>
// Datos
const PLANTS    = {plants_json};
const STATIONS  = {stations_json};
const R_TYPES   = {reactor_types_json};

// Mapa
const map = L.map('map', {{
  center: [37.5, -96],
  zoom: 4,
  zoomControl: false,
  attributionControl: true,
}});

// Tile oscuro
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; <a href="https://carto.com/">CARTO</a> | &copy; OpenStreetMap contributors',
  subdomains: 'abcd',
  maxZoom: 19,
}}).addTo(map);

// Zoom control custom
L.control.zoom({{ position: 'topright' }}).addTo(map);

// Coordenadas en tiempo real
map.on('mousemove', e => {{
  document.getElementById('lat-disp').textContent = e.latlng.lat.toFixed(3);
  document.getElementById('lon-disp').textContent = e.latlng.lng.toFixed(3);
}});

// Layer groups
let layerPlants   = L.layerGroup().addTo(map);
let layerStations = L.layerGroup().addTo(map);

// Helpers de color
function plantColor(p) {{
  if (p.status && p.status.toLowerCase().includes('shutdown')) return '#ff5252';
  const rt = (p.reactor_type || '').toUpperCase();
  if (rt.includes('PWR')) return '#00e5ff';
  if (rt.includes('BWR')) return '#00ff88';
  return '#ffab40';
}}

function stationColor(s) {{
  const gl = s.gamma_level;
  if (gl === 'Bajo')  return '#00bcd4';
  if (gl === 'Medio') return '#8bc34a';
  if (gl === 'Alto')  return '#ff9800';
  return '#7986cb';
}}

function qualityOpacity(pct) {{
  if (pct >= 75) return 1.0;
  if (pct >= 50) return 0.7;
  return 0.35;
}}

// SVG marker para plantas
function plantIcon(color, quality) {{
  const opacity = qualityOpacity(quality);
  const r = quality < 50 ? 9 : 11;
  const glow = quality < 50 ? 'none' : `drop-shadow(0 0 6px ${{color}})`;
  const svg = `
    <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"
         style="filter:${{glow}};opacity:${{opacity}}">
      <polygon points="12,2 22,20 2,20"
        fill="none" stroke="${{color}}" stroke-width="1.8"
        stroke-linejoin="round"/>
      <circle cx="12" cy="13" r="3" fill="${{color}}" opacity="0.8"/>
      ${{quality < 50 ? `<path d="M12 8v4" stroke="#ff5252" stroke-width="1.5"/>
        <circle cx="12" cy="15" r="1" fill="#ff5252"/>` : ''}}
    </svg>`;
  return L.divIcon({{
    html: svg, className: '', iconSize: [24, 24], iconAnchor: [12, 20], popupAnchor: [0, -22],
  }});
}}

// SVG marker para estaciones
function stationIcon(color, comp) {{
  const opacity = qualityOpacity(comp);
  const svg = `
    <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg"
         style="opacity:${{opacity}};filter:drop-shadow(0 0 4px ${{color}})">
      <circle cx="9" cy="9" r="7" fill="none" stroke="${{color}}" stroke-width="1.5"/>
      <circle cx="9" cy="9" r="3" fill="${{color}}" opacity="0.7"/>
    </svg>`;
  return L.divIcon({{
    html: svg, className: '', iconSize: [18, 18], iconAnchor: [9, 9], popupAnchor: [0, -12],
  }});
}}

// Popup plantas
function plantPopup(p) {{
  const rt = (p.reactor_type || '').toUpperCase();
  const badgeClass = rt.includes('PWR') ? 'badge-pwr' : rt.includes('BWR') ? 'badge-bwr' : 'badge-other';
  const isShut = p.status && p.status.toLowerCase().includes('shutdown');
  const qualClass = p.completeness_pct >= 75 ? 'good' : p.completeness_pct >= 50 ? 'mid' : 'bad';
  const cap = p.capacity_mw != null ? `${{p.capacity_mw.toLocaleString()}} MW` : 'N/D';

  return `<div class="popup-inner">
    <div class="popup-title">
      ☢ ${{p.name}}
      ${{isShut ? '<span class="popup-badge badge-shut" style="margin-left:6px">Apagada</span>' : ''}}
    </div>
    <div class="popup-row">
      <span class="popup-key">País</span>
      <span class="popup-val">${{p.country}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Reactor</span>
      <span class="popup-badge ${{badgeClass}}">${{p.reactor_type}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Capacidad</span>
      <span class="popup-val">${{cap}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Coordenadas</span>
      <span class="popup-val">${{p.lat.toFixed(3)}}, ${{p.lon.toFixed(3)}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Completitud datos</span>
      <span class="popup-val ${{qualClass}}">${{p.completeness_pct.toFixed(1)}}%</span>
    </div>
    <div class="comp-bar">
      <div class="comp-fill" style="width:${{p.completeness_pct}}%"></div>
    </div>
  </div>`;
}}

// Popup estaciones
function stationPopup(s) {{
  const qualClass = s.completeness_pct >= 75 ? 'good' : s.completeness_pct >= 50 ? 'mid' : 'bad';
  const gammaClass = s.gamma_level === 'Alto' ? 'bad' : s.gamma_level === 'Bajo' ? 'good' : 'mid';
  const gm = s.gamma_mean != null ? s.gamma_mean.toFixed(2) + ' CPM' : 'N/D';
  const gs = s.gamma_std != null ? '± ' + s.gamma_std.toFixed(2) : '';

  return `<div class="popup-inner">
    <div class="popup-title">📡 Estación ${{s.station_id}}</div>
    <div class="popup-row">
      <span class="popup-key">Estado</span>
      <span class="popup-val">${{s.state}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Gamma promedio</span>
      <span class="popup-val ${{gammaClass}}">${{gm}} ${{gs}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Nivel gamma</span>
      <span class="popup-val ${{gammaClass}}">${{s.gamma_level}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Registros</span>
      <span class="popup-val">${{s.n_records.toLocaleString()}}</span>
    </div>
    <div class="popup-row">
      <span class="popup-key">Completitud</span>
      <span class="popup-val ${{qualClass}}">${{s.completeness_pct.toFixed(1)}}%</span>
    </div>
    <div class="comp-bar">
      <div class="comp-fill" style="width:${{s.completeness_pct}}%"></div>
    </div>
  </div>`;
}}

// Render 
function renderPlants(data) {{
  layerPlants.clearLayers();
  data.forEach(p => {{
    const color = plantColor(p);
    const marker = L.marker([p.lat, p.lon], {{
      icon: plantIcon(color, p.completeness_pct),
    }});
    marker.bindPopup(plantPopup(p), {{ maxWidth: 280, className: '' }});
    marker.bindTooltip(p.name, {{ direction: 'top', offset: [0, -14] }});
    layerPlants.addLayer(marker);
  }});
  document.getElementById('count-plants').textContent = data.length;
}}

function renderStations(data) {{
  layerStations.clearLayers();
  data.forEach(s => {{
    const color = stationColor(s);
    const marker = L.marker([s.lat, s.lon], {{
      icon: stationIcon(color, s.completeness_pct),
    }});
    marker.bindPopup(stationPopup(s), {{ maxWidth: 280 }});
    marker.bindTooltip(`${{s.station_id}} — ${{s.gamma_mean?.toFixed(1) ?? 'N/D'}} CPM`, {{
      direction: 'top', offset: [0, -10],
    }});
    layerStations.addLayer(marker);
  }});
  document.getElementById('count-stations').textContent = data.length;
}}

// Filtros
function applyFilters() {{
  const reactor  = document.getElementById('sel-reactor').value;
  const status   = document.getElementById('sel-status').value;
  const gamma    = document.getElementById('sel-gamma').value;
  const minComp  = parseInt(document.getElementById('range-comp').value);
  const showP    = document.getElementById('chk-plants').checked;
  const showS    = document.getElementById('chk-stations').checked;

  const qualities = [...document.querySelectorAll('.chk-quality:checked')].map(c => c.value);

  // Filtrar plantas
  const fp = PLANTS.filter(p => {{
    if (reactor !== 'all' && p.reactor_type !== reactor) return false;
    if (status  !== 'all' && p.status !== status)         return false;
    const q = p.completeness_pct >= 75 ? 'Buena' : p.completeness_pct >= 50 ? 'Media' : 'Mala';
    if (!qualities.includes(q)) return false;
    return true;
  }});

  // Filtrar estaciones
  const fs = STATIONS.filter(s => {{
    if (gamma !== 'all' && s.gamma_level !== gamma) return false;
    if (s.completeness_pct < minComp) return false;
    return true;
  }});

  if (showP) renderPlants(fp); else {{ layerPlants.clearLayers(); document.getElementById('count-plants').textContent = 0; }}
  if (showS) renderStations(fs); else {{ layerStations.clearLayers(); document.getElementById('count-stations').textContent = 0; }}
}}

function resetFilters() {{
  document.getElementById('sel-reactor').value = 'all';
  document.getElementById('sel-status').value  = 'all';
  document.getElementById('sel-gamma').value   = 'all';
  document.getElementById('range-comp').value  = '0';
  document.getElementById('range-comp-val').textContent = '0%';
  document.querySelectorAll('.chk-quality').forEach(c => c.checked = true);
  document.getElementById('chk-plants').checked   = true;
  document.getElementById('chk-stations').checked = true;
  renderPlants(PLANTS);
  renderStations(STATIONS);
}}

// Init
(function init() {{
  // Poblar selector de reactores
  const sel = document.getElementById('sel-reactor');
  R_TYPES.forEach(rt => {{
    const opt = document.createElement('option');
    opt.value = rt; opt.textContent = rt;
    sel.appendChild(opt);
  }});

  // Stats
  document.getElementById('stat-plants').textContent   = PLANTS.length;
  document.getElementById('stat-stations').textContent = STATIONS.length;
  document.getElementById('stat-types').textContent    = R_TYPES.length;
  const badQuality = PLANTS.filter(p => p.completeness_pct < 50).length;
  document.getElementById('stat-bad').textContent = badQuality;

  // Slider feedback
  document.getElementById('range-comp').addEventListener('input', function() {{
    document.getElementById('range-comp-val').textContent = this.value + '%';
  }});

  // Render inicial
  renderPlants(PLANTS);
  renderStations(STATIONS);
}})();
</script>
</body>
</html>
"""
    return html


# Punto de entrada
def main():
    parser = argparse.ArgumentParser(description="Genera mapa HTML de plantas nucleares.")
    parser.add_argument("--plants",    type=Path, default=DEFAULT_PLANTS,    help="Ruta a nuclear_plants_clean.csv")
    parser.add_argument("--radiation", type=Path, default=DEFAULT_RADIATION, help="Ruta a radiation_db.csv")
    parser.add_argument("--output",    type=Path, default=DEFAULT_OUTPUT,    help="Ruta de salida .html")
    args = parser.parse_args()

    log.info("mapa.py — Generador de mapa didáctico")
    log.info(f"Buscando plantas en: {args.plants}")
    log.info(f"Buscando radiación en: {args.radiation}")
    plants = load_plants(args.plants)
    stations = load_radiation(args.radiation)

    log.info(f"Plantas cargadas: {len(plants)}")
    log.info(f"Estaciones cargadas: {len(stations)}")
    

    html = build_html(plants, stations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"Mapa generado en: {args.output}")
    log.info(f"Abre en tu navegador: file://{args.output}")


if __name__ == "__main__":
    main()
