"""
analisis.py — 15 queries con visualizaciones sobre datos de radiación ambiental

Responde 5 preguntas de negocio (3 queries cada una):

  P1 — ¿Las estaciones cercanas a plantas registran gamma_cpm distinto?
  P2 — ¿Es posible detectar anomalías automáticamente?
  P3 — ¿Qué tipo/capacidad de reactor correlaciona más con gamma_cpm?
  P4 — ¿Es posible predecir gamma_cpm por historial y proximidad?
  P5 — ¿Qué estaciones tienen peores problemas de calidad?

Entradas:
  radiation_clean.csv
  nuclear_plants_clean.csv
  station_plant_distances.csv

Salidas:
  datos_procesados/reportes/figuras/Q01_*.png … Q15_*.png

Uso:
  python scripts/analisis.py
  python scripts/analisis.py --output-dir datos_procesados/reportes/figuras
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Logging y estilo
logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s", datefmt = "%H:%M:%S")
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE_STD = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (10, 10)
DPI = 150

# Bandas de distancia para P1
DIST_BINS = [0, 50, 200, np.inf]
DIST_LABELS = ["< 50 km", "50-200 km", "> 200 km"]

# Carga de datos
def load_data(rad_path: str, plants_path: str, dist_path: str) -> tuple:
    """
    Carga las tres tablas y construye datasets derivados de uso frecuente.
    Para radiation_clean (17M filas) se hace una agregación temprana por estación
    para evitar operaciones costosas en cada query.
    """
    log.info("Cargando radiation_clean…")
    rad = pd.read_csv(rad_path, low_memory = False, parse_dates = ["timestamp"],
        dtype = {"station_id": str, "gamma_imputed": bool, "is_outlier": bool})
    log.info("%d filas, %d columnas", *rad.shape)

    log.info("Cargando nuclear_plants_clean")
    plants = pd.read_csv(plants_path, low_memory = False)

    log.info("Cargando station_plant_distances")
    dist = pd.read_csv(dist_path, low_memory = False)

    # Dataset de la planta más cercana por estación
    closest = dist[dist["rank_proximity"] == 1].copy()

    # Agregación a nivel de estación
    log.info("Calculando agregaciones por estación…")
    station_agg = (rad.groupby("station_id").agg(
            gamma_cpm_mean=("gamma_cpm", "mean"),
            gamma_cpm_median=("gamma_cpm", "median"),
            gamma_cpm_std=("gamma_cpm", "std"),
            n_registros=("gamma_cpm", "count"),
            n_outliers=("is_outlier", "sum"),
            n_imputed=("gamma_imputed", "sum"),
            n_dose_nulos=("dose_nSv_h", lambda x: x.isna().sum()),
            n_total=("gamma_cpm", "size"),
            years_of_data=("years_of_data", "first"),
            state=("state", lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)
        ).reset_index())
    station_agg["pct_outliers"] = station_agg["n_outliers"] / station_agg["n_total"]
    station_agg["pct_imputed"] = station_agg["n_imputed"] / station_agg["n_total"]
    station_agg["pct_dose_nulos"] = station_agg["n_dose_nulos"] / station_agg["n_total"]

    # Unir con distancia a planta más cercana
    station_agg = station_agg.merge(closest[["station_id", "distance_km", "reactor_type_std", "capacity_mw", 
                                             "plant_name", "plant_status"]], on = "station_id", how = "left")

    # Banda de distancia
    station_agg["dist_band"] = pd.cut(station_agg["distance_km"], bins = DIST_BINS, labels = DIST_LABELS, right = False)

    log.info("Datasets listos.")
    return rad, plants, dist, station_agg, closest


# Guardado
def save_fig(fig: plt.Figure, path: Path, query_id: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    log.info("Q%s a %s", query_id, path.name)

# P1 — Dosis vs distancia a plantas (Q01–Q03)
def q01_distribucion_por_banda(station_agg: pd.DataFrame, out: Path) -> None:
    """Q01 — Histograma de gamma_cpm por banda de distancia."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    palette = {"< 50 km": "#d62728", "50-200 km": "#ff7f0e", "> 200 km": "#1f77b4"}

    for band, grp in station_agg.dropna(subset=["dist_band"]).groupby("dist_band", observed = True):
        
        gamma_na = grp['gamma_cpm_mean'].dropna()
        ax.hist(gamma_na, bins = 20, alpha = 0.6, label = band, color = palette.get(str(band)))

    ax.set_xlabel("gamma_cpm promedio por estación")
    ax.set_ylabel("Número de estaciones")
    ax.set_title("P1 — Q01: Distribución de gamma_cpm por banda de distancia a planta")
    ax.legend(title="Distancia a planta más cercana")
    save_fig(fig, out / "Q01_distribucion_banda_distancia.png", "01")


def q02_boxplot_por_banda(station_agg: pd.DataFrame, out: Path) -> None:
    """Q02 — Boxplot de gamma_cpm por banda de distancia."""
    data = station_agg.dropna(subset=["dist_band", "gamma_cpm_mean"])
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    sns.boxplot(data=data, x="dist_band", y="gamma_cpm_mean", order=DIST_LABELS, palette="Set2", ax=ax)
    ax.set_xlabel("Distancia a planta operacional más cercana")
    ax.set_ylabel("gamma_cpm promedio")
    ax.set_title("P1 — Q02: Boxplot de gamma_cpm por banda de distancia")

    # Añadir medias como puntos
    for i, band in enumerate(DIST_LABELS):
        vals = data[data["dist_band"] == band]["gamma_cpm_mean"]
        if not vals.empty:
            ax.scatter(i, vals.mean(), color="black", zorder=5, s=60, marker="D")

    save_fig(fig, out / "Q02_boxplot_banda_distancia.png", "02")


def q03_scatter_distancia_gamma(station_agg: pd.DataFrame, out: Path) -> None:
    """Q03 — Scatter distance_km vs gamma_cpm con regresión lineal."""
    data = station_agg.dropna(subset=["distance_km", "gamma_cpm_mean"])
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    ax.scatter(
        data["distance_km"], data["gamma_cpm_mean"],
        alpha=0.6, s=50, color="#4c72b0", edgecolors="white", linewidths=0.5,
    )

    # Regresión
    slope, intercept, r, p, _ = stats.linregress(
        data["distance_km"], data["gamma_cpm_mean"]
    )
    x_line = np.linspace(data["distance_km"].min(), data["distance_km"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2,
            label=f"Regresión: r={r:.3f}, p={p:.4f}")

    ax.set_xlabel("Distancia a planta más cercana (km)")
    ax.set_ylabel("gamma_cpm promedio")
    ax.set_title("P1 — Q03: Relación entre distancia a planta y gamma_cpm")
    ax.legend()
    save_fig(fig, out / "Q03_scatter_distancia_gamma.png", "03")


# P2 — Detección de anomalías (Q04–Q06)
def q04_serie_temporal_outliers(rad: pd.DataFrame, station_agg: pd.DataFrame, out: Path) -> None:
    """Q04 — Serie temporal de la estación con más outliers, marcando anomalías."""
    # Elegir la estación con más outliers absolutos que tenga serie larga
    candidata = (station_agg[station_agg["years_of_data"] >= 3].nlargest(1, "n_outliers")["station_id"].iloc[0])
    log.info("Q04: estación seleccionada = %s", candidata)

    serie = (rad[rad["station_id"] == candidata].set_index("timestamp")["gamma_cpm"].resample("W").median().dropna())
    outlier_ts = (rad[(rad["station_id"] == candidata) & (rad["is_outlier"])]
        .set_index("timestamp")["gamma_cpm"].resample("W").median().dropna())

    fig, ax = plt.subplots(figsize = FIGSIZE_WIDE)
    ax.plot(serie.index, serie.values, color="#4c72b0", linewidth=1, label="gamma_cpm semanal")
    ax.scatter(outlier_ts.index, outlier_ts.values, color="#d62728", s=30, zorder=5, label="Outlier IQR", alpha=0.8)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("gamma_cpm (mediana semanal)")
    ax.set_title(f"P2 — Q04: Serie temporal con anomalías — {candidata}")
    ax.legend()
    save_fig(fig, out / "Q04_serie_temporal_outliers.png", "04")

def q05_top10_estaciones_outliers(station_agg: pd.DataFrame, out: Path) -> None:
    """Q05 — Top 10 estaciones por porcentaje de outliers."""
    top10 = station_agg.nlargest(10, "pct_outliers")[["station_id", "pct_outliers"]].sort_values("pct_outliers")

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    bars = ax.barh(top10["station_id"], top10["pct_outliers"] * 100, color=sns.color_palette("Reds_r", 10))
    ax.set_xlabel("% de lecturas marcadas como outlier")
    ax.set_title("P2 — Q05: Top 10 estaciones con mayor tasa de anomalías")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    for bar, val in zip(bars, top10["pct_outliers"]):

        ax.text(val * 100 + 0.2, bar.get_y() + bar.get_height() / 2, f"{val*100:.1f}%", va="center", fontsize=9)
    
    save_fig(fig, out / "Q05_top10_outliers.png", "05")

def q06_heatmap_outliers_anio(rad: pd.DataFrame, station_agg: pd.DataFrame, out: Path) -> None:
    """Q06 — Heatmap de % outliers por estación (top 20) y año."""
    top20 = station_agg.nlargest(20, "pct_outliers")["station_id"].tolist()
    subset = rad[rad["station_id"].isin(top20)].copy()
    subset["year"] = subset["timestamp"].dt.year

    pivot = (subset.groupby(["station_id", "year"])["is_outlier"].mean().unstack(fill_value=0) * 100)
    # Ordenar estaciones por tasa total
    pivot = pivot.loc[top20]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
        cbar_kws={"label": "% outliers"}, linewidths=0.3)
    ax.set_title("P2 — Q06: Frecuencia de outliers por estación y año (%)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Estación")
    save_fig(fig, out / "Q06_heatmap_outliers_anio.png", "06")


# P3 — Tipo de reactor vs gamma_cpm (Q07–Q09)
def q07_gamma_por_reactor_type(station_agg: pd.DataFrame, out: Path) -> None:
    """Q07 — gamma_cpm promedio por tipo de reactor."""
    data = (station_agg.dropna(subset=["reactor_type_std", "gamma_cpm_mean"])
        .groupby("reactor_type_std")["gamma_cpm_mean"].agg(["mean", "sem", "count"])
        .reset_index().sort_values("mean", ascending=False))
    
    data = data[data["count"] >= 2]  # al menos 2 estaciones

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    bars = ax.bar(data["reactor_type_std"], data["mean"], yerr=data["sem"], capsize=4,
                  color=sns.color_palette("Set2", len(data)), edgecolor="white")
    ax.set_xlabel("Tipo de reactor (categoría IAEA)")
    ax.set_ylabel("gamma_cpm promedio ± SEM")
    ax.set_title("P3 — Q07: gamma_cpm promedio por tipo de reactor más cercano")

    for bar in bars:

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    save_fig(fig, out / "Q07_gamma_por_reactor_type.png", "07")


def q08_scatter_capacity_gamma(station_agg: pd.DataFrame, out: Path) -> None:
    """Q08 — Scatter capacity_mw vs gamma_cpm con regresión por tipo de reactor."""
    data = station_agg.dropna(subset=["capacity_mw", "gamma_cpm_mean", "reactor_type_std"])

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    palette = sns.color_palette("tab10", data["reactor_type_std"].nunique())
    color_map = dict(zip(data["reactor_type_std"].unique(), palette))

    for rtype, grp in data.groupby("reactor_type_std"):
        ax.scatter(grp["capacity_mw"], grp["gamma_cpm_mean"], label=rtype, color=color_map[rtype], alpha=0.7, s=60)

    # Regresión global
    slope, intercept, r, p, _ = stats.linregress(data["capacity_mw"], data["gamma_cpm_mean"])
    x_line = np.linspace(data["capacity_mw"].min(), data["capacity_mw"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.5, label=f"Global r={r:.3f}")

    ax.set_xlabel("Capacidad instalada (MW)")
    ax.set_ylabel("gamma_cpm promedio")
    ax.set_title("P3 — Q08: Capacidad del reactor vs gamma_cpm por tipo")
    ax.legend(title="Tipo reactor", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_fig(fig, out / "Q08_scatter_capacity_gamma.png", "08")


def q09_heatmap_correlacion(station_agg: pd.DataFrame, out: Path) -> None:
    """Q09 — Heatmap de correlación entre capacity_mw, distance_km y gamma_cpm."""
    cols = ["capacity_mw", "distance_km", "gamma_cpm_mean", "gamma_cpm_std", "pct_outliers", "years_of_data"]
    corr_data = station_agg[cols].dropna()
    corr = corr_data.corr(method="spearman")

    labels = {
        "capacity_mw": "Capacidad (MW)",
        "distance_km": "Distancia (km)",
        "gamma_cpm_mean": "gamma_cpm medio",
        "gamma_cpm_std": "gamma_cpm std",
        "pct_outliers": "% outliers",
        "years_of_data": "Años de datos"
    }
    corr = corr.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, ax=ax, mask=mask, linewidths=0.5, cbar_kws={"label": "Correlación de Spearman"})
    ax.set_title("P3 — Q09: Correlación Spearman entre variables de interés")
    save_fig(fig, out / "Q09_heatmap_correlacion.png", "09")

# P4 — Predicción de gamma_cpm (Q10–Q12)
def _rolling_forecast(serie: pd.Series, window: int = 12) -> pd.Series:
    """Media móvil centrada como predictor de referencia."""
    return serie.rolling(window=window, center=True, min_periods=4).mean()

def q10_descomposicion_serie(rad: pd.DataFrame, station_agg: pd.DataFrame, out: Path) -> None:
    """Q10 — Tendencia + estacionalidad de la estación con más años de datos."""
    candidata = station_agg.nlargest(1, "years_of_data")["station_id"].iloc[0]
    log.info("  Q10: estación seleccionada = %s", candidata)

    serie = (rad[rad["station_id"] == candidata].set_index("timestamp")["gamma_cpm"].resample("ME").median().dropna())

    tendencia = serie.rolling(12, center=True, min_periods=6).mean()
    residuo = serie - tendencia

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(serie.index, serie.values, color="#4c72b0", linewidth=1)
    axes[0].set_ylabel("gamma_cpm")
    axes[0].set_title(f"P4 — Q10: Descomposición de serie temporal — {candidata}")

    axes[1].plot(tendencia.index, tendencia.values, color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("Tendencia")

    axes[2].bar(residuo.index, residuo.values, color="#d62728", alpha=0.6, width=20)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Residuo")
    axes[2].set_xlabel("Fecha")

    save_fig(fig, out / "Q10_descomposicion_serie.png", "10")


def q11_prediccion_media_movil(rad: pd.DataFrame, station_agg: pd.DataFrame, out: Path) -> None:
    """Q11 — Predicción con media móvil (ventana 12 meses) vs valores reales."""
    # Elegir estación con rango largo y pocos outliers
    candidata = (station_agg[station_agg["years_of_data"] >= 5].nsmallest(1, "pct_outliers")["station_id"].iloc[0])
    log.info("  Q11: estación seleccionada = %s", candidata)

    serie = (rad[rad["station_id"] == candidata].set_index("timestamp")["gamma_cpm"].resample("ME").median().dropna())
    pred = _rolling_forecast(serie, window=12)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(serie.index, serie.values, color="#4c72b0", linewidth=1, alpha=0.8, label="Real")
    ax.plot(pred.index, pred.values, color="#d62728", linewidth=2, label="Media móvil (12 meses)")
    ax.fill_between(serie.index, pred - serie.std(), pred + serie.std(), alpha=0.15, color="#d62728", label="±1 SD")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("gamma_cpm (mediana mensual)")
    ax.set_title(f"P4 — Q11: Predicción por media móvil vs real — {candidata}")
    ax.legend()
    save_fig(fig, out / "Q11_prediccion_media_movil.png", "11")


def q12_error_vs_distancia(rad: pd.DataFrame, station_agg: pd.DataFrame, out: Path) -> None:
    """Q12 — MAE de media móvil por estación vs distancia a planta más cercana."""
    resultados = []
    # Solo estaciones con >= 24 meses de datos
    candidatas = station_agg[station_agg["years_of_data"] >= 2]["station_id"].tolist()

    log.info("  Q12: calculando MAE para %d estaciones", len(candidatas))
    for sid in candidatas:
        serie = (rad[rad["station_id"] == sid].set_index("timestamp")["gamma_cpm"].resample("ME").median().dropna())
        if len(serie) < 12:

            continue
        
        pred = _rolling_forecast(serie, window=12).dropna()
        real = serie.loc[pred.index]
        mae = (real - pred).abs().mean()
        resultados.append({"station_id": sid, "mae": mae})

    mae_df = pd.DataFrame(resultados).merge(station_agg[["station_id", "distance_km"]], on="station_id").dropna()

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.scatter(mae_df["distance_km"], mae_df["mae"], alpha=0.65, s=55, color="#9467bd", edgecolors="white")

    slope, intercept, r, p, _ = stats.linregress(mae_df["distance_km"], mae_df["mae"])
    x_line = np.linspace(mae_df["distance_km"].min(), mae_df["distance_km"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2, label=f"r={r:.3f}, p={p:.4f}")

    ax.set_xlabel("Distancia a planta más cercana (km)")
    ax.set_ylabel("MAE media móvil (cpm)")
    ax.set_title("P4 — Q12: Error de predicción vs proximidad a planta nuclear")
    ax.legend()
    save_fig(fig, out / "Q12_error_vs_distancia.png", "12")


# P5 — Calidad por estación (Q13–Q15)
def _indice_calidad(row: pd.Series) -> float:
    """
    Índice compuesto de calidad [0–1], donde 1 = peor calidad.
    Componentes (peso igual):
      - % dose_nSv_h nulos
      - % gamma_cpm imputados
      - % outliers IQR
      - 1 - (años_datos / max_años) → penaliza series cortas
    """
    max_years = 20.0
    score = (
        row["pct_dose_nulos"] +
        row["pct_imputed"] +
        row["pct_outliers"] +
        (1 - min(row["years_of_data"], max_years) / max_years)
    ) / 4
    return round(score, 4)


def q13_ranking_calidad(station_agg: pd.DataFrame, out: Path) -> None:
    """Q13 — Ranking de las 20 peores estaciones por índice compuesto de calidad."""
    df = station_agg.copy()
    df["indice_calidad"] = df.apply(_indice_calidad, axis=1)
    top20 = df.nlargest(20, "indice_calidad").sort_values("indice_calidad")

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.cm.get_cmap("RdYlGn_r")
    colors = [cmap(v) for v in (top20["indice_calidad"] - top20["indice_calidad"].min()) /
              (top20["indice_calidad"].max() - top20["indice_calidad"].min() + 1e-9)]

    bars = ax.barh(top20["station_id"], top20["indice_calidad"], color=colors)
    ax.set_xlabel("Índice compuesto de calidad (0 = mejor, 1 = peor)")
    ax.set_title("P5 — Q13: Ranking de las 20 estaciones con peor calidad de datos")
    ax.axvline(top20["indice_calidad"].mean(), color="navy", linestyle="--",
               linewidth=1.5, label=f"Media = {top20['indice_calidad'].mean():.3f}")
    ax.legend()

    for bar, val in zip(bars, top20["indice_calidad"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    save_fig(fig, out / "Q13_ranking_calidad.png", "13")


def q14_nulos_dose_por_estacion(station_agg: pd.DataFrame, out: Path) -> None:
    """Q14 — % nulos en dose_nSv_h por estación, ordenado descendente."""
    data = station_agg.sort_values("pct_dose_nulos", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#d62728" if v >= 0.5 else "#ff7f0e" if v >= 0.2 else "#2ca02c"
              for v in data["pct_dose_nulos"]]
    ax.barh(data["station_id"], data["pct_dose_nulos"] * 100, color=colors)
    ax.axvline(69.9, color="black", linestyle="--", linewidth=1, label="Promedio global (69.9%)")
    ax.set_xlabel("% de registros con dose_nSv_h nulo")
    ax.set_title("P5 — Q14: Completitud de dose_nSv_h por estación")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend()
    save_fig(fig, out / "Q14_nulos_dose_por_estacion.png", "14")


def q15_imputed_vs_real(station_agg: pd.DataFrame, out: Path) -> None:
    """Q15 — Registros imputados vs reales por estación (top 20 con más imputados)."""
    top20 = station_agg.nlargest(20, "pct_imputed").sort_values("pct_imputed")

    real_pct = (1 - top20["pct_imputed"]) * 100
    imputed_pct = top20["pct_imputed"] * 100

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top20["station_id"], real_pct,    color="#4c72b0", label="Medición real")
    ax.barh(top20["station_id"], imputed_pct, left=real_pct, color="#d62728", alpha=0.8, label="Imputado (mediana/estación)")
    ax.set_xlabel("% de registros")
    ax.set_title("P5 — Q15: Proporción de registros imputados vs reales (top 20)")
    ax.axvline(100, color="black", linewidth=0.5)
    ax.set_xlim(0, 105)
    ax.legend()
    save_fig(fig, out / "Q15_imputed_vs_real.png", "15")

# CLI
def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    proc = project_root / "datos_procesados"

    parser = argparse.ArgumentParser(
        description="Análisis de calidad de datos de radiación — 15 queries"
    )
    parser.add_argument(
        "--radiation",
        default=str(proc / "radiation_clean.csv"),
    )
    parser.add_argument(
        "--plants",
        default=str(proc / "nuclear_plants_clean.csv"),
    )
    parser.add_argument(
        "--distances",
        default=str(proc / "station_plant_distances.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(proc / "reportes" / "figuras"),
    )
    return parser.parse_args()


# Main
def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rad, plants, dist, station_agg, closest = load_data(args.radiation, args.plants, args.distances)
    log.info("Generando visualizaciones")

    # P1 — Dosis vs distancia
    q01_distribucion_por_banda(station_agg, out)
    q02_boxplot_por_banda(station_agg, out)
    q03_scatter_distancia_gamma(station_agg, out)

    # P2 — Anomalías
    q04_serie_temporal_outliers(rad, station_agg, out)
    q05_top10_estaciones_outliers(station_agg, out)
    q06_heatmap_outliers_anio(rad, station_agg, out)

    # P3 — Tipo/capacidad reactor
    q07_gamma_por_reactor_type(station_agg, out)
    q08_scatter_capacity_gamma(station_agg, out)
    q09_heatmap_correlacion(station_agg, out)

    # P4 — Predicción
    q10_descomposicion_serie(rad, station_agg, out)
    q11_prediccion_media_movil(rad, station_agg, out)
    q12_error_vs_distancia(rad, station_agg, out)

    # P5 — Calidad por estación
    q13_ranking_calidad(station_agg, out)
    q14_nulos_dose_por_estacion(station_agg, out)
    q15_imputed_vs_real(station_agg, out)

    log.info("Análisis completado. 15 figuras guardadas en: %s", out)

if __name__ == "__main__":

    main()