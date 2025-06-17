#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Pre-Flight Quality Control (QC) Script for fMRI Time Series Analysis
================================================================================
Version: 1.1.0
Author: [Tu Nombre/Laboratorio] - Adaptado por Gemini

Objetivo:
Este script realiza una serie de chequeos y análisis previos imprescindibles
antes de ejecutar el pipeline principal de extracción de características
(e.g., feature_extraction.py). Su propósito es garantizar que cada serie
temporal BOLD y cada ROI cumplen con las suposiciones fundamentales del
pipeline (tamaño, orientación, longitud, calidad, alineación anatómica,
estacionariedad, etc.).

Esto previene errores silenciosos, matrices mal formadas o data-leakage
estadístico en pasos posteriores como el filtrado, reordenamiento,
causalidad de Granger, conectividad dinámica (dFC), etc.

Genera dos salidas principales:
1.  subjects_overview_preflight_QC.csv: Un archivo CSV con un resumen de las
    métricas de calidad para cada sujeto.
2.  preflight_QC_report.html: Un informe HTML con visualizaciones para una
    inspección rápida del dataset completo.

Uso:
python preflight_qc.py --base_path /ruta/a/tus/datos --output_dir /ruta/para/salidas
"""
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import warnings
import nibabel as nib
import numpy as np
import pandas as pd
pd.options.mode.use_inf_as_na = True  # Compatibilidad con versiones anteriores de pandas
import scipy.io as sio
from nilearn import image as nli_image
from nilearn import plotting as nli_plotting
from nilearn.datasets import fetch_atlas_yeo_2011
from scipy.signal import welch
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración del Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Funciones de Validación de Atlas ---

def check_atlas_integrity(aal3_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Verifica la integridad y alineación entre los atlas AAL3 y Yeo-17.

    Args:
        aal3_path (Path): Ruta al archivo NIfTI del atlas AAL3.
        output_dir (Path): Directorio para guardar el gráfico de chequeo visual.

    Returns:
        Dict[str, Any]: Un diccionario con el reporte de la validación.
    """
    logger.info("--- 1. Verificando Integridad y Alineación de Atlas ---")
    report = {
        "aal3_exists": False,
        "yeo_fetched": False,
        "atlases_match_geometry": None,
        "affine_mismatch": None,
        "shape_mismatch": None,
        "dice_score": None,
        "visual_check_plot_path": None
    }

    if not aal3_path.exists():
        logger.error(f"Atlas AAL3 no encontrado en: {aal3_path}. No se puede continuar.")
        return report
    report["aal3_exists"] = True

    try:
        logger.info("Descargando/cargando atlas Yeo-17...")
        yeo_atlas = fetch_atlas_yeo_2011()
        yeo_img = nib.load(yeo_atlas.thick_17)
        report["yeo_fetched"] = True
    except Exception as e:
        logger.error(f"Fallo al descargar/cargar el atlas Yeo-17: {e}", exc_info=True)
        return report

    aal_img = nib.load(aal3_path)
    logger.info(f"AAL3 Shape: {aal_img.shape}, Affine: \n{np.round(aal_img.affine[:3, :], 2)}")
    logger.info(f"Yeo-17 Shape: {yeo_img.shape}, Affine: \n{np.round(yeo_img.affine[:3, :], 2)}")

    # Comprobar si la geometría coincide
    affine_match = np.allclose(aal_img.affine, yeo_img.affine, atol=1e-3)
    shape_match = aal_img.shape == yeo_img.shape
    report["affine_mismatch"] = not affine_match
    report["shape_mismatch"] = not shape_match
    report["atlases_match_geometry"] = affine_match and shape_match

    if report["atlases_match_geometry"]:
        logger.info("Geometría de atlas (affine y shape) coinciden. No se requiere remuestreo.")
        aal_data_for_comparison = aal_img.get_fdata()
        bg_img_for_plot = aal_img # Usar AAL3 como fondo si coinciden
    else:
        logger.warning("La geometría no coincide. Se remuestreará AAL3 a Yeo solo para comparación.")
        try:
            aal_resampled = nli_image.resample_to_img(aal_img, yeo_img, interpolation='nearest')
            aal_data_for_comparison = aal_resampled.get_fdata()
            bg_img_for_plot = yeo_img # Usar Yeo como fondo para ver la alineación
        except Exception as e:
            logger.error(f"Fallo al remuestrear AAL3: {e}", exc_info=True)
            return report

    # Calcular solapamiento de máscaras cerebrales (Dice Score)
    aal_mask = aal_data_for_comparison > 0
    yeo_mask = yeo_img.get_fdata() > 0
    intersection = np.sum(aal_mask & yeo_mask)
    dice = (2. * intersection) / (np.sum(aal_mask) + np.sum(yeo_mask))
    report["dice_score"] = dice
    logger.info(f"Coeficiente de Dice entre máscaras AAL3 y Yeo-17: {dice:.4f}")

    # Visualización de una ROI para confirmación manual
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        roi_color_to_plot = 1  # Precentral_L, una ROI grande y robusta
        roi_mask_img = nli_image.math_img(f"img == {roi_color_to_plot}", img=aal_img)
        
        display = nli_plotting.plot_roi(
            roi_mask_img, bg_img=bg_img_for_plot,
            title=f"Chequeo Visual: AAL3 ROI {roi_color_to_plot} sobre Atlas de Fondo",
            axes=ax, cut_coords=(0, 0, 0), display_mode='ortho', colorbar=True
        )
        plot_path = output_dir / "visual_atlas_alignment_check.png"
        display.savefig(plot_path)
        plt.close(fig)
        report["visual_check_plot_path"] = str(plot_path)
        logger.info(f"Guardado chequeo visual de alineación de atlas en: {plot_path}")
    except Exception as e:
        logger.error(f"Fallo al generar el chequeo visual del atlas: {e}", exc_info=True)
    
    return report


# --- Funciones de Inspección de Series Temporales ---

def inspect_single_subject_timeseries(mat_path: Path, expected_cols: int) -> Dict[str, Any]:
    """
    Inspecciona un archivo .mat de un sujeto para chequear formato, NaNs, etc.

    Args:
        mat_path (Path): Ruta al archivo .mat del sujeto.
        expected_cols (int): Número de ROIs/columnas esperadas.

    Returns:
        Dict[str, Any]: Un diccionario con el reporte de la inspección.
    """
    subject_id = mat_path.stem.split('_')[-1]
    subject_report = {
        "subject_id": subject_id, "file_found": True, "n_timepoints": 0,
        "n_rois_raw": 0, "is_transposed": None, "nan_count": 0, "inf_count": 0,
        "constant_rois_count": 0, "low_variance_rois_indices": [], "error": None
    }

    try:
        data = sio.loadmat(mat_path)
        possible_keys = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "roi_ts"]
        raw_sigs = None
        for key in possible_keys:
            if key in data:
                raw_sigs = data[key]
                break
        
        if raw_sigs is None:
            subject_report["error"] = f"No signal key found. Keys: {list(data.keys())}"
            return subject_report

        if raw_sigs.shape[0] == expected_cols and raw_sigs.shape[1] != expected_cols:
            subject_report["is_transposed"] = True
            raw_sigs = raw_sigs.T
        elif raw_sigs.shape[1] == expected_cols:
            subject_report["is_transposed"] = False
        else:
            subject_report["error"] = f"Shape {raw_sigs.shape} mismatch with expected {expected_cols} ROIs."
            subject_report["n_timepoints"] = raw_sigs.shape[0]
            subject_report["n_rois_raw"] = raw_sigs.shape[1]
            return subject_report

        subject_report["n_timepoints"] = raw_sigs.shape[0]
        subject_report["n_rois_raw"] = raw_sigs.shape[1]
        
        subject_report["nan_count"] = np.isnan(raw_sigs).sum()
        subject_report["inf_count"] = np.isinf(raw_sigs).sum()

        clean_sigs = np.nan_to_num(raw_sigs, nan=0.0, posinf=0.0, neginf=0.0)
        
        stds = np.std(clean_sigs, axis=0)
        low_variance_mask = stds < 1e-6
        subject_report["constant_rois_count"] = np.sum(low_variance_mask)
        subject_report["low_variance_rois_indices"] = np.where(low_variance_mask)[0].tolist()
        
        subject_report["timeseries_data"] = clean_sigs

    except Exception as e:
        subject_report["error"] = str(e)
    
    return subject_report


# --- Funciones de Validación Espectral y Estacionariedad ---

def validate_spectral_and_stationarity(subject_report: Dict[str, Any], tr: float, output_dir: Path) -> Dict[str, Any]:
    """
    Realiza análisis de PSD y estacionariedad para un sujeto.

    Args:
        subject_report (Dict[str, Any]): Reporte del sujeto, debe contener 'timeseries_data'.
        tr (float): Tiempo de repetición.
        output_dir (Path): Directorio para guardar el gráfico de PSD.

    Returns:
        Dict[str, Any]: El reporte del sujeto actualizado.
    """
    if subject_report.get("timeseries_data") is None or subject_report.get("error"):
        for key in ["psd_plot_path", "percent_stationary_adf", "percent_stationary_kpss", "percent_stationary_joint"]:
            subject_report[key] = None if key.endswith("path") else 0.0
        return subject_report

    ts_data = subject_report["timeseries_data"]
    fs = 1.0 / tr
    n_rois = ts_data.shape[1]
    
    # 1. PSD con Welch
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        n_rois_to_plot = min(5, n_rois)
        plot_indices = np.random.choice(n_rois, n_rois_to_plot, replace=False)
        
        for i in plot_indices:
            f, Pxx = welch(ts_data[:, i], fs=fs, nperseg=min(256, ts_data.shape[0]))
            ax.semilogy(f, Pxx, label=f'ROI {i}', alpha=0.8)
            
        ax.set_xlabel('Frecuencia (Hz)'); ax.set_ylabel('PSD (V**2/Hz)')
        ax.set_title(f'Densidad Espectral de Potencia (Welch) - Sujeto {subject_report["subject_id"]}')
        ax.legend(); ax.grid(True); ax.set_xlim(0, fs / 2)
        
        psd_plot_path = output_dir / f'psd_{subject_report["subject_id"]}.png'
        fig.savefig(psd_plot_path); plt.close(fig)
        subject_report["psd_plot_path"] = str(psd_plot_path)
    except Exception as e:
        logger.error(f'Sujeto {subject_report["subject_id"]}: Fallo al calcular PSD: {e}')
        subject_report["psd_plot_path"] = None

    # 2. Estacionariedad
    adf_stationary, kpss_stationary, joint_stationary = 0, 0, 0
    valid_rois_for_test = 0
    for i in range(n_rois):
        roi_ts = ts_data[:, i]
        if np.std(roi_ts) < 1e-6: continue
        valid_rois_for_test += 1
        
        try:
            is_adf_stationary = adfuller(roi_ts)[1] < 0.05
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                is_kpss_stationary = kpss(roi_ts, regression='c', nlags="auto")[1] > 0.05
            
            if is_adf_stationary: adf_stationary += 1
            if is_kpss_stationary: kpss_stationary += 1
            if is_adf_stationary and is_kpss_stationary: joint_stationary += 1
        except Exception as e:
            logger.warning(f'Sujeto {subject_report["subject_id"]}, ROI {i}: Fallo test est.: {e}')

    n = valid_rois_for_test
    subject_report["percent_stationary_adf"] = (adf_stationary / n) * 100 if n > 0 else 0
    subject_report["percent_stationary_kpss"] = (kpss_stationary / n) * 100 if n > 0 else 0
    subject_report["percent_stationary_joint"] = (joint_stationary / n) * 100 if n > 0 else 0
    
    del subject_report["timeseries_data"] # Liberar memoria
    return subject_report


# --- Función de Generación de Informe ---

def generate_html_report(df: pd.DataFrame, atlas_report: Dict[str, Any], output_dir: Path):
    """
    Genera un informe HTML con los resultados del QC.

    Args:
        df (pd.DataFrame): DataFrame con los reportes de todos los sujetos.
        atlas_report (Dict[str, Any]): Reporte de la validación de atlas.
        output_dir (Path): Directorio donde se guardan las salidas.
    """
    logger.info("--- Generando Informe HTML de Pre-Flight QC ---")
    
    plots = {}
    sns.set_theme(style="whitegrid")
    
    # Generar gráficos de resumen
    try:
        # Gráfico 1: Distribución de Timepoints
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='n_timepoints', data=df, ax=ax, palette='viridis')
        ax.set_title('Distribución de Sujetos por Número de Timepoints')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        plots['tp_countplot_path'] = output_dir / "summary_tp_countplot.png"
        fig.savefig(plots['tp_countplot_path']); plt.close(fig)

        # Gráfico 2: Estacionariedad
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[['percent_stationary_adf', 'percent_stationary_kpss', 'percent_stationary_joint']], ax=ax)
        ax.set_title('% de ROIs Estacionarios por Sujeto (Tests ADF y KPSS)')
        ax.set_ylabel('% Estacionario'); ax.set_xticklabels(['ADF (<0.05)', 'KPSS (>0.05)', 'Ambos'])
        plots['stationarity_boxplot_path'] = output_dir / "summary_stationarity_boxplot.png"
        fig.savefig(plots['stationarity_boxplot_path']); plt.close(fig)

        # Gráfico 3: ROIs con Varianza Nula
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['constant_rois_count'], ax=ax, discrete=True, kde=False)
        ax.set_title('Distribución de ROIs con Varianza Nula por Sujeto'); ax.set_xlabel('Número de ROIs Constantes')
        plots['low_var_hist_path'] = output_dir / "summary_low_variance_histogram.png"
        fig.savefig(plots['low_var_hist_path']); plt.close(fig)

    except Exception as e:
        logger.error(f"Fallo al generar gráficos de resumen: {e}", exc_info=True)

    # Construir HTML
    html = """
    <html><head><title>Pre-Flight QC Report</title><style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }
        h1, h2, h3 { color: #0056b3; border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 1em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #dee2e6; padding: 12px; text-align: left; }
        th { background-color: #e9ecef; } tr:nth-child(even) { background-color: #f8f9fa; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
        .plot-container { background-color: #fff; border: 1px solid #dee2e6; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        img { max-width: 100%; height: auto; border-radius: 5px; } .fail {color: red; font-weight: bold;} .pass {color: green; font-weight: bold;}
    </style></head><body>
    <h1>Informe de Control de Calidad Previo (Pre-Flight QC)</h1>
    <p>Fecha del informe: """ f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"

    # Sección Atlas
    html += "<h2>1. Integridad y Alineación de Atlas</h2><div class='plot-container'>"
    geo_match_text = f"<span class='pass'>Sí</span>" if atlas_report.get('atlases_match_geometry') else f"<span class='fail'>No</span>"
    dice_score = atlas_report.get('dice_score', 0)
    dice_text = f"<span class='pass'>{dice_score:.4f}</span>" if dice_score > 0.7 else f"<span class='fail'>{dice_score:.4f}</span>"
    html += f"""<ul>
        <li>Geometría Coincidente (sin remuestreo): <b>{geo_match_text}</b></li>
        <li>Dice Score (solapamiento > 0.7 es bueno): <b>{dice_text}</b></li></ul>
        <img src="{Path(atlas_report.get('visual_check_plot_path')).name}" alt="Visual Atlas Alignment Check"></div>"""

    # Sección Resumen Dataset
    html += "<h2>2. Resumen General del Dataset</h2>"
    html_nan_summary = f"<p>Sujetos con NaNs: <b class='{'fail' if df['nan_count'].sum() > 0 else 'pass'}'>{df[df['nan_count'] > 0].shape[0]}</b></p>"
    html_nan_summary += f"<p>Sujetos con Infs: <b class='{'fail' if df['inf_count'].sum() > 0 else 'pass'}'>{df[df['inf_count'] > 0].shape[0]}</b></p>"
    html += f"""<div class="summary-grid">
        <div class="plot-container"><img src="{plots.get('tp_countplot_path', '').name}" alt="Plot"></div>
        <div class="plot-container"><img src="{plots.get('stationarity_boxplot_path', '').name}" alt="Plot"></div>
        <div class="plot-container"><img src="{plots.get('low_var_hist_path', '').name}" alt="Plot"></div>
        <div class="plot-container"><h3>Resumen de Artefactos</h3>{html_nan_summary}</div>
    </div>"""

    # Sección Tabla
    html += "<h3>Reporte Detallado por Sujeto (Primeros 20)</h3>"
    html += df.head(20).to_html(classes='table', index=False, na_rep='N/A', float_format='%.2f')
    
    # Sección PSD
    html += "<h2>3. Ejemplos de PSD de Sujetos Aleatorios</h2><div class='summary-grid'>"
    sample_subjects = df.dropna(subset=['psd_plot_path']).sample(min(4, df.dropna(subset=['psd_plot_path']).shape[0]))
    for _, row in sample_subjects.iterrows():
        html += f"""<div class="plot-container"><h4>Sujeto: {row['subject_id']}</h4>
        <img src="{Path(row['psd_plot_path']).name}" alt="PSD Plot"></div>"""
    html += "</div></body></html>"

    report_path = output_dir / "preflight_QC_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Informe HTML guardado en: {report_path}")


# --- Flujo Principal de Ejecución ---

def main(args):
    """Función principal que orquesta el pipeline de QC."""
    script_start_time = pd.Timestamp.now()
    logger.info("Iniciando Pre-Flight QC Script...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_report = check_atlas_integrity(Path(args.aal3_nifti_path), output_dir)
    
    roi_signals_dir = Path(args.roi_signals_dir)
    if not roi_signals_dir.is_dir():
        logger.critical(f"Directorio de señales no existe: {roi_signals_dir}. Abortando.")
        return

    subject_mat_files = sorted(list(roi_signals_dir.glob(args.roi_filename_template)))
    if not subject_mat_files:
        logger.critical(f"No se encontraron archivos con '{args.roi_filename_template}' en {roi_signals_dir}. Abortando.")
        return
    logger.info(f"Encontrados {len(subject_mat_files)} archivos de sujetos para procesar.")

    all_reports = []
    for mat_path in tqdm(subject_mat_files, desc="Inspeccionando Sujetos"):
        report = inspect_single_subject_timeseries(mat_path, args.expected_rois_raw)
        report = validate_spectral_and_stationarity(report, args.tr, output_dir)
        all_reports.append(report)

    qc_df = pd.DataFrame(all_reports)
    if 'timeseries_data' in qc_df.columns:
        qc_df = qc_df.drop(columns=['timeseries_data'])

    csv_path = output_dir / "subjects_overview_preflight_QC.csv"
    qc_df.to_csv(csv_path, index=False, float_format='%.3f')
    logger.info(f"Informe CSV de Pre-Flight QC guardado en: {csv_path}")
    
    generate_html_report(qc_df, atlas_report, output_dir)
    
    duration = (pd.Timestamp.now() - script_start_time).total_seconds()
    logger.info(f"Pre-Flight QC completado en {duration:.2f} segundos.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-Flight Quality Control Script for fMRI Connectivity Pipeline.")

    parser.add_argument('--base_path', type=str, required=True, help="Ruta base del proyecto.")
    parser.add_argument('--output_dir', type=str, default=None, help="Directorio para guardar informes. Default: [base_path]/preflight_qc_report")
    parser.add_argument('--roi_signals_dir', type=str, default=None, help="Directorio con archivos .mat. Default: [base_path]/ROISignals_...")
    parser.add_argument('--roi_filename_template', type=str, default='ROISignals_*.mat', help="Patrón de nombre de archivo para datos de sujetos.")
    parser.add_argument('--aal3_nifti_path', type=str, default=None, help="Ruta al atlas AAL3 NIfTI. Default: [base_path]/AAL3v1.nii.gz")
    
    parser.add_argument('--tr', type=float, default=3.0, help="Tiempo de Repetición (TR) en segundos.")
    parser.add_argument('--expected_rois_raw', type=int, default=170, help="Número de ROIs esperado en los archivos .mat crudos.")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    if args.output_dir is None:
        args.output_dir = base_path / "preflight_qc_report"
    if args.roi_signals_dir is None:
        args.roi_signals_dir = base_path / 'ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm'
    if args.aal3_nifti_path is None:
        args.aal3_nifti_path = base_path / "AAL3v1.nii.gz"
        
    main(args)