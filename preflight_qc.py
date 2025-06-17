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
from typing import List, Tuple, Dict, Optional
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
pd.options.mode.use_inf_as_na = True  # compatibilidad
import scipy.io as sio
from nilearn import image as nli_image
from nilearn import plotting as nli_plotting
from scipy.signal import welch
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Funciones de Validación de Atlas ---

def check_atlas_integrity(aal3_path: Path, output_dir: Path) -> Dict:
    """
    Verifica la integridad y alineación entre los atlas AAL3 y Yeo-17.
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
        logger.error(f"Atlas AAL3 no encontrado en: {aal3_path}. No se puede continuar con la validación de atlas.")
        return report
    report["aal3_exists"] = True

    try:
        from nilearn.datasets import fetch_atlas_yeo_2011
        logger.info("Descargando/cargando atlas Yeo-17...")
        yeo_atlas = fetch_atlas_yeo_2011()
        yeo_img = nib.load(yeo_atlas.thick_17)
        report["yeo_fetched"] = True
    except Exception as e:
        logger.error(f"Fallo al descargar/cargar el atlas Yeo-17: {e}")
        return report

    aal_img = nib.load(aal3_path)
    logger.info(f"AAL3 Shape: {aal_img.shape}, Affine: \n{np.round(aal_img.affine[:3, :], 2)}")
    logger.info(f"Yeo-17 Shape: {yeo_img.shape}, Affine: \n{np.round(yeo_img.affine[:3, :], 2)}")

    # Comprobar si la geometría coincide
    affine_match = np.allclose(aal_img.affine, yeo_img.affine, atol=1e-3)
    shape_match = (aal_img.shape == yeo_img.shape)
    report["affine_mismatch"] = not affine_match
    report["shape_mismatch"] = not shape_match

    if affine_match and shape_match:
        logger.info("Geometría de atlas (affine y shape) coinciden. No se requiere remuestreo.")
        report["atlases_match_geometry"] = True
        aal_data = aal_img.get_fdata()
    else:
        logger.warning("La geometría de los atlas no coincide. Se recomienda verificar que `feature_extraction.py` realice el remuestreo correctamente.")
        report["atlases_match_geometry"] = False
        try:
            aal_resampled = nli_image.resample_to_img(aal_img, yeo_img, interpolation='nearest')
            aal_data = aal_resampled.get_fdata()
        except Exception as e:
            logger.error(f"Fallo al intentar remuestrear AAL3 al espacio de Yeo para la validación: {e}")
            return report

    # Calcular solapamiento de máscaras cerebrales (Dice Score)
    aal_mask = aal_data > 0
    yeo_mask = yeo_img.get_fdata() > 0
    intersection = np.sum(aal_mask & yeo_mask)
    dice = (2. * intersection) / (np.sum(aal_mask) + np.sum(yeo_mask))
    report["dice_score"] = dice
    logger.info(f"Coeficiente de Dice entre máscaras cerebrales AAL3 y Yeo-17: {dice:.4f}")

    # Visualización de una ROI para confirmación manual
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        # Tomamos una ROI del centro del cerebro como ejemplo (e.g., Tálamo, color 81 en AAL3v1)
        # Como los índices 81 y 82 se eliminan, usemos una región grande como el Precentral_L (color 1)
        roi_color_to_plot = 1
        roi_mask_img = nli_image.math_img(f"img == {roi_color_to_plot}", img=aal_img)

        display = nli_plotting.plot_roi(roi_mask_img, bg_img=yeo_img, title="Chequeo Visual: ROI 1 (AAL3) sobre Atlas Yeo-17",
                                        axes=ax, cut_coords=(0, 0, 0), display_mode='ortho')
        
        plot_path = output_dir / "visual_atlas_alignment_check.png"
        display.savefig(plot_path)
        plt.close(fig)
        report["visual_check_plot_path"] = str(plot_path)
        logger.info(f"Guardado chequeo visual de alineación de atlas en: {plot_path}")
    except Exception as e:
        logger.error(f"Fallo al generar el chequeo visual del atlas: {e}")

    return report


# --- Funciones de Inspección de Series Temporales ---

def inspect_single_subject_timeseries(mat_path: Path, expected_cols: int) -> Dict:
    """
    Inspecciona un archivo .mat de un sujeto para chequear orientación, NaNs, etc.
    """
    subject_report = {
        "subject_id": mat_path.stem.split('_')[-1],
        "file_found": False,
        "n_timepoints": 0,
        "n_rois_raw": 0,
        "is_transposed": None,
        "nan_count": 0,
        "inf_count": 0,
        "constant_rois_count": 0,
        "low_variance_rois_indices": [],
        "error": None
    }

    if not mat_path.exists():
        subject_report["error"] = "File not found"
        return subject_report
    subject_report["file_found"] = True

    try:
        # Claves posibles extraídas de feature_extraction.py
        possible_keys = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "AAL3_signals", "roi_ts"]
        data = sio.loadmat(mat_path)
        raw_sigs = None
        for key in possible_keys:
            if key in data:
                raw_sigs = data[key]
                break
        
        if raw_sigs is None:
            subject_report["error"] = f"No valid signal key found. Keys: {list(data.keys())}"
            return subject_report

        # 1. Orientación y conteo de columnas
        if raw_sigs.shape[0] == expected_cols and raw_sigs.shape[1] != expected_cols:
            subject_report["is_transposed"] = True
            raw_sigs = raw_sigs.T
        elif raw_sigs.shape[1] == expected_cols:
            subject_report["is_transposed"] = False
        else:
            subject_report["error"] = f"Neither dimension matches expected ROIs ({expected_cols}). Shape is {raw_sigs.shape}."
            subject_report["n_timepoints"] = raw_sigs.shape[0]
            subject_report["n_rois_raw"] = raw_sigs.shape[1]
            return subject_report

        subject_report["n_timepoints"] = raw_sigs.shape[0]
        subject_report["n_rois_raw"] = raw_sigs.shape[1]

        # 2. NaNs / Infs
        subject_report["nan_count"] = np.isnan(raw_sigs).sum()
        subject_report["inf_count"] = np.isinf(raw_sigs).sum()

        # Rellenar para los siguientes cálculos
        if subject_report["nan_count"] > 0 or subject_report["inf_count"] > 0:
            raw_sigs = np.nan_to_num(raw_sigs)

        # 3. Constantes / varianza casi nula
        stds = np.std(raw_sigs, axis=0)
        low_variance_mask = stds < 1e-6
        subject_report["constant_rois_count"] = np.sum(low_variance_mask)
        subject_report["low_variance_rois_indices"] = np.where(low_variance_mask)[0].tolist()
        
        # Guardar la serie para análisis espectral y de estacionariedad
        subject_report["timeseries_data"] = raw_sigs

    except Exception as e:
        subject_report["error"] = str(e)

    return subject_report


# --- Funciones de Validación Espectral y Estacionariedad ---

def validate_spectral_and_stationarity(subject_report: Dict, tr: float, output_dir: Path) -> Dict:
    """
    Realiza análisis de PSD y estacionariedad para un sujeto.
    """
    if "timeseries_data" not in subject_report or subject_report["timeseries_data"] is None:
        subject_report["psd_plot_path"] = None
        subject_report["percent_stationary_adf"] = 0.0
        subject_report["percent_stationary_kpss"] = 0.0
        subject_report["percent_stationary_joint"] = 0.0
        return subject_report

    ts_data = subject_report["timeseries_data"]
    fs = 1.0 / tr
    n_rois = ts_data.shape[1]
    
    # 1. PSD con Welch
    try:
        # Seleccionar hasta 5 ROIs aleatorios para graficar
        n_rois_to_plot = min(5, n_rois)
        plot_indices = np.random.choice(n_rois, n_rois_to_plot, replace=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in plot_indices:
            f, Pxx = welch(ts_data[:, i], fs=fs, nperseg=min(256, len(ts_data[:,i])-1))
            Pxx[Pxx <= 0] = np.finfo(float).eps  # fuerza positividad
            ax.semilogy(f, Pxx, label=f'ROI {i}')
        
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('PSD (V**2/Hz)')
        ax.set_title(f'Densidad Espectral de Potencia (Welch) para Sujeto {subject_report["subject_id"]}')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, fs / 2)
        
        psd_plot_path = output_dir / f'psd_{subject_report["subject_id"]}.png'
        fig.savefig(psd_plot_path)
        plt.close(fig)
        subject_report["psd_plot_path"] = str(psd_plot_path)
    except Exception as e:
        logger.error(f'Sujeto {subject_report["subject_id"]}: Fallo al calcular/graficar PSD: {e}')
        subject_report["psd_plot_path"] = None

    # 2. Estacionariedad (ADF y KPSS)
    adf_stationary_rois = 0
    kpss_stationary_rois = 0
    joint_stationary_rois = 0
    
    for i in range(n_rois):
        roi_ts = ts_data[:, i]
        if np.std(roi_ts) < 1e-6:  # Omitir ROIs constantes
            continue

        try:
            # ADF Test: H0 = tiene raíz unitaria (no estacionaria)
            adf_result = adfuller(roi_ts)
            is_adf_stationary = adf_result[1] < 0.05
            if is_adf_stationary:
                adf_stationary_rois += 1
            
            # KPSS Test: H0 = es estacionaria alrededor de una tendencia
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                kpss_result = kpss(roi_ts, regression='c', nlags="auto")
            is_kpss_stationary = kpss_result[1] > 0.05
            if is_kpss_stationary:
                kpss_stationary_rois += 1

            if is_adf_stationary and is_kpss_stationary:
                joint_stationary_rois += 1

        except Exception as e:
            logger.warning(f'Sujeto {subject_report["subject_id"]}, ROI {i}: Fallo test de estacionariedad: {e}')
            
    subject_report["percent_stationary_adf"] = (adf_stationary_rois / n_rois) * 100 if n_rois > 0 else 0
    subject_report["percent_stationary_kpss"] = (kpss_stationary_rois / n_rois) * 100 if n_rois > 0 else 0
    subject_report["percent_stationary_joint"] = (joint_stationary_rois / n_rois) * 100 if n_rois > 0 else 0

    del subject_report["timeseries_data"] # Liberar memoria
    return subject_report


# --- Función de Generación de Informe ---

def generate_html_report(df: pd.DataFrame, atlas_report: Dict, output_dir: Path):
    """
    Genera un informe HTML con los resultados del QC.
    """
    logger.info("--- Generando Informe HTML de Pre-Flight QC ---")
    
    # Estilo del HTML
    html = f"""
    <html>
    <head>
        <title>Pre-Flight QC Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 80%; margin: 1em 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }}
            .plot-container {{ border: 1px solid #ccc; padding: 10px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Informe de Control de Calidad Previo (Pre-Flight QC)</h1>
        <p>Este informe resume los chequeos automáticos realizados sobre el dataset antes de la extracción de características.</p>
    """

    # Sección 1: Integridad del Atlas
    html += "<h2>1. Integridad y Alineación de Atlas</h2>"
    if atlas_report:
        html += f"""
        <ul>
            <li>Geometría Coincidente (sin remuestreo): <b>{atlas_report.get('atlases_match_geometry', 'N/A')}</b></li>
            <li>Dice Score (solapamiento de máscaras): <b>{atlas_report.get('dice_score', 'N/A'):.4f}</b></li>
        </ul>
        <div class="plot-container">
            <h3>Chequeo Visual de Alineación</h3>
            <p>ROI 1 del atlas AAL3 superpuesto al atlas Yeo-17. Verificar que la ROI caiga en una región cortical plausible.</p>
            <img src="{Path(atlas_report.get('visual_check_plot_path')).name}" alt="Visual Atlas Alignment Check">
        </div>
        """
    else:
        html += "<p>No se pudo generar el informe del atlas.</p>"

    # Sección 2: Resumen del Dataset
    html += "<h2>2. Resumen de las Series Temporales</h2>"
    html += "<h3>Primeros 10 Sujetos del Reporte</h3>"
    html += df.head(10).to_html(classes='table table-striped', index=False, na_rep='N/A')
    
    # Generar y guardar gráficos de resumen
    plots = {}
    try:
        # Gráfico de conteo de Timepoints (mejor para valores discretos)
        fig, ax = plt.subplots(figsize=(10, 6))
        order = [140, 197, 200]          # orden natural
        sns.countplot(x='n_timepoints', data=df, ax=ax,
                      palette='viridis', order=order)
        ax.set_title('Conteo de Sujetos por Número de Timepoints')
        ax.set_xlabel('Número de Timepoints')
        ax.set_ylabel('Cantidad de Sujetos')
        # Añadir etiquetas de conteo sobre las barras
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        plt.tight_layout()
        plots['tp_countplot_path'] = output_dir / "summary_tp_countplot.png"
        fig.savefig(plots['tp_countplot_path']); plt.close(fig)

        # Boxplot de % ROIs Estacionarios
        fig, ax = plt.subplots()
        sns.boxplot(data=df[['percent_stationary_adf', 'percent_stationary_kpss', 'percent_stationary_joint']], ax=ax)
        ax.set_title('% de ROIs Estacionarios por Sujeto')
        ax.set_ylabel('% Estacionario')
        plots['stationarity_boxplot_path'] = output_dir / "summary_stationarity_boxplot.png"
        fig.savefig(plots['stationarity_boxplot_path']); plt.close(fig)

        # Conteo de ROIs con Varianza Nula
        fig, ax = plt.subplots()
        sns.histplot(df['constant_rois_count'], ax=ax, discrete=True)
        ax.set_title('Distribución de ROIs con Varianza Nula por Sujeto')
        plots['low_var_hist_path'] = output_dir / "summary_low_variance_histogram.png"
        fig.savefig(plots['low_var_hist_path']); plt.close(fig)
        
        # Conteo de Sujetos con NaNs
        html_nan_summary = f"<p>Total de sujetos con valores NaN en sus series: <b>{df[df['nan_count'] > 0].shape[0]}</b></p>"
        html_nan_summary += f"<p>Total de sujetos con valores Inf en sus series: <b>{df[df['inf_count'] > 0].shape[0]}</b></p>"


    except Exception as e:
        logger.error(f"Fallo al generar gráficos de resumen para el informe HTML: {e}")

    html += f"""
    <h3>Visualizaciones del Dataset Completo</h3>
    <div class="summary-grid">
        <div class="plot-container">
            <h4>Distribución de Timepoints</h4>
            <img src="{plots.get('tp_countplot_path', '').name}" alt="Timepoints Count Plot">
        </div>
        <div class="plot-container">
            <h4>Estacionariedad de ROIs</h4>
            <img src="{plots.get('stationarity_boxplot_path', '').name}" alt="Stationarity Boxplot">
        </div>
        <div class="plot-container">
            <h4>ROIs con Varianza Nula</h4>
            <img src="{plots.get('low_var_hist_path', '').name}" alt="Low Variance Histogram">
        </div>
        <div class="plot-container">
            <h4>Resumen de NaNs/Infs</h4>
            {html_nan_summary}
        </div>
    </div>
    """

    # Sección 3: Ejemplos de Sujetos
    html += "<h2>3. Ejemplos de PSD de Sujetos Aleatorios</h2>"
    html += "<div class='summary-grid'>"
    sample_subjects = df.dropna(subset=['psd_plot_path']).sample(min(4, df.dropna(subset=['psd_plot_path']).shape[0]))
    for _, row in sample_subjects.iterrows():
        html += f"""
        <div class="plot-container">
            <h4>Sujeto: {row['subject_id']}</h4>
            <img src="{Path(row['psd_plot_path']).name}" alt="PSD Plot for {row['subject_id']}">
        </div>
        """
    html += "</div>"
    
    html += "</body></html>"

    report_path = output_dir / "preflight_QC_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Informe HTML guardado en: {report_path}")


# --- Flujo Principal de Ejecución ---

def main(args):
    """
    Función principal que orquesta el pipeline de QC.
    """
    script_start_time = pd.Timestamp.now()
    logger.info("Iniciando Pre-Flight QC Script...")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- PASO 1: Validar Atlas ---
    atlas_report = check_atlas_integrity(Path(args.aal3_nifti_path), output_dir)
    
    # --- PASO 2: Cargar metadatos de sujetos ---
    # Usaremos una lista de archivos .mat como proxy de la lista de sujetos
    roi_signals_dir = Path(args.roi_signals_dir)
    if not roi_signals_dir.is_dir():
        logger.critical(f"El directorio de señales de ROI no existe: {roi_signals_dir}. Abortando.")
        return
        
    subject_mat_files = sorted(list(roi_signals_dir.glob(args.roi_filename_template)))
    if not subject_mat_files:
        logger.critical(f"No se encontraron archivos de señales con el patrón '{args.roi_filename_template}' en {roi_signals_dir}. Abortando.")
        return
    
    logger.info(f"Encontrados {len(subject_mat_files)} archivos de sujetos para procesar.")

    # --- PASO 3: Procesar cada sujeto ---
    all_subject_reports = []
    for mat_path in tqdm(subject_mat_files, desc="Inspeccionando Sujetos"):
        # Inspección básica del archivo .mat
        subject_report = inspect_single_subject_timeseries(mat_path, args.expected_rois_raw)
        
        # Análisis espectral y de estacionariedad
        subject_report = validate_spectral_and_stationarity(subject_report, args.tr, output_dir)
        
        all_subject_reports.append(subject_report)

    # --- PASO 4: Consolidar y guardar resultados ---
    # Convertir lista de dicts a DataFrame y limpiar
    qc_df = pd.DataFrame(all_subject_reports)
    if 'timeseries_data' in qc_df.columns:
        qc_df = qc_df.drop(columns=['timeseries_data'])

    # Guardar el informe CSV detallado
    csv_path = output_dir / "subjects_overview_preflight_QC.csv"
    qc_df.to_csv(csv_path, index=False)
    logger.info(f"Informe CSV de Pre-Flight QC guardado en: {csv_path}")

    # --- PASO 5: Generar informe HTML ---
    generate_html_report(qc_df, atlas_report, output_dir)

    logger.info(f"Pre-Flight QC completado en {(pd.Timestamp.now() - script_start_time).total_seconds():.2f} segundos.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-Flight Quality Control Script for fMRI Connectivity Pipeline.")
    
    # --- Argumentos basados en la configuración de feature_extraction.py ---
    # Es crucial que estos argumentos coincidan con tu setup.
    parser.add_argument('--base_path', type=str, default='/home/diego/Escritorio/AAL3',
                        help="Ruta base del proyecto.")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directorio para guardar los informes de QC. Por defecto: [base_path]/preflight_qc_report")

    # Argumentos sobre los datos
    parser.add_argument('--roi_signals_dir', type=str, default=None,
                        help="Directorio que contiene los archivos ROISignals_*.mat. Por defecto: [base_path]/ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm")
    parser.add_argument('--roi_filename_template', type=str, default='ROISignals_*.mat',
                        help="Patrón de nombre de archivo para los datos de los sujetos.")
    parser.add_argument('--aal3_nifti_path', type=str, default=None,
                        help="Ruta al archivo NIfTI del atlas AAL3. Por defecto: [base_path]/AAL3v1.nii.gz")
    
    # Parámetros del experimento
    parser.add_argument('--tr', type=float, default=3.0, help="Tiempo de Repetición (TR) en segundos.")
    parser.add_argument('--expected_rois_raw', type=int, default=170,
                        help="Número de columnas/ROIs esperado en los archivos .mat crudos.")

    args = parser.parse_args()
    
    # Configurar rutas por defecto si no se proporcionan
    base_path = Path(args.base_path)
    if args.output_dir is None:
        args.output_dir = base_path / "preflight_qc_report"
    if args.roi_signals_dir is None:
        args.roi_signals_dir = base_path / 'ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm'
    if args.aal3_nifti_path is None:
        args.aal3_nifti_path = base_path / "AAL3v1.nii.gz"

    main(args)