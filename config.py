# -*- coding: utf-8 -*-
# config.py
"""
Archivo de Configuración para el Pipeline de Conectividad fMRI
Versión: v7.0.0_Refactored

Este archivo centraliza todos los parámetros, rutas y flags para facilitar
la experimentación y reproducibilidad sin modificar el código del pipeline.
"""
from pathlib import Path
from typing import List, Dict

# En config.py

# --- 1. Rutas Principales (Paths) ---
BASE_PATH = Path('/home/diego/Escritorio/desde_cero') 
ROI_SIGNALS_DIR = BASE_PATH / 'ROISignalsAAL3' 

# --- Archivos de Metadatos y Atlas ---
# Ruta al archivo CSV que CONTIENE ÚNICAMENTE los sujetos que han pasado el QC.
CLEANED_SUBJECT_LIST_PATH = BASE_PATH / 'SubjectsData_cleaned.csv'

# Rutas a los archivos de Atlas y plantillas
AAL3_META_PATH = BASE_PATH / 'ROI_MNI_V7_vol.txt' 
AAL3_NIFTI_PATH = BASE_PATH / "AAL3v1_1mm.nii.gz"
ROI_FILENAME_TEMPLATE = 'ROISignals_{subject_id}.mat'
#ROI_FILENAME_TEMPLATE = 'ROISignals_*.mat' # Usamos el comodín que definiste en el QC
# CORREGIDO: Usar el mismo atlas NIfTI que en el script de QC.
AAL3_META_PATH = BASE_PATH / 'ROI_MNI_V7_vol.txt'
AAL3_NIFTI_PATH = BASE_PATH / "AAL3v1_1mm.nii.gz"   # ¡IMPORTANTE! Asegúrate que está en el mismo espacio que Yeo (e.g., MNI 2mm)

# --- 2. Parámetros de Preprocesamiento de Series Temporales ---
# Parámetros del escáner y filtrado
TR_SECONDS = 3.0
LOW_CUT_HZ = 0.01
HIGH_CUT_HZ = 0.08
FILTER_ORDER = 2
TAPER_ALPHA = 0.1  # Parámetro para la ventana Tukey antes del filtrado

# Homogeneización de la longitud de las series temporales
TARGET_LEN_TS = 140 # Todas las series se truncarán o interpolarán a esta longitud

# Deconvolución HRF (Hemodynamic Response Function)
# NOTA: Mantener en False es la opción más segura si no hay una justificación
#       sólida o una estimación de HRF por sujeto.
APPLY_HRF_DECONVOLUTION = False
HRF_MODEL = 'glover'  # Opciones: 'glover', 'spm'

# --- 3. Parámetros de Reducción y Atlas AAL3 ---
# Estos parámetros deben coincidir con los usados en tu script de QC.
RAW_DATA_EXPECTED_COLUMNS = 170  # Número inicial de ROIs en los archivos .mat
AAL3_MISSING_INDICES_1BASED = [35, 36, 81, 82]  # ROIs sistemáticamente ausentes
SMALL_ROI_VOXEL_THRESHOLD = 100  # Umbral para descartar ROIs por volumen bajo

# Claves posibles para buscar las señales en los archivos .mat
POSSIBLE_ROI_KEYS = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "AAL3_signals", "roi_ts"]

# --- 4. Configuración de los Canales de Conectividad ---
# Usa estos flags para activar/desactivar el cálculo de cada modalidad.
# El orden en CONNECTIVITY_CHANNEL_NAMES definirá el orden en el tensor final.

# Canal 1: Pearson con OMST (o fallback a Pearson completo si dyconnmap falla)
USE_PEARSON_OMST_CHANNEL = True

# Canal 2: Pearson completo (sin podado)
# Se recomienda mantenerlo para comparar o si OMST no está disponible.
USE_PEARSON_FULL_SIGNED_CHANNEL = True

# Canal 3: Información Mutua (Mutual Information)
# Computacionalmente intensivo.
USE_MI_CHANNEL_FOR_THESIS = True
N_NEIGHBORS_MI = 5  # Parámetro 'k' para el estimador de MI basado en KNN

# Canales 4 & 5: Conectividad Dinámica (dFC)
# Se calculan a partir de ventanas deslizantes.
USE_DFC_ABS_DIFF_MEAN_CHANNEL = True
USE_DFC_STDDEV_CHANNEL = True
DFC_WIN_SECONDS = 90.0  # Longitud de la ventana en segundos
DFC_STEP_SECONDS = 15.0   # Paso de la ventana en segundos

# Canal 6: Causalidad de Granger
# NOTA: Interpretar con cautela dado el TR largo. Útil para análisis de ablación.
USE_GRANGER_CHANNEL = True
GRANGER_MAX_LAG = 1

# --- Nombres y Orden de los Canales ---
# Esta lista se construye dinámicamente en el pipeline principal basándose
# en los flags de arriba. No es necesario modificarla aquí.

# --- 5. Parámetros de Ejecución y Paralelización ---
# El número de workers se puede ajustar. Un buen punto de partida es
# la mitad de los cores de la CPU para dejar recursos al sistema.
try:
    import multiprocessing
    TOTAL_CPU_CORES = multiprocessing.cpu_count()
    # Dejar al menos un core libre para el sistema operativo y el proceso principal
    MAX_WORKERS = max(1, TOTAL_CPU_CORES // 2 if TOTAL_CPU_CORES > 2 else 1)
except NotImplementedError:
    TOTAL_CPU_CORES = 1
    MAX_WORKERS = 1

# --- 6. Configuración de Salida ---
# El nombre del directorio de salida se generará dinámicamente en el pipeline
# para reflejar los parámetros utilizados en la ejecución.
OUTPUT_DIR_NAME_BASE = "AAL3_Tesis_Refactored_v7"

# Mapeo de etiquetas de Yeo-17 a nombres de redes para el reordenamiento de ROIs
YEO17_LABELS_TO_NAMES: Dict[int, str] = {
    0: "Background/NonCortical",
    1: "Visual_Peripheral", 2: "Visual_Central",
    3: "Somatomotor_A", 4: "Somatomotor_B",
    5: "DorsalAttention_A", 6: "DorsalAttention_B",
    7: "Salience_VentralAttention_A", 8: "Salience_VentralAttention_B",
    9: "Limbic_A_TempPole", 10: "Limbic_B_OFC",
    11: "Control_C", 12: "Control_A", 13: "Control_B",
    14: "DefaultMode_Temp", 15: "DefaultMode_Core",
    16: "DefaultMode_DorsalMedial", 17: "DefaultMode_VentralMedial"
}