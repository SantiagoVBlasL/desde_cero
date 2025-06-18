# pipeline.py.py
# -*- coding: utf-8 -*-
"""
Pipeline Principal para Extracción de Características de Conectividad fMRI
Versión: v7.0.1_Completed

Descripción:
Orquesta el flujo completo de extracción de características. Este script es el 
motor principal del proyecto y se configura a través de `config.py` y utiliza
las funciones de `utils.py`.
"""
# --- Importaciones Estándar y de Terceros ---
import gc
import logging
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nli_image
from nilearn.datasets import fetch_atlas_yeo_2011
import scipy.io as sio
from scipy.signal import butter, filtfilt, deconvolve, windows
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

# --- Importaciones del Proyecto ---
import config
import utils

# --- Configuración del Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Variables Globales del Pipeline ---
VALID_AAL3_ROI_INFO_DF: Optional[pd.DataFrame] = None
AAL3_MISSING_INDICES_0BASED: Optional[List[int]] = None
INDICES_OF_SMALL_ROIS_TO_DROP: Optional[List[int]] = None
FINAL_N_ROIS_EXPECTED: Optional[int] = None
AAL3_ROI_ORDER_MAPPING: Optional[Dict[str, Any]] = None
CONNECTIVITY_CHANNEL_NAMES: List[str] = []
N_CHANNELS = 0
OUTPUT_DIR: Optional[Path] = None


# --- 1. Lógica de Inicialización y Preparación ---

def _get_aal3_network_mapping() -> Optional[Dict[str, Any]]:
    # Esta función está completa y funcional
    logger.info("Attempting to map AAL3 ROIs to Yeo-17 networks for reordering.")
    if not config.AAL3_NIFTI_PATH.exists():
        logger.error(f"AAL3 NIfTI file NOT found: {config.AAL3_NIFTI_PATH}. Cannot reorder.")
        return None
    if any(v is None for v in [VALID_AAL3_ROI_INFO_DF, INDICES_OF_SMALL_ROIS_TO_DROP, FINAL_N_ROIS_EXPECTED]):
        logger.error("Global AAL3 variables not initialized. Cannot reorder.")
        return None
    try:
        yeo_atlas_obj = fetch_atlas_yeo_2011()
        yeo_img = nib.load(yeo_atlas_obj.thick_17)
        yeo_data = yeo_img.get_fdata().astype(int)
        aal_img_orig = nib.load(config.AAL3_NIFTI_PATH)
        if not np.allclose(aal_img_orig.affine, yeo_img.affine, atol=1e-3) or aal_img_orig.shape != yeo_img.shape:
            logger.warning("AAL3 and Yeo atlases do not match. Resampling AAL3 to Yeo space.")
            aal_img_resampled = nli_image.resample_to_img(aal_img_orig, yeo_img, interpolation='nearest')
            aal_data = aal_img_resampled.get_fdata().astype(int)
        else:
            aal_data = aal_img_orig.get_fdata().astype(int)
        final_rois_df = VALID_AAL3_ROI_INFO_DF.drop(INDICES_OF_SMALL_ROIS_TO_DROP).reset_index(drop=True)
        original_colors = final_rois_df['color'].tolist()
        original_names = final_rois_df['nom_c'].tolist()
        roi_network_mapping = []
        for idx, color in enumerate(original_colors):
            mask = (aal_data == color)
            if not np.any(mask):
                winner_label, yeo_name, overlap = 0, config.YEO17_LABELS_TO_NAMES[0], 0.0
            else:
                overlapping_voxels = yeo_data[mask]
                labels, counts = np.unique(overlapping_voxels[overlapping_voxels != 0], return_counts=True)
                if len(counts) > 0:
                    winner_label = labels[np.argmax(counts)]
                    yeo_name = config.YEO17_LABELS_TO_NAMES.get(winner_label, f"UnknownYeo{winner_label}")
                    overlap = (counts.max() / np.sum(mask)) * 100
                else:
                    winner_label, yeo_name, overlap = 0, config.YEO17_LABELS_TO_NAMES[0], 0.0
            roi_network_mapping.append((color, original_names[idx], winner_label, yeo_name, overlap, idx))
        roi_network_mapping_sorted = sorted(roi_network_mapping, key=lambda x: (x[2] == 0, x[2], x[0]))
        new_order_indices = [item[5] for item in roi_network_mapping_sorted]
        return {
            'order_name': 'aal3_to_yeo17_overlap_sorted',
            'roi_names_original_order': original_names,
            'roi_names_new_order': [item[1] for item in roi_network_mapping_sorted],
            'network_labels_new_order': [item[3] for item in roi_network_mapping_sorted],
            'new_order_indices': new_order_indices
        }
    except Exception as e:
        logger.error(f"Error during ROI reordering: {e}", exc_info=True)
        return None

def initialize_pipeline() -> bool:
    # Esta función está completa y funcional
    global VALID_AAL3_ROI_INFO_DF, AAL3_MISSING_INDICES_0BASED, \
           INDICES_OF_SMALL_ROIS_TO_DROP, FINAL_N_ROIS_EXPECTED, \
           AAL3_ROI_ORDER_MAPPING, CONNECTIVITY_CHANNEL_NAMES, N_CHANNELS
    logger.info("--- Initializing Pipeline ---")
    if not config.AAL3_META_PATH.exists(): return False
    try:
        meta_df = pd.read_csv(config.AAL3_META_PATH, sep='\t')
        meta_df['color'] = pd.to_numeric(meta_df['color'], errors='coerce').dropna().astype(int)
        AAL3_MISSING_INDICES_0BASED = [idx - 1 for idx in config.AAL3_MISSING_INDICES_1BASED]
        VALID_AAL3_ROI_INFO_DF = meta_df[~meta_df['color'].isin(config.AAL3_MISSING_INDICES_1BASED)].copy()
        VALID_AAL3_ROI_INFO_DF.sort_values(by='color', inplace=True); VALID_AAL3_ROI_INFO_DF.reset_index(drop=True, inplace=True)
        small_rois_mask = VALID_AAL3_ROI_INFO_DF['vol_vox'] < config.SMALL_ROI_VOXEL_THRESHOLD
        INDICES_OF_SMALL_ROIS_TO_DROP = VALID_AAL3_ROI_INFO_DF[small_rois_mask].index.tolist()
        FINAL_N_ROIS_EXPECTED = len(VALID_AAL3_ROI_INFO_DF) - len(INDICES_OF_SMALL_ROIS_TO_DROP)
        logger.info(f"Final expected ROI count after reductions: {FINAL_N_ROIS_EXPECTED}")
        AAL3_ROI_ORDER_MAPPING = _get_aal3_network_mapping()
        channels = []
        if config.USE_PEARSON_OMST_CHANNEL: channels.append(utils.PEARSON_OMST_CHANNEL_NAME)
        if config.USE_PEARSON_FULL_SIGNED_CHANNEL and utils.PEARSON_OMST_FALLBACK_NAME not in channels: channels.append(utils.PEARSON_OMST_FALLBACK_NAME)
        if config.USE_MI_CHANNEL_FOR_THESIS: channels.append("MI_KNN_Symmetric")
        if config.USE_DFC_ABS_DIFF_MEAN_CHANNEL: channels.append("dFC_AbsDiffMean")
        if config.USE_DFC_STDDEV_CHANNEL: channels.append("dFC_StdDev")
        if config.USE_GRANGER_CHANNEL: channels.append(f"Granger_F_lag{config.GRANGER_MAX_LAG}")
        CONNECTIVITY_CHANNEL_NAMES = list(dict.fromkeys(channels))
        N_CHANNELS = len(CONNECTIVITY_CHANNEL_NAMES)
        logger.info(f"Connectivity channels to compute: {CONNECTIVITY_CHANNEL_NAMES}")
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize pipeline: {e}", exc_info=True)
        return False

def setup_output_directory() -> bool:
    # Esta función está completa y funcional
    global OUTPUT_DIR
    reorder_suffix = "_ROIreordered" if AAL3_ROI_ORDER_MAPPING else "_AnatomicalOrder"
    dir_name = f"{config.OUTPUT_DIR_NAME_BASE}_{FINAL_N_ROIS_EXPECTED}ROIs_{N_CHANNELS}Ch{reorder_suffix}"
    OUTPUT_DIR = config.BASE_PATH / dir_name
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "individual_tensors").mkdir(exist_ok=True)
        return True
    except OSError as e:
        logger.critical(f"Could not create output directory: {e}")
        return False

# En pipeline.py

def load_subjects_to_process() -> Optional[pd.DataFrame]:
    """
    Carga la lista de sujetos pre-filtrada y validada desde un archivo CSV.
    Se asume que todos los sujetos en este archivo son válidos para procesar.
    """
    logger.info("--- Loading Cleaned Subject List ---")
    
    # 1. Comprobar que el archivo de sujetos limpios existe.
    subjects_path = config.CLEANED_SUBJECT_LIST_PATH
    if not subjects_path.exists():
        logger.critical(f"Cleaned subject list file not found: {subjects_path}")
        logger.critical("Please create a 'SubjectsData_cleaned.csv' with a 'SubjectID' column containing the subjects to process.")
        return None

    try:
        # 2. Cargar el dataframe desde el archivo CSV.
        subjects_df = pd.read_csv(subjects_path)
        
        # 3. Validar que la columna 'SubjectID' exista.
        if 'SubjectID' not in subjects_df.columns:
            logger.critical(f"The file {subjects_path} must contain a column named 'SubjectID'.")
            return None
            
        # 4. Estandarizar los IDs de sujeto a string y eliminar espacios.
        subjects_df['SubjectID'] = subjects_df['SubjectID'].astype(str).str.strip()
        
        # 5. Eliminar filas donde el SubjectID pueda ser nulo o vacío después de la conversión.
        subjects_df.dropna(subset=['SubjectID'], inplace=True)
        subjects_df = subjects_df[subjects_df['SubjectID'] != '']
        
        n_subjects = len(subjects_df)
        if n_subjects == 0:
            logger.warning("The loaded subject list is empty. No subjects to process.")
            return None
        
        logger.info(f"Successfully loaded {n_subjects} subjects to be processed.")
        return subjects_df

    except Exception as e:
        logger.critical(f"Error loading the cleaned subject list from {subjects_path}: {e}", exc_info=True)
        return None


# --- 2. Lógica de Procesamiento por Sujeto (IMPLEMENTACIÓN COMPLETA) ---

def process_subject(subject_info_tuple: Tuple[int, pd.Series]) -> Dict[str, Any]:
    """
    Pipeline completo para un único sujeto. Carga, preprocesa, calcula conectividad
    y guarda el tensor individual.
    """
    _, subject_info = subject_info_tuple
    subject_id = subject_info['SubjectID']
    result = {"id": subject_id, "status": "PENDING", "path": None, "error_log": []}

    try:
        # 1. Cargar datos
        mat_path = config.ROI_SIGNALS_DIR / config.ROI_FILENAME_TEMPLATE.format(subject_id=subject_id)
        if not mat_path.exists():
            result["status"] = f"ERROR: Archivo .mat no encontrado en {mat_path}"
            return result
        
        raw_sigs = utils.load_signals_from_mat(mat_path, config.POSSIBLE_ROI_KEYS)
        if raw_sigs is None:
            result["status"] = "ERROR: No se encontró una clave de señal válida en el .mat"
            return result
        
        # 2. Orientar y reducir ROIs
        oriented_sigs = utils.orient_and_reduce_rois(
            raw_sigs,
            subject_id,
            AAL3_MISSING_INDICES_0BASED,
            INDICES_OF_SMALL_ROIS_TO_DROP,
            FINAL_N_ROIS_EXPECTED,
            config.RAW_DATA_EXPECTED_COLUMNS  # <-- AÑADIR ESTA LÍNEA
        )

        if oriented_sigs is None:
            result["status"] = "ERROR: Fallo en la orientación o reducción de ROIs"
            return result

        # 3. Preprocesar series temporales
        processed_ts = utils.preprocess_time_series(oriented_sigs, subject_id)
        if processed_ts is None:
            result["status"] = "ERROR: Fallo en el preprocesamiento de la serie temporal"
            return result

        # 4. Reordenar ROIs en la serie temporal si el mapeo está activo
        if AAL3_ROI_ORDER_MAPPING and 'new_order_indices' in AAL3_ROI_ORDER_MAPPING:
            logger.info(f"S {subject_id}: Aplicando reordenamiento de ROIs a la serie temporal.")
            processed_ts = processed_ts[:, AAL3_ROI_ORDER_MAPPING['new_order_indices']]

        # 5. Calcular todas las modalidades de conectividad
        conn_matrices = {}
        for name in CONNECTIVITY_CHANNEL_NAMES:
            matrix = None
            # --- BLOQUE CORREGIDO: Llama a la función correcta de utils.py ---
            if name == utils.PEARSON_OMST_CHANNEL_NAME:
                matrix = utils.calculate_pearson_omst_signed_weighted(processed_ts, subject_id)
            elif name == utils.PEARSON_OMST_FALLBACK_NAME:
                matrix = utils.calculate_pearson_full_fisher_z_signed(processed_ts, subject_id)
            elif name == "MI_KNN_Symmetric":
                matrix = utils.calculate_mi_knn_connectivity(processed_ts, config.N_NEIGHBORS_MI, subject_id)
            # --- INICIO DEL BLOQUE CORREGIDO PARA DFC ---
            elif name == "dFC_AbsDiffMean":
                # Convertir segundos a puntos de tiempo (TRs)
                win_points = int(config.DFC_WIN_SECONDS / config.TR_SECONDS)
                step_points = int(config.DFC_STEP_SECONDS / config.TR_SECONDS)
                logger.info(f"dFC (S {subject_id}): Using window={win_points} TRs, step={step_points} TRs.")
                matrix = utils.calculate_custom_dfc_abs_diff_mean(processed_ts, win_points, step_points, subject_id)

            elif name == "dFC_StdDev":
                # Convertir segundos a puntos de tiempo (TRs)
                win_points = int(config.DFC_WIN_SECONDS / config.TR_SECONDS)
                step_points = int(config.DFC_STEP_SECONDS / config.TR_SECONDS)
                # No es necesario repetir el log si los parámetros son los mismos
                matrix = utils.calculate_dfc_std_dev(processed_ts, win_points, step_points, subject_id)
            # --- FIN DEL BLOQUE CORREGIDO PARA DFC ---

            elif name.startswith("Granger_F_lag"):
                matrix = utils.calculate_granger_f_matrix(processed_ts, config.GRANGER_MAX_LAG, subject_id)
            
            if matrix is None:
                result['error_log'].append(f"Channel '{name}' calculation failed.")
                # Usamos una matriz de ceros para no romper el apilado del tensor
                conn_matrices[name] = np.zeros((FINAL_N_ROIS_EXPECTED, FINAL_N_ROIS_EXPECTED), dtype=np.float32)
            else:
                conn_matrices[name] = matrix

        # 6. Normalizar cada canal y apilar en un tensor
        stacked_matrices = []
        for name in CONNECTIVITY_CHANNEL_NAMES:
            matrix = conn_matrices[name]
            off_diagonal_values = matrix[~np.eye(matrix.shape[0], dtype=bool)]
            if np.std(off_diagonal_values) > 1e-9:
                scaler = RobustScaler()
                scaled_values = scaler.fit_transform(off_diagonal_values.reshape(-1, 1)).flatten()
                scaled_matrix = np.zeros_like(matrix, dtype=np.float32)
                scaled_matrix[~np.eye(matrix.shape[0], dtype=bool)] = scaled_values
                stacked_matrices.append(scaled_matrix)
            else:
                stacked_matrices.append(matrix)

        subject_tensor = np.stack(stacked_matrices, axis=0).astype(np.float32)

        # 7. Guardar tensor individual
        tensor_path = OUTPUT_DIR / "individual_tensors" / f"tensor_{subject_id}.npz"
        np.savez_compressed(tensor_path, tensor_data=subject_tensor, subject_id=subject_id)
        
        result["status"] = "SUCCESS"
        result["path"] = str(tensor_path)

    except Exception as e:
        logger.error(f"CRITICAL ERROR processing subject {subject_id}: {e}", exc_info=True)
        result["status"] = f"ERROR: {e}"
    
    return result


# --- 3. Orquestación y Ejecución Principal ---

def main():
    """Flujo de ejecución principal del pipeline."""
    start_time = time.time()
    if not initialize_pipeline(): return
    if not setup_output_directory(): return

    # MODIFICADO: Llamar a la nueva función simplificada
    subjects_to_process = load_subjects_to_process() 
    
    if subjects_to_process is None or subjects_to_process.empty:
        logger.warning("No subjects to process. Exiting.")
        return

    all_results = []
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # Pasamos tupla (índice, fila) a la función de procesamiento
        futures = [executor.submit(process_subject, item) for item in subjects_to_process.iterrows()]
        for future in tqdm(as_completed(futures), total=len(subjects_to_process), desc="Processing Subjects"):
            all_results.append(future.result())

    # Guardar log de procesamiento
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "processing_log.csv", index=False)
    
    # Ensamblar tensor global
    successful_results = [res for res in all_results if res['status'] == 'SUCCESS']
    if not successful_results:
        logger.warning("No subjects were processed successfully. Global tensor not created.")
    else:
        logger.info(f"Assembling global tensor from {len(successful_results)} subjects...")
        tensor_list = [np.load(res['path'])['tensor_data'] for res in successful_results]
        subject_ids = [res['id'] for res in successful_results]
        global_tensor = np.stack(tensor_list, axis=0)
        
        global_tensor_path = OUTPUT_DIR / f"GLOBAL_TENSOR_{OUTPUT_DIR.name}.npz"
        np.savez_compressed(
            global_tensor_path,
            global_tensor_data=global_tensor,
            subject_ids=np.array(subject_ids, dtype=str),
            channel_names=np.array(CONNECTIVITY_CHANNEL_NAMES, dtype=str),
            roi_order_mapping=AAL3_ROI_ORDER_MAPPING
        )
        logger.info(f"Global tensor saved to {global_tensor_path}")

    total_time = (time.time() - start_time) / 60
    logger.info(f"--- Pipeline Finished. Total time: {total_time:.2f} minutes. ---")
    logger.info(f"Check outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()