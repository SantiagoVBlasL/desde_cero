# utils.py

# --- Importaciones Estándar y de Terceros ---
import warnings
import logging
from typing import Optional, List
from pathlib import Path
import numpy as np
import networkx as nx  # Necesario para el manejo de errores de versión en OMST
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
import config                     # lo usas varias veces
import scipy.io as sio            # _load_signals_from_mat
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt   

# --- Importaciones del Proyecto (dyconnmap) ---
# Es buena práctica definir aquí las variables que se esperan del exterior
# para que el código sea más explícito.
try:
    from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency
    orthogonal_minimum_spanning_tree = threshold_omst_global_cost_efficiency
    OMST_PYTHON_LOADED = True
except ImportError:
    orthogonal_minimum_spanning_tree = None
    OMST_PYTHON_LOADED = False

PEARSON_OMST_CHANNEL_NAME = "Pearson_OMST_GCE_Signed_Weighted"
PEARSON_OMST_FALLBACK_NAME = "Pearson_Full_FisherZ_Signed"

# Esta parte ya la tienes, está perfecta
logger = logging.getLogger(__name__)

# utils.py  – zona de pre-procesamiento
def load_signals_from_mat(mat_path: Path, possible_keys: List[str]):
    return _load_signals_from_mat(mat_path, possible_keys)


# --- 3. Connectivity Calculation Functions ---
def fisher_r_to_z(r_matrix: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r_clean = np.nan_to_num(r_matrix.astype(np.float32), nan=0.0) 
    r_clipped = np.clip(r_clean, -1.0 + eps, 1.0 - eps)
    z_matrix = np.arctanh(r_clipped)
    np.fill_diagonal(z_matrix, 0.0) 
    return z_matrix.astype(np.float32)

def calculate_pearson_full_fisher_z_signed(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]: 
    if ts_subject.shape[0] < 2:
        logger.warning(f"Pearson_Full_FisherZ_Signed (S {sid}): Insufficient timepoints ({ts_subject.shape[0]} < 2).")
        return None
    try:
        corr_matrix = np.corrcoef(ts_subject, rowvar=False).astype(np.float32)
        if corr_matrix.ndim == 0: 
            logger.warning(f"Pearson_Full_FisherZ_Signed (S {sid}): Correlation resulted in a scalar. Input shape: {ts_subject.shape}.")
            num_rois = ts_subject.shape[1]
            return np.zeros((num_rois, num_rois), dtype=np.float32) if num_rois > 0 else None
        
        z_transformed_matrix = fisher_r_to_z(corr_matrix) 
        # logger.info(f"Pearson_Full_FisherZ_Signed (S {sid}): Successfully calculated.") # Can be verbose
        return z_transformed_matrix
    except Exception as e:
        logger.error(f"Error calculating Pearson_Full_FisherZ_Signed for S {sid}: {e}", exc_info=True)
        return None

def calculate_pearson_omst_signed_weighted(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]: 
    if not OMST_PYTHON_LOADED or orthogonal_minimum_spanning_tree is None:
        logger.error(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Dyconnmap OMST function not available. Cannot calculate.")
        return None 
    
    if ts_subject.shape[0] < 2: 
        logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Insufficient timepoints ({ts_subject.shape[0]} < 2).")
        return None
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning) 
            
            corr_matrix = np.corrcoef(ts_subject, rowvar=False).astype(np.float32)
            
            if corr_matrix.ndim == 0: 
                logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Correlation resulted in a scalar. Input shape: {ts_subject.shape}. Returning zero matrix.")
                num_rois = ts_subject.shape[1]
                return np.zeros((num_rois, num_rois), dtype=np.float32) if num_rois > 0 else None
                
            z_transformed_matrix = fisher_r_to_z(corr_matrix) 
            weights_for_omst_gce = np.abs(z_transformed_matrix) 
            np.fill_diagonal(weights_for_omst_gce, 0.0) 

            if np.all(np.isclose(weights_for_omst_gce, 0)):
                 logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): All input weights for OMST GCE are zero. Returning zero matrix (original Z-transformed).")
                 return z_transformed_matrix.astype(np.float32) 
                 
            # logger.info(f"S {sid}: Calling dyconnmap.threshold_omst_global_cost_efficiency with ABSOLUTE weights shape {weights_for_omst_gce.shape}") # Verbose
            
            omst_outputs = orthogonal_minimum_spanning_tree(weights_for_omst_gce, n_msts=None) 
            
            if isinstance(omst_outputs, tuple) and len(omst_outputs) >= 2:
                omst_adjacency_matrix_gce_weighted = np.asarray(omst_outputs[1]).astype(np.float32) 
                # logger.debug(f"S {sid}: dyconnmap.threshold_omst_global_cost_efficiency returned multiple outputs. Using the second one (CIJtree) as omst_adjacency_matrix.") # Verbose
            else:
                logger.error(f"S {sid}: dyconnmap.threshold_omst_global_cost_efficiency returned an unexpected type or insufficient outputs: {type(omst_outputs)}. Cannot extract OMST matrix.")
                return None

            if not isinstance(omst_adjacency_matrix_gce_weighted, np.ndarray): 
                logger.error(f"S {sid}: Extracted omst_adjacency_matrix_gce_weighted is not a numpy array (type: {type(omst_adjacency_matrix_gce_weighted)}). Cannot proceed.")
                return None
            
            binary_omst_mask = (omst_adjacency_matrix_gce_weighted > 0).astype(int)
            signed_weighted_omst_matrix = z_transformed_matrix * binary_omst_mask
            np.fill_diagonal(signed_weighted_omst_matrix, 0.0) 
            
            # logger.info(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Successfully calculated. Matrix density: {np.count_nonzero(signed_weighted_omst_matrix) / signed_weighted_omst_matrix.size:.4f}") # Verbose
            return signed_weighted_omst_matrix.astype(np.float32)

    except AttributeError as ae:
        if 'from_numpy_matrix' in str(ae).lower() or 'from_numpy_array' in str(ae).lower(): 
            logger.error(f"Error calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) for S {sid}: NetworkX version incompatibility. "
                         f"Dyconnmap (v1.0.4) may be using a deprecated NetworkX function. "
                         f"Your NetworkX version: {nx.__version__}. Consider using NetworkX 2.x. Original error: {ae}", exc_info=False) 
        else:
            logger.error(f"AttributeError calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) for S {sid}: {ae}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) connectivity for S {sid}: {e}", exc_info=True)
        return None


# utils.py - VERSIÓN OPTIMIZADA de calculate_mi_knn_connectivity

def _calculate_mi_for_pair(X_i_reshaped, y_j, n_neighbors_val):
    """Calcula el valor de MI para un único par de series temporales."""
    try:
        # El estimador de MI de scikit-learn
        mi_val = mutual_info_regression(X_i_reshaped, y_j, n_neighbors=n_neighbors_val, random_state=42, discrete_features=False)[0]
        return mi_val
    except Exception:
        return 0.0

def calculate_mi_knn_connectivity(ts_subject: np.ndarray, n_neighbors_val: int, sid: str) -> Optional[np.ndarray]:
    """
    Calcula la conectividad por Información Mutua.
    Optimizado para evitar cálculos redundantes (simetría) y paralelismo anidado.
    """
    n_tp, n_rois = ts_subject.shape
    if n_tp <= n_neighbors_val:
        logger.warning(f"MI_KNN (S {sid}): Timepoints ({n_tp}) <= n_neighbors ({n_neighbors_val}). Skipping MI.")
        return np.zeros((n_rois, n_rois), dtype=np.float32)

    mi_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)

    # Crear tareas solo para el triángulo superior de la matriz
    tasks = [{'i': i, 'j': j, 'data_i': ts_subject[:, i].reshape(-1, 1), 'data_j': ts_subject[:, j]}
             for i in range(n_rois) for j in range(i + 1, n_rois)]

    # Ignorar warnings de convergencia que pueden ocurrir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        
        # --- AJUSTE CLAVE 1: n_jobs=1 para evitar paralelismo anidado ---
        # --- AJUSTE CLAVE 2: Calcular MI una sola vez (es simétrica) ---
        results_list = Parallel(n_jobs=1)(
            delayed(_calculate_mi_for_pair)(task['data_i'], task['data_j'], n_neighbors_val) for task in tasks
        )

    # Llenar la matriz simétrica con los resultados
    k = 0
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            # Asignar el mismo valor a (i, j) y (j, i)
            mi_matrix[i, j] = mi_matrix[j, i] = results_list[k]
            k += 1
            
    np.fill_diagonal(mi_matrix, 0.0)
    return mi_matrix

def calculate_custom_dfc_abs_diff_mean(ts_subject: np.ndarray, win_points_val: int, step_val: int, sid: str) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape
    if n_tp < win_points_val: 
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Timepoints ({n_tp}) < window length ({win_points_val}). Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None 
        
    num_windows = (n_tp - win_points_val) // step_val + 1
    if num_windows < 2: 
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Fewer than 2 windows ({num_windows}) can be formed. Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None 
        
    sum_abs_diff_matrix = np.zeros((n_rois, n_rois), dtype=np.float64) 
    n_diffs_calculated = 0
    prev_corr_matrix_abs: Optional[np.ndarray] = None
    
    for idx in range(num_windows):
        start_idx = idx * step_val
        end_idx = start_idx + win_points_val
        window_ts = ts_subject[start_idx:end_idx, :]
        
        if window_ts.shape[0] < 2: continue 
            
        try:
            corr_matrix_window = np.corrcoef(window_ts, rowvar=False)
            if corr_matrix_window.ndim < 2 or corr_matrix_window.shape != (n_rois, n_rois):
                logger.warning(f"dFC_AbsDiffMean (S {sid}), Window {idx}: corrcoef returned unexpected shape {corr_matrix_window.shape}. Using zeros for this window's contribution.")
                corr_matrix_window = np.full((n_rois, n_rois), 0.0, dtype=np.float32) 
            else:
                corr_matrix_window = np.nan_to_num(corr_matrix_window.astype(np.float32), nan=0.0) 
            
            current_corr_matrix_abs = np.abs(corr_matrix_window)
            np.fill_diagonal(current_corr_matrix_abs, 0) 
            
            if prev_corr_matrix_abs is not None:
                sum_abs_diff_matrix += np.abs(current_corr_matrix_abs - prev_corr_matrix_abs)
                n_diffs_calculated += 1
            prev_corr_matrix_abs = current_corr_matrix_abs
        except Exception as e: 
            logger.error(f"dFC_AbsDiffMean (S {sid}), Window {idx}: Error calculating/processing correlation: {e}")
            
    if n_diffs_calculated == 0: 
        logger.warning(f"dFC_AbsDiffMean (S {sid}): No valid differences between windowed correlations were calculated. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    mean_abs_diff_matrix = (sum_abs_diff_matrix / n_diffs_calculated).astype(np.float32)
    np.fill_diagonal(mean_abs_diff_matrix, 0) 
    return mean_abs_diff_matrix

def calculate_dfc_std_dev(ts_subject: np.ndarray, win_points_val: int, step_val: int, sid: str) -> Optional[np.ndarray]: 
    n_tp, n_rois = ts_subject.shape
    if n_tp < win_points_val:
        logger.warning(f"dFC_StdDev (S {sid}): Timepoints ({n_tp}) < window length ({win_points_val}). Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    num_windows = (n_tp - win_points_val) // step_val + 1
    if num_windows < 2: 
        logger.warning(f"dFC_StdDev (S {sid}): Fewer than 2 windows ({num_windows}) can be formed. StdDev would be trivial (0). Skipping and returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    window_corr_matrices_list = []
    
    for idx in range(num_windows):
        start_idx = idx * step_val
        end_idx = start_idx + win_points_val
        window_ts = ts_subject[start_idx:end_idx, :]
        
        if window_ts.shape[0] < 2: continue 
            
        try:
            corr_matrix_window = np.corrcoef(window_ts, rowvar=False)
            if corr_matrix_window.ndim < 2 or corr_matrix_window.shape != (n_rois, n_rois):
                logger.warning(f"dFC_StdDev (S {sid}), Window {idx}: corrcoef returned unexpected shape {corr_matrix_window.shape}. Skipping this window for StdDev.")
                continue 
            else:
                corr_matrix_window = np.nan_to_num(corr_matrix_window.astype(np.float32), nan=0.0) 
            
            np.fill_diagonal(corr_matrix_window, 0) 
            window_corr_matrices_list.append(corr_matrix_window)
        except Exception as e: 
            logger.error(f"dFC_StdDev (S {sid}), Window {idx}: Error calculating/processing correlation: {e}")
            
    if len(window_corr_matrices_list) < 2: 
        logger.warning(f"dFC_StdDev (S {sid}): Fewer than 2 valid windowed correlation matrices were calculated ({len(window_corr_matrices_list)}). Cannot compute StdDev. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    stacked_corr_matrices = np.stack(window_corr_matrices_list, axis=0) 
    std_dev_matrix = np.std(stacked_corr_matrices, axis=0).astype(np.float32)
    np.fill_diagonal(std_dev_matrix, 0) 
    
    # logger.info(f"dFC_StdDev (S {sid}): Successfully calculated from {len(window_corr_matrices_list)} windows.") # Can be verbose
    return std_dev_matrix

def _granger_pair(ts1, ts2, maxlag, sid, i, j):     
    f_ij, f_ji = 0.0, 0.0
    try:
        data_for_ij = np.column_stack([ts2, ts1]) 
        if np.any(np.std(data_for_ij, axis=0) < 1e-6): 
             # logger.debug(f"S {sid}: GC pair ({i}->{j}): Datos con varianza casi nula. Saltando F_ij.") # Verbose
             pass
        else:
            granger_result_ij = grangercausalitytests(data_for_ij, maxlag=[maxlag], verbose=False)
            f_ij = granger_result_ij[maxlag][0]['ssr_ftest'][0] 
        
        data_for_ji = np.column_stack([ts1, ts2]) 
        if np.any(np.std(data_for_ji, axis=0) < 1e-6):
            # logger.debug(f"S {sid}: GC pair ({j}->{i}): Datos con varianza casi nula. Saltando F_ji.") # Verbose
            pass
        else:
            granger_result_ji = grangercausalitytests(data_for_ji, maxlag=[maxlag], verbose=False)
            f_ji = granger_result_ji[maxlag][0]['ssr_ftest'][0]
            
        return f_ij, f_ji
    except Exception as e:
        # logger.debug(f"S {sid}: GC pair ({i},{j}) failed: {e}. Returning (0.0, 0.0)") # Verbose
        return 0.0, 0.0 
        
def calculate_granger_f_matrix(ts_subject: np.ndarray, maxlag: int, sid: str) -> Optional[np.ndarray]:
    """
    Calcula la conectividad por Causalidad de Granger.
    Optimizado para evitar paralelismo anidado.
    """
    n_tp, n_rois = ts_subject.shape
    if n_tp <= maxlag * 4 + 5: # Condición de aplicabilidad de Granger
        logger.warning(f"Granger (S {sid}): Too few TPs ({n_tp}) for lag {maxlag}. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    gc_mat_symmetric = np.zeros((n_rois, n_rois), dtype=np.float32)

    tasks = [{'i': i, 'j': j, 'ts1': ts_subject[:, i], 'ts2': ts_subject[:, j]}
             for i in range(n_rois) for j in range(i + 1, n_rois)]

    # Ignorar advertencias futuras de statsmodels durante el cálculo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        
        # --- AJUSTE CLAVE: n_jobs=1 para evitar paralelismo anidado ---
        results_pairs = Parallel(n_jobs=1)(
            delayed(_granger_pair)(task['ts1'], task['ts2'], maxlag, sid, task['i'], task['j'])
            for task in tasks
        )

    # Llenar la matriz simétrica con los resultados promediados
    for k, task in enumerate(tasks):
        i, j = task['i'], task['j']
        f_ij, f_ji = results_pairs[k]
        # La simetrización por promedio es matemáticamente correcta y se mantiene
        f_sym = (f_ij + f_ji) / 2.0
        gc_mat_symmetric[i, j] = gc_mat_symmetric[j, i] = f_sym

    np.fill_diagonal(gc_mat_symmetric, 0)
    return gc_mat_symmetric

# AÑADE ESTE BLOQUE COMPLETO AL FINAL DE TU ARCHIVO utils.py

# --- Funciones de Pre-procesamiento de Series Temporales ---

def _load_signals_from_mat(mat_path: Path, possible_keys: List[str]) -> Optional[np.ndarray]:
    try:
        data = sio.loadmat(mat_path)
        for key in possible_keys:
            if key in data and isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
                return data[key].astype(np.float64)
        logger.warning(f"No valid signal keys {possible_keys} found in {mat_path.name}.")
        return None
    except Exception as e:
        logger.error(f"Could not load .mat file: {mat_path}. Error: {e}")
        return None
    
# En utils.py

def orient_and_reduce_rois(
    raw_sigs: np.ndarray, 
    subject_id: str, 
    missing_indices: List[int], 
    small_rois_indices: List[int], 
    final_n_rois_expected: int,
    raw_data_expected_columns: int  # <--- PARÁMETRO ADICIONAL
) -> Optional[np.ndarray]:
    """
    Orienta la matriz de señales y reduce las ROIs según los índices precalculados.
    """
    # Usar el parámetro en lugar de config.RAW_DATA_EXPECTED_COLUMNS
    if raw_sigs.shape[1] == raw_data_expected_columns:
        oriented_sigs = raw_sigs
    elif raw_sigs.shape[0] == raw_data_expected_columns:
        oriented_sigs = raw_sigs.T
    else:
        logger.error(f"S {subject_id}: Shape {raw_sigs.shape} does not match expected ROIs ({raw_data_expected_columns}).")
        return None
    
    try:
        # --- INICIO DE LA CORRECCIÓN CLAVE ---
        # Usar los parámetros de la función (missing_indices, small_rois_indices)
        # en lugar de los nombres de variables globales.
        
        # Eliminar ROIs sistemáticamente ausentes
        sigs_after_missing = np.delete(oriented_sigs, missing_indices, axis=1)
        
        # Eliminar ROIs pequeños
        final_sigs = np.delete(sigs_after_missing, small_rois_indices, axis=1)
        # --- FIN DE LA CORRECCIÓN CLAVE ---
        
        if final_sigs.shape[1] != final_n_rois_expected:
            logger.warning(f"S {subject_id}: Final ROI count mismatch. Expected {final_n_rois_expected}, got {final_sigs.shape[1]}.")
            return None
        return final_sigs
        
    except Exception as e:
        logger.error(f"S {subject_id}: Error reducing ROIs: {e}", exc_info=True)
        return None

def _bandpass_filter(sigs: np.ndarray, subject_id: str) -> np.ndarray:
    fs = 1.0 / config.TR_SECONDS
    low = config.LOW_CUT_HZ / (0.5 * fs)
    high = config.HIGH_CUT_HZ / (0.5 * fs)
    b, a = butter(config.FILTER_ORDER, [low, high], btype='band')
    return filtfilt(b, a, sigs, axis=0)

def _homogenize_length(sigs: np.ndarray, subject_id: str) -> np.ndarray:
    target_len = config.TARGET_LEN_TS
    current_len = sigs.shape[0]
    if current_len == target_len:
        return sigs
    
    logger.info(f"S {subject_id}: Homogenizing length from {current_len} to {target_len}.")
    if current_len > target_len:
        return sigs[:target_len, :]
    else: # Interpolar si es más corto
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, sigs, axis=0, kind='linear', fill_value="extrapolate")
        return f(x_new)

def preprocess_time_series(sigs: np.ndarray, subject_id: str) -> Optional[np.ndarray]:
    try:
        # 1. Filtrado Pasa-Banda
        filtered_sigs = _bandpass_filter(sigs, subject_id)
        
        # 2. Escalado (Z-score)
        scaled_sigs = StandardScaler().fit_transform(filtered_sigs)
        
        # 3. Homogeneizar longitud
        final_sigs = _homogenize_length(scaled_sigs, subject_id)
        
        return final_sigs.astype(np.float32)
    except Exception as e:
        logger.error(f"S {subject_id}: Error in preprocessing pipeline: {e}", exc_info=True)
        return None