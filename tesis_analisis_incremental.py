#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Incremental y Avanzado de Conectividad fMRI
=====================================================
Autor: [Tu Nombre] - Adaptado para análisis incremental por Gemini
Versión: 1.2.1-incremental (2025-06-18)

Descripción
-----------
Este script está diseñado para analizar los resultados del pipeline de conectividad
(pipeline.py) a medida que se generan, sin necesidad de esperar a que la ejecución
completa termine.

Flujo de Análisis (Modificado):
-------------------------------
1.  **Carga de Configuración del Pipeline:** Lee el mismo archivo `config.py`
    utilizado por el pipeline para reconstruir metadatos clave.
2.  **Reconstrucción de Metadatos:** Genera la lista de canales de conectividad
    y el orden de los ROIs (anatómico o reordenado por redes Yeo-17),
    asegurando consistencia con la corrida del pipeline.
3.  **Carga Incremental de Datos:** Escanea el directorio de salida del pipeline
    (ej. `.../individual_tensors`), carga todos los tensores de sujetos
    disponibles y los apila en un tensor global temporal.
4.  **Análisis Completo:** Con el tensor temporalmente ensamblado, procede con
    el flujo de análisis completo del script original (Teoría de la Información,
    UMAP, RF, Análisis de Redes, etc.).

Uso
---
Este script puede ser ejecutado en cualquier momento mientras el `pipeline.py`
esté corriendo.

```bash
# Ejemplo de uso
python Tesis_Analisis_Incremental.py \
    --tensor-dir /ruta/a/AAL3_Tesis_Refactored_v7_.../individual_tensors \
    --metadata /ruta/a/SubjectsData_cleaned.csv \
    --roi-meta /ruta/a/ROI_MNI_V7_vol.txt \
    --outdir ./resultados_analisis_parciales
```
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import importlib.metadata as md
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import umap
from captum.attr import LayerGradCam
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# --- Importaciones del pipeline para reconstruir metadatos ---
try:
    import config
    import utils
    from pipeline import _get_aal3_network_mapping, initialize_pipeline
except ImportError as e:
    print(f"ERROR: No se pudo importar un módulo del pipeline ({e}).")
    print("Asegúrate de que este script esté en el mismo directorio que 'pipeline.py', 'utils.py' y 'config.py'.")
    sys.exit(1)


# --- Configuración de Estilo para Plots y Semilla Global ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
SEED = 42
logger = logging.getLogger(__name__)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ MÓDULO 0: CONFIGURACIÓN Y REPRODUCIBILIDAD (Sin cambios)           ║
# ╚════════════════════════════════════════════════════════════════════╝

def setup_logging(outdir: Path) -> logging.Logger:
    """Configura un logger dual para consola y archivo."""
    log_file = outdir / "analysis_run.log"
    # Limpiar handlers existentes para evitar logs duplicados
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    return logger

def set_seeds(seed: int):
    """Fija las semillas para reproducibilidad en numpy, torch y otros."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_run_config(args: argparse.Namespace, outdir: Path):
    """Guarda los argumentos y versiones de librerías en un JSON."""
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    try:
        required_packages = [
            "numpy", "pandas", "scikit-learn", "torch", "captum", 
            "umap-learn", "nilearn", "networkx", "statsmodels", "seaborn", "matplotlib"
        ]
        versions = {pkg: md.version(pkg) for pkg in required_packages}
    except md.PackageNotFoundError as e:
        versions = {"error": f"Could not retrieve package version for {e.name}"}

    full_config = {"arguments": config_dict, "versions": versions}
    with open(outdir / "run_config.json", "w") as f:
        json.dump(full_config, f, indent=4)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ MÓDULO 1: CARGA DE DATOS Y METADATOS (MODIFICADO)                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def reconstruct_pipeline_metadata() -> Tuple[List[str], Dict[str, Any]]:
    """
    Reconstruye los nombres de canales y la info de ROIs usando la lógica del pipeline.
    """
    logger.info("--- 1.1: Reconstruyendo Metadatos desde la Configuración del Pipeline ---")
    
    # Ejecutar la inicialización del pipeline para poblar las variables globales
    # que necesitamos, como AAL3_ROI_ORDER_MAPPING
    if not initialize_pipeline():
        raise RuntimeError("Fallo al inicializar la configuración del pipeline para reconstruir metadatos.")

    # Reconstruir dinámicamente la lista de canales según los flags de config
    channel_names: List[str] = []
    if getattr(config, "USE_PEARSON_OMST_CHANNEL", False):
        channel_names.append("Pearson_OMST_GCE_Signed_Weighted")
    if getattr(config, "USE_PEARSON_FULL_SIGNED_CHANNEL", False):
        channel_names.append("Pearson_Full_FisherZ_Signed")
    if getattr(config, "USE_MI_CHANNEL_FOR_THESIS", False):
        channel_names.append("MI_KNN_Symmetric")
    if getattr(config, "USE_DFC_ABS_DIFF_MEAN_CHANNEL", False):
        channel_names.append("dFC_AbsDiffMean")
    if getattr(config, "USE_DFC_STDDEV_CHANNEL", False):
        channel_names.append("dFC_StdDev")
    if getattr(config, "USE_GRANGER_CHANNEL", False):
        # usa el parámetro GRANGER_MAX_LAG para el nombre
        channel_names.append(f"Granger_F_lag{config.GRANGER_MAX_LAG}")

    
    # La función `initialize_pipeline` ya ha llamado a `_get_aal3_network_mapping`
    # y ha almacenado el resultado en `AAL3_ROI_ORDER_MAPPING`.
    # En vez de leer config.AAL3_ROI_ORDER_MAPPING, invocamos directamente:
    roi_mapping = _get_aal3_network_mapping()

    if roi_mapping:
        logger.info("Reordenamiento de ROIs detectado. Usando nombres y etiquetas de red reordenados.")
        analysis_metadata = {
            "roi_names": roi_mapping['roi_names_new_order'],
            "network_labels": roi_mapping['network_labels_new_order']
        }
    else:
        logger.warning("No se detectó reordenamiento de ROIs. Se usará el orden anatómico filtrado.")
        # Recrear la lista de ROIs filtrados si no hay reordenamiento
        meta_df = pd.read_csv(config.AAL3_META_PATH, sep='\t')
        valid_rois = meta_df[~meta_df['color'].isin(config.AAL3_MISSING_INDICES_1BASED)]
        small_rois_mask = valid_rois['vol_vox'] < config.SMALL_ROI_VOXEL_THRESHOLD
        final_rois_df = valid_rois[~small_rois_mask]
        analysis_metadata = {
            "roi_names": final_rois_df['nom_c'].tolist(),
            "network_labels": [] # No hay etiquetas de red sin el mapeo
        }

    logger.info(f"Canales reconstruidos: {channel_names}")
    logger.info(f"Reconstruidos {len(analysis_metadata['roi_names'])} nombres de ROIs.")
    logger.info("-" * 60)
    
    return channel_names, analysis_metadata


def load_data_incrementally(
    tensor_dir: Path, metadata_path: Path, label_col: str, 
    analysis_metadata: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]]:
    """Carga tensores individuales desde un directorio, los apila y alinea los metadatos."""
    logger.info(f"--- 1.2: Cargando Datos Incrementales desde: {tensor_dir} ---")

    if not tensor_dir.is_dir():
        logger.critical(f"El directorio de tensores especificado no existe: {tensor_dir}")
        return None

    individual_tensor_files = sorted(list(tensor_dir.glob("tensor_*.npz")))

    if not individual_tensor_files:
        logger.critical(f"No se encontraron archivos de tensor ('tensor_*.npz') en {tensor_dir}")
        return None

    logger.info(f"Encontrados {len(individual_tensor_files)} sujetos procesados en el directorio.")

    all_tensors, all_subject_ids = [], []
    for f in tqdm(individual_tensor_files, desc="Cargando tensores individuales", leave=False):
        try:
            data = np.load(f)
            all_tensors.append(data["tensor_data"])
            all_subject_ids.append(str(data["subject_id"]))
        except Exception as e:
            logger.warning(f"No se pudo cargar o leer el archivo {f.name}: {e}")

    if not all_tensors:
        logger.critical("No se pudo cargar ningún tensor válido.")
        return None

    # Apilar en un tensor global temporal
    tensor = np.stack(all_tensors, axis=0).astype(np.float32)

    df_meta = pd.read_csv(metadata_path)
    df_meta['SubjectID'] = df_meta['SubjectID'].astype(str)
    
    df_subjects_found = pd.DataFrame({'SubjectID': all_subject_ids})
    df_aligned = pd.merge(df_subjects_found, df_meta, on='SubjectID', how='left')
    
    # Manejar sujetos sin metadatos
    if df_aligned[label_col].isnull().any():
        nan_mask = df_aligned[label_col].isnull()
        valid_mask = ~nan_mask
        excluded_subjects = df_aligned[nan_mask]['SubjectID'].tolist()
        logger.warning(f"{len(excluded_subjects)} sujetos no tienen metadatos y serán excluidos: {excluded_subjects}")
        
        df_aligned = df_aligned[valid_mask].reset_index(drop=True)
        tensor = tensor[valid_mask]
        all_subject_ids = df_aligned['SubjectID'].tolist()

    if df_aligned.empty:
        logger.critical("Ningún sujeto tenía metadatos válidos. No se puede continuar.")
        return None

    le = LabelEncoder()
    y = le.fit_transform(df_aligned[label_col])
    analysis_metadata['label_encoder'] = le

    logger.info(f"Tensor final ensamblado: {tensor.shape} (Sujetos, Canales, ROIs, ROIs)")
    logger.info(f"Metadatos alineados: {df_aligned.shape[0]} sujetos")
    logger.info(f"Distribución de etiquetas: {dict(zip(le.classes_, np.bincount(y)))}")
    logger.info("-" * 60)
    
    return tensor, y, all_subject_ids, df_aligned

# --- SE COPIAN LAS FUNCIONES DE ANÁLISIS DEL SCRIPT ORIGINAL ---
# --- (Estas funciones no necesitan cambios) ---

def regress_out_confounds(X: np.ndarray, df_covariates: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    logger.info(f"Regresando {df_covariates.shape[1]} covariables de la matriz de datos X.")
    is_cat = df_covariates.select_dtypes(include=['object', 'category']).columns
    is_num = df_covariates.select_dtypes(include=np.number).columns
    Z_parts = []
    if not is_num.empty: Z_parts.append(StandardScaler().fit_transform(df_covariates[is_num]))
    if not is_cat.empty: Z_parts.append(pd.get_dummies(df_covariates[is_cat], drop_first=True).values)
    Z = np.hstack(Z_parts)
    lm = LinearRegression(n_jobs=-1)
    X_residuals = np.zeros_like(X)
    for i in tqdm(range(X.shape[1]), desc="Regresando confundidos", leave=False):
        lm.fit(Z, X[:, i]); X_residuals[:, i] = X[:, i] - lm.predict(Z)
    return X_residuals

def flatten_tensor_and_create_feature_names(tensor: np.ndarray, channel_names: List[str], roi_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    n_subj, n_ch, n_roi, _ = tensor.shape; mask = np.triu_indices(n_roi, k=1); n_edges = len(mask[0])
    X_flat = np.empty((n_subj, n_ch * n_edges), dtype=np.float32)
    feature_names = [f"{c_name}|{roi_names[i]}-{roi_names[j]}" for c_name in channel_names for i, j in zip(*mask)]
    for s_idx in range(n_subj): X_flat[s_idx, :] = np.concatenate([tensor[s_idx, c_idx][mask] for c_idx in range(n_ch)])
    return X_flat, feature_names

def information_theory_analysis(X_flat: np.ndarray, y: np.ndarray, feature_names: List[str], channel_names: List[str], outdir: Path, n_top_features: int = 20) -> pd.DataFrame:
    logger.info("--- 2.1: Análisis con Teoría de la Información (Mutual Information) ---")
    logger.info("Calculando Información Mutua para cada conexión..."); mi_scores = mutual_info_classif(X_flat, y, random_state=SEED)
    df_mi = pd.DataFrame({'feature_name': feature_names, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
    logger.info(f"\nTop {n_top_features} conexiones más informativas (mayor MI):\n{df_mi.head(n_top_features).to_string()}")
    df_mi['channel'] = df_mi['feature_name'].apply(lambda x: x.split('|')[0])
    channel_importance = df_mi.groupby('channel')['mi_score'].sum().sort_values(ascending=False)
    logger.info(f"\nImportancia agregada por canal de conectividad:\n{channel_importance.to_string()}")
    plt.figure(figsize=(10, 6)); sns.barplot(x=channel_importance.values, y=channel_importance.index, palette="viridis")
    plt.title("Importancia Agregada por Canal de Conectividad (Suma de MI)"); plt.xlabel("Suma de Puntuaciones de Información Mutua"); plt.ylabel("Canal de Conectividad")
    plt.tight_layout(); plt.savefig(outdir / "importancia_canales_mi.png", dpi=300, bbox_inches='tight'); plt.close()
    logger.info("-" * 60); return df_mi

def visualize_and_classify(X_flat: np.ndarray, y: np.ndarray, df_mi: pd.DataFrame, le: LabelEncoder, outdir: Path, top_k_features: int):
    logger.info("--- 3.1: Visualización UMAP y Clasificación de Baseline ---")
    top_features_indices = df_mi.head(top_k_features).index; X_top = X_flat[:, top_features_indices]
    logger.info(f"Generando proyección UMAP 2D usando las {top_k_features} mejores features..."); reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    embedding = reducer.fit_transform(X_top)
    plt.figure(figsize=(8, 7))
    for i, class_name in enumerate(le.classes_): plt.scatter(embedding[y == i, 0], embedding[y == i, 1], label=class_name, alpha=0.7)
    plt.title(f'Proyección UMAP de las {top_k_features} Features Más Informativas'); plt.xlabel('Componente UMAP 1'); plt.ylabel('Componente UMAP 2'); plt.legend()
    plt.savefig(outdir / f"umap_projection_top{top_k_features}.png", dpi=300, bbox_inches='tight'); plt.close(); np.save(outdir / f"umap_embedding_top{top_k_features}.npy", embedding)
    logger.info("Realizando clasificación con RandomForest..."); clf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight='balanced'); skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, reports = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_top, y)):
        X_train, X_test = X_top[train_idx], X_top[test_idx]; y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled, y_train); y_proba = clf.predict_proba(X_test_scaled)
        if len(le.classes_) == 2: fold_auc = roc_auc_score(y_test, y_proba[:, 1])
        else: y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_))); fold_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        y_pred = clf.predict(X_test_scaled); aucs.append(fold_auc); reports.append(classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)); logger.info(f"Fold {fold+1}/5 - AUC: {fold_auc:.4f}")
    logger.info(f"\nResultado Final - AUC Promedio: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    with open(outdir / "clasificacion_baseline_rf_results.json", "w") as f: json.dump({"roc_auc_folds": aucs, "classification_reports": reports}, f, indent=4)
    logger.info("-" * 60)

def network_analysis(tensor: np.ndarray, y: np.ndarray, channel_names: List[str], analysis_metadata: Dict[str, Any], outdir: Path, fdr_alpha: float) -> Dict[str, pd.DataFrame]:
    logger.info("--- 4.1: Análisis Neurocientífico a Nivel de Redes (Yeo-17) ---")
    network_labels = analysis_metadata.get("network_labels"); le = analysis_metadata.get("label_encoder")
    if not network_labels or not le or len(network_labels) == 0:
        logger.error("No se encontraron etiquetas de red o codificador. Saltando análisis de redes.")
        return {}
    unique_networks = sorted(list(set(network_labels))); network_indices = {net: np.where(np.array(network_labels) == net)[0] for net in unique_networks}
    results = {}
    for c_idx, channel in enumerate(channel_names):
        logger.info(f"\nAnalizando Canal de Red: {channel}")
        avg_network_conn = np.zeros((tensor.shape[0], len(unique_networks), len(unique_networks)))
        for s_idx in range(tensor.shape[0]):
            conn_matrix = tensor[s_idx, c_idx, :, :]
            for i, net1 in enumerate(unique_networks):
                for j, net2 in enumerate(unique_networks):
                    idx1, idx2 = network_indices.get(net1, []), network_indices.get(net2, [])
                    if len(idx1) > 0 and len(idx2) > 0: avg_network_conn[s_idx, i, j] = np.mean(np.abs(conn_matrix[np.ix_(idx1, idx2)]))
        group0_conn, group1_conn = avg_network_conn[y == 0], avg_network_conn[y == 1]
        t_stat, p_val = ttest_ind(group0_conn, group1_conn, axis=0, equal_var=False, nan_policy='omit')
        p_flat = p_val.flatten(); _, p_fdr, _, _ = multipletests(p_flat[~np.isnan(p_flat)], alpha=fdr_alpha, method='fdr_bh'); p_val_fdr = np.full(p_flat.shape, np.nan); p_val_fdr[~np.isnan(p_flat)] = p_fdr; p_val_fdr = p_val_fdr.reshape(p_val.shape)
        mean0, mean1 = np.nanmean(group0_conn, axis=0), np.nanmean(group1_conn, axis=0)
        std0, std1 = np.nanstd(group0_conn, axis=0), np.nanstd(group1_conn, axis=0)
        pooled_std = np.sqrt(((len(group0_conn) - 1) * std0**2 + (len(group1_conn) - 1) * std1**2) / (len(group0_conn) + len(group1_conn) - 2)); cohen_d = (mean1 - mean0) / pooled_std
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1, 1, 1.2]}); fig.suptitle(f"Análisis de Conectividad de Red - Canal: {channel}", fontsize=16)
        g0_label, g1_label = le.classes_[0], le.classes_[1]
        sns.heatmap(mean0, xticklabels=unique_networks, yticklabels=unique_networks, ax=axes[0], cmap="viridis"); axes[0].set_title(f"Conectividad Media - {g0_label}")
        sns.heatmap(mean1, xticklabels=unique_networks, yticklabels=unique_networks, ax=axes[1], cmap="viridis"); axes[1].set_title(f"Conectividad Media - {g1_label}")
        significant_mask = p_val_fdr < fdr_alpha; sns.heatmap(cohen_d, xticklabels=unique_networks, yticklabels=unique_networks, ax=axes[2], cmap="coolwarm", center=0, annot=significant_mask, fmt=".2f"); axes[2].set_title(f"Cohen's d [{g1_label} - {g0_label}] (FDR < {fdr_alpha})")
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(outdir / f"network_analysis_{channel}.png", dpi=300); plt.close()
    logger.info("-" * 60); return results

# ╔════════════════════════════════════════════════════════════════════╗
# ║ FUNCIÓN PRINCIPAL (main) - MODIFICADA PARA CARGA INCREMENTAL       ║
# ╚════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Análisis Incremental de Conectividad Funcional.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tensor-dir", type=Path, required=True, help="Directorio que contiene los tensores individuales (ej. '.../individual_tensors').")
    parser.add_argument("--metadata", type=Path, required=True, help="Ruta al archivo CSV de metadatos de sujetos.")
    parser.add_argument("--roi-meta", type=Path, required=True, help="Ruta al archivo de metadatos de ROIs (ROI_MNI_V7_vol.txt).")
    parser.add_argument("--outdir", type=Path, default=Path("./resultados_analisis_incremental"), help="Directorio para guardar los resultados.")
    parser.add_argument("--label-col", type=str, default="ResearchGroup", help="Nombre de la columna con las etiquetas de grupo.")
    parser.add_argument("--covariates", nargs='*', default=None, help="Lista de covariables a regresar (ej. Age Sex).")
    parser.add_argument("--top-k", type=int, default=2500, help="Número de 'top features' a usar para UMAP y clasificación.")
    parser.add_argument("--skip-network", action='store_true', help="Omitir el análisis de grafos a nivel de red.")
    
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(args.outdir)
    set_seeds(SEED)
    save_run_config(args, args.outdir)

    # 1. Reconstruir metadatos desde la configuración del pipeline
    channel_names, analysis_metadata = reconstruct_pipeline_metadata()
    
    # 2. Cargar datos de forma incremental
    loaded_data = load_data_incrementally(args.tensor_dir, args.metadata, args.label_col, analysis_metadata)
    if loaded_data is None:
        logger.critical("La carga de datos falló. No se puede continuar.")
        return
    tensor, y, subject_ids, df_aligned = loaded_data

    # 3. Aplanar el tensor y crear nombres de características
    X_flat, feature_names = flatten_tensor_and_create_feature_names(tensor, channel_names, analysis_metadata['roi_names'])

    # 4. Regresión de confundidos (opcional)
    if args.covariates:
        df_covariates = df_aligned[args.covariates].copy()
        if df_covariates.isnull().values.any():
            logger.warning("Se encontraron NaNs en las covariables. Se necesita imputación. Saltando regresión.")
        else:
            X_flat = regress_out_confounds(X_flat, df_covariates)

    # 5. Análisis con Teoría de la Información
    df_mi = information_theory_analysis(X_flat, y, feature_names, channel_names, args.outdir)

    # 6. Visualización y Clasificación Baseline
    visualize_and_classify(X_flat, y, df_mi, analysis_metadata['label_encoder'], args.outdir, args.top_k)

    # 7. Análisis de Redes (Opcional)
    if not args.skip_network:
        network_analysis(tensor, y, channel_names, analysis_metadata, args.outdir, fdr_alpha=0.05)
    
    logger.info("Análisis incremental completado exitosamente.")


if __name__ == "__main__":
    main()