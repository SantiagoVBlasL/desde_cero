#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo Utilitario para Procesamiento de ROIs AAL3.
Contiene funciones para mapear ROIs a redes y para inicializar la configuración del pipeline.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nli_image
from nilearn.datasets import fetch_atlas_yeo_2011

logger = logging.getLogger(__name__)

# --- Mapeo de ROIs a Redes Yeo-17 ---

YEO17_LABELS_TO_NAMES = {
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

def get_aal3_network_mapping_and_order(
    aal3_nifti_path: Path,
    valid_aal3_roi_info_df: pd.DataFrame,
    indices_to_drop: List[int],
    final_n_rois_expected: int,
    base_path_for_mapping_csv: Path
) -> Optional[Dict[str, Any]]:
    """
    Carga/define el mapeo de ROIs AAL3 a redes Yeo-17 y el nuevo orden.
    """
    logger.info("Attempting to map AAL3 ROIs to Yeo-17 networks and reorder.")

    if not aal3_nifti_path.exists():
        logger.error(f"AAL3 NIfTI file NOT found at: {aal3_nifti_path}. Cannot reorder.")
        return None

    try:
        # 1. Cargar AAL3 (será nuestra referencia espacial)
        logger.info(f"Loading AAL3 NIfTI from: {aal3_nifti_path}")
        aal_img = nib.load(aal3_nifti_path)

        # 2. Cargar Yeo-17 y RESAMPLEARLO al espacio de AAL3
        logger.info("Fetching Yeo 17-network atlas…")
        yeo_atlas_obj = fetch_atlas_yeo_2011()
        yeo_img_orig = nib.load(yeo_atlas_obj.thick_17)
        if not np.allclose(aal_img.affine, yeo_img_orig.affine, atol=1e-3) or \
           aal_img.shape != yeo_img_orig.shape:
            logger.warning("Resampling Yeo-17 ➔ AAL3 space (nearest).")
            yeo_img = nli_image.resample_to_img(yeo_img_orig, aal_img,
                                                interpolation='nearest')
        else:
            yeo_img = yeo_img_orig

        aal_data  = aal_img.get_fdata().astype(int)
        yeo_data  = yeo_img.get_fdata().astype(int)

        # Índices de AAL3 que *siempre* son constantes en tu dataset
        constant_rois = [34, 35, 80, 81]
        drop_all = sorted(set(indices_to_drop) | set(constant_rois))

        final_aal3_rois_info_df = (valid_aal3_roi_info_df
                                   .drop(drop_all)
                                   .reset_index(drop=True))
        
        if len(final_aal3_rois_info_df) != final_n_rois_expected:
             logger.error("Mismatch in expected final ROI count. Cannot proceed with reordering.")
             return None

        original_aal3_colors = final_aal3_rois_info_df['color'].tolist()
        original_aal3_names = final_aal3_rois_info_df['nom_c'].tolist()
        
        roi_network_mapping = []
        for aal3_idx, aal3_color in enumerate(original_aal3_colors):
            aal3_name = original_aal3_names[aal3_idx]
            aal3_roi_mask = (aal_data == aal3_color)
            
            if not np.any(aal3_roi_mask):
                winner_yeo_label, yeo17_name, overlap_percentage = 0, YEO17_LABELS_TO_NAMES[0], 0.0
            else:
                overlapping_yeo_voxels = yeo_data[aal3_roi_mask]
                unique_yeo_labels, counts = np.unique(overlapping_yeo_voxels[overlapping_yeo_voxels != 0], return_counts=True)
                
                if len(counts) > 0:
                    winner_yeo_label_idx = np.argmax(counts)
                    winner_yeo_label = unique_yeo_labels[winner_yeo_label_idx]
                    total_roi_voxels = np.sum(aal3_roi_mask)
                    overlap_percentage = (counts[winner_yeo_label_idx] / total_roi_voxels) * 100 if total_roi_voxels > 0 else 0.0
                    yeo17_name = YEO17_LABELS_TO_NAMES.get(winner_yeo_label, f"UnknownYeo{winner_yeo_label}")
                else:
                    winner_yeo_label, yeo17_name, overlap_percentage = 0, YEO17_LABELS_TO_NAMES[0], 0.0

            roi_network_mapping.append((aal3_color, aal3_name, winner_yeo_label, yeo17_name, overlap_percentage, aal3_idx))

        # 4. Generar el nuevo orden
        roi_network_mapping_sorted = sorted(roi_network_mapping, key=lambda x: (x[2] == 0, x[2], x[0]))
        new_order_indices = [item[5] for item in roi_network_mapping_sorted]
        
        # 5. Guardar y retornar resultados
        mapping_df = pd.DataFrame(roi_network_mapping_sorted, columns=['AAL3_Color', 'AAL3_Name', 'Yeo17_Label', 'Yeo17_Network', 'Overlap_Percent', 'Original_Index_0_N'])
        mapping_filename = base_path_for_mapping_csv / f"aal3_{final_n_rois_expected}_to_yeo17_mapping.csv"
        mapping_filename.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(mapping_filename, index=False)
        logger.info(f"AAL3 to Yeo-17 mapping saved to: {mapping_filename}")
        
        return {
            'order_name': 'aal3_to_yeo17_overlap_sorted',
            'roi_names_original_order': original_aal3_names,
            'roi_names_new_order': [item[1] for item in roi_network_mapping_sorted],
            'network_labels_new_order': [item[3] for item in roi_network_mapping_sorted],
            'new_order_indices': new_order_indices
        }

    except Exception as e:
        logger.error(f"Error during ROI reordering: {e}", exc_info=True)
        return None