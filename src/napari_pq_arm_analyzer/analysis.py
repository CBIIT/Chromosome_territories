# Author: Adib Keikhosravi, Ph.D.
# Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI
# National Institutes of Health
# Email: adib.keikhosravi@nih.gov
# License: MIT License

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import mesh_surface_area, marching_cubes, regionprops
from skimage.morphology import ball
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from tifffile import imwrite

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - joblib is installed with scikit-learn
    Parallel = None
    delayed = None

ProgressCallback = Optional[Callable[[str], None]]
SpacingZYX = Tuple[float, float, float]
EPS = 1.0e-8

# Columns kept for internal QC/tuning but intentionally omitted from the final
# per-nucleus CSV requested for the streamlined public table.
EXCLUDED_RESULT_COLUMNS = {
    "p_field_background_median_raw",
    "p_field_background_mad_raw",
    "q_field_background_median_raw",
    "q_field_background_mad_raw",
    "p_shell_1_frac",
    "q_shell_1_frac",
    "nucleus_shape_bbox_z_um",
    "nucleus_shape_bbox_y_um",
    "nucleus_shape_bbox_x_um",
    "p_shape_bbox_z_um",
    "p_shape_bbox_y_um",
    "p_shape_bbox_x_um",
    "p_shape_bbox_volume_um3",
    "q_shape_bbox_z_um",
    "q_shape_bbox_y_um",
    "q_shape_bbox_x_um",
    "q_shape_bbox_volume_um3",
    "pq_overlap_shape_bbox_z_um",
    "pq_overlap_shape_bbox_y_um",
    "pq_overlap_shape_bbox_x_um",
    "pq_overlap_shape_bbox_volume_um3",
    "p_nucleus_distance_transform_max_um",
    "p_nucleus_distance_transform_boundary_um",
    "q_nucleus_distance_transform_max_um",
    "q_nucleus_distance_transform_boundary_um",
    "pq_overlap_nucleus_distance_transform_max_um",
    "pq_overlap_nucleus_distance_transform_boundary_um",
    "p_delta_bic_or_llr",
    "p_expected_signal_volume_um3",
    "p_expected_signal_fraction",
    "p_signal_snr",
    "p_top1pct_mean_norm_intensity",
    "p_mean_posterior_over_expected_signal",
    "p_presence_probability",
    "p_presence_accepted",
    "p_rejection_reason",
    "p_signal_snr_p95",
    "p_gmm_n_components_used",
    "p_gmm_delta_bic",
    "p_gmm_means_sorted",
    "p_n_candidate_components",
    "p_n_kept_components",
    "p_best_component_score",
    "p_best_component_volume_um3",
    "p_best_component_mean_prob",
    "p_mean_posterior_in_final_mask",
    "q_delta_bic_or_llr",
    "q_expected_signal_volume_um3",
    "q_expected_signal_fraction",
    "q_signal_snr",
    "q_top1pct_mean_norm_intensity",
    "q_mean_posterior_over_expected_signal",
    "q_presence_probability",
    "q_presence_accepted",
    "q_rejection_reason",
    "q_signal_snr_p95",
    "q_gmm_n_components_used",
    "q_gmm_delta_bic",
    "q_gmm_means_sorted",
    "q_n_candidate_components",
    "q_n_kept_components",
    "q_best_component_score",
    "q_best_component_volume_um3",
    "q_best_component_mean_prob",
    "q_mean_posterior_in_final_mask",
}


def filter_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the public-facing results table with internal QC columns removed."""
    if df is None or df.empty:
        return df
    return df.drop(columns=[c for c in EXCLUDED_RESULT_COLUMNS if c in df.columns])


ARM_DETECTION_METHODS = [
    "Legacy 1D GMM",
    "Upgraded 1D GMM + gate + scoring",
    "MRF/CRF refinement",
]

NUCLEI_SEGMENTATION_MODES = [
    "Cellpose 3D whole volume",
    "Cellpose 2D stack batch + overlap stitch",
    "Cellpose 2D parallel slices + overlap stitch",
]

PARALLEL_BACKENDS = [
    "threading",
    "loky",
    "sequential",
]

FIELD_NORMALIZATION_MODES = [
    "none",
    "outside_nuclei",
    "nuclear_low_percentile",
    "whole_image_low_percentile",
    "mixed_outside_or_low_percentile",
]

COMPONENT_SELECTION_MODES = [
    "largest",
    "best_score",
    "all_passing_score",
    "all_after_size_filter",
    "none",
]


@dataclass
class AnalysisParameters:
    """All tunable parameters used by the napari widget and batch analysis."""

    nuc_channel: int = 1
    p_channel: int = 2
    q_channel: int = 3

    # Cellpose nuclei segmentation
    cellpose_model: str = "nuclei"
    gpu: bool = False
    diameter: float = 30.0
    xy_downsample: float = 2.0
    min_nucleus_volume_um3: float = 5.0
    bbox_pad_xy: int = 32
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    stitch_threshold: float = 0.0
    cellpose_segmentation_mode: str = "Cellpose 3D whole volume"
    cellpose_batch_size: int = 8
    cellpose_stitch_overlap: float = 0.10

    # Parallel execution. parallel_n_jobs=0 means automatic.
    parallel_n_jobs: int = 0
    parallel_backend: str = "threading"

    # Method selector and common arm parameters
    arm_detection_method: str = "Upgraded 1D GMM + gate + scoring"
    gmm_components: int = 4
    auto_choose_gmm_components_by_bic: bool = False
    p_min_sorted_class: int = 3
    q_min_sorted_class: int = 3
    gmm_covariance_type: str = "full"
    gmm_random_state: int = 0
    max_voxels_per_nucleus_fit: int = 50000

    # Intensity normalization and population priors
    field_normalization_mode: str = "mixed_outside_or_low_percentile"
    background_percentile: float = 35.0
    max_context_sample_voxels: int = 250000
    external_prior_json: str = ""
    use_external_prior_if_available: bool = True

    # Smoothing and optional old binary morphology
    arm_smoothing_sigma: float = 0.0
    apply_binary_morphology: bool = False
    arm_opening_radius: int = 0
    arm_closing_radius: int = 0
    min_arm_volume_um3: float = 0.1
    keep_largest_arm_component: bool = False  # kept for backward compatibility; component_selection supersedes this

    # Presence / absence gate
    enable_presence_gate: bool = True
    min_delta_bic: float = 10.0
    min_signal_snr: float = 2.0
    min_signal_fraction: float = 0.0005
    max_signal_fraction: float = 0.65
    min_mean_posterior: float = 0.35

    # Probability thresholding and hysteresis
    hard_probability_threshold: float = 0.50
    use_hysteresis: bool = True
    probability_low_threshold: float = 0.35
    probability_high_threshold: float = 0.75

    # Component scoring
    component_selection: str = "all_passing_score"
    component_score_threshold: float = 0.30
    component_weight_probability: float = 1.00
    component_weight_contrast: float = 0.60
    component_weight_volume: float = 0.35
    component_weight_boundary_penalty: float = 0.20

    # Spatial GMM feature weights, method 8.1
    spatial_coordinate_weight: float = 0.20
    spatial_radial_weight: float = 0.45
    spatial_contrast_weight: float = 0.50
    spatial_gradient_weight: float = 0.25

    # MRF/CRF-like refinement, method 8.2
    use_mrf_refinement: bool = False
    mrf_iterations: int = 5
    mrf_lambda: float = 1.0
    mrf_edge_sigma: float = 1.0

    # Nucleus-level presence classifier, method 8.4
    classifier_threshold: float = 0.50
    classifier_bias: float = 0.0
    classifier_peak_z_threshold: float = 3.0

    # Measurements
    contact_dilation_radius: int = 1
    n_shells: int = 5
    limit_nuclei: int = 0

    # Output options
    save_qc: bool = True
    save_label_masks: bool = True
    save_probability_maps: bool = True


@dataclass
class ArmFieldStats:
    arm_name: str
    normalization_mode: str = "none"
    background_median_raw: float = 0.0
    background_mad_raw: float = 1.0
    background_n_voxels: int = 0
    prior_mu_bg: float = 0.0
    prior_sigma_bg: float = 1.0
    prior_mu_sig: float = 3.0
    prior_sigma_sig: float = 1.0
    prior_pi_sig: float = 0.05
    prior_delta_bic: float = 0.0
    prior_source: str = "current_field"
    prior_n_voxels: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ArmRuntimeContext:
    arm_name: str
    raw_img: np.ndarray
    normalized_img: np.ndarray
    stats: ArmFieldStats


@dataclass
class ArmSegmentationResult:
    mask: np.ndarray
    probability: np.ndarray
    metrics: dict


@dataclass
class AnalysisOutputs:
    output_dir: Path
    plot_dir: Path
    per_nucleus_csv: Path
    population_summary_csv: Path
    nuclei_labels_tif: Path
    p_mask_tif: Path
    q_mask_tif: Path
    overlap_mask_tif: Path
    n_nuclei: int
    summary: dict


def log(message: str, progress: ProgressCallback = None) -> None:
    if progress is not None:
        progress(message)
    else:
        print(message, flush=True)


def _morph_call(func, image: np.ndarray, selem: np.ndarray) -> np.ndarray:
    try:
        return func(image, footprint=selem)
    except TypeError:
        return func(image, selem=selem)


from skimage.morphology import binary_closing as _binary_closing
from skimage.morphology import binary_erosion as _binary_erosion
from skimage.morphology import binary_opening as _binary_opening


def binary_opening_compat(image: np.ndarray, selem: np.ndarray) -> np.ndarray:
    return _morph_call(_binary_opening, image, selem)


def binary_closing_compat(image: np.ndarray, selem: np.ndarray) -> np.ndarray:
    return _morph_call(_binary_closing, image, selem)


def binary_erosion_compat(image: np.ndarray, selem: np.ndarray) -> np.ndarray:
    return _morph_call(_binary_erosion, image, selem)


def safe_label_dtype(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.astype(np.uint16)
    max_label = int(np.nanmax(arr))
    if max_label <= np.iinfo(np.uint16).max:
        return arr.astype(np.uint16)
    return arr.astype(np.uint32)


def write_tiff(path: Path, arr: np.ndarray) -> None:
    """Save arrays as ImageJ-compatible TIFF when possible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)

    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.integer) and arr.dtype.kind in {"i", "u"}:
        if arr.size and int(np.nanmax(arr)) > np.iinfo(np.uint8).max:
            arr = safe_label_dtype(arr)
        else:
            arr = arr.astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)

    imagej_ok = arr.dtype in (np.uint8, np.uint16, np.float32)
    if imagej_ok:
        imwrite(path, arr, imagej=True)
    else:
        imwrite(path, arr)


def get_scene_names(image_path: str | Path) -> list[str]:
    from .image_io import get_scene_names as _get_scene_names

    return _get_scene_names(image_path)


def load_scene_channels(image_path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    from .image_io import load_scene_channels as _load_scene_channels

    return _load_scene_channels(image_path, scene_index_zero_based)

def effective_n_jobs(params: AnalysisParameters, n_tasks: int | None = None) -> int:
    """Resolve the requested parallel worker count.

    parallel_n_jobs <= 0 means automatic: use all visible CPU cores, capped by the
    number of tasks when known. The value is never less than 1.
    """
    try:
        requested = int(params.parallel_n_jobs)
    except Exception:
        requested = 0
    if requested > 0:
        n = requested
    else:
        n = int(os.cpu_count() or 1)
    if n_tasks is not None and n_tasks > 0:
        n = min(n, int(n_tasks))
    return max(1, int(n))


def _parallel_backend_name(params: AnalysisParameters) -> str:
    backend = str(getattr(params, "parallel_backend", "threading") or "threading").lower()
    if backend not in {"threading", "loky", "sequential"}:
        backend = "threading"
    return backend


def _parallel_map(func, items: Sequence, params: AnalysisParameters, progress: ProgressCallback = None, task_name: str = "tasks"):
    items = list(items)
    n_tasks = len(items)
    n_jobs = effective_n_jobs(params, n_tasks)
    backend = _parallel_backend_name(params)
    if n_tasks == 0:
        return []
    if n_jobs <= 1 or backend == "sequential" or Parallel is None or delayed is None:
        return [func(item) for item in items]
    log(f"Running {task_name} in parallel: {n_tasks} task(s), {n_jobs} worker(s), backend={backend}", progress)
    prefer = "threads" if backend == "threading" else None
    return Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer)(delayed(func)(item) for item in items)


def _relabel_by_min_size(labels: np.ndarray, min_vox: int) -> np.ndarray:
    """Filter labels by voxel count and relabel densely using a vectorized lookup table."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0 or int(labels.max(initial=0)) <= 0:
        return np.zeros_like(labels, dtype=np.int32)
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero(counts >= int(max(1, min_vox)))
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros_like(labels, dtype=np.int32)
    lut = np.zeros(counts.shape[0], dtype=np.int32)
    lut[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return lut[labels]


def _filter_labels_by_volume(labels: np.ndarray, min_volume_um3: float, spacing_zyx: SpacingZYX) -> np.ndarray:
    voxel_um3 = float(np.prod(spacing_zyx))
    min_vox = max(1, int(round(float(min_volume_um3) / max(voxel_um3, EPS))))
    return _relabel_by_min_size(labels, min_vox)

def normalize_uint8(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    mn = float(np.nanmin(vol)) if vol.size else 0.0
    mx = float(np.nanmax(vol)) if vol.size else 0.0
    if not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(vol, dtype=np.uint8)
    out = (vol - mn) / (mx - mn)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def compute_foreground_bbox_3d(vol: np.ndarray, pad_xy: int = 32, progress: ProgressCallback = None):
    """Estimate a tissue/foreground XY bounding box from the nucleus channel."""
    log("Estimating foreground bounding box from the nucleus channel", progress)
    sm = gaussian(vol.astype(np.float32), sigma=(0.5, 2, 2), preserve_range=True)
    proj = sm.max(axis=0)
    try:
        thr = threshold_otsu(proj) if np.unique(proj).size > 1 else float(np.mean(proj))
    except Exception:
        thr = float(np.mean(proj))
    fg2d = proj > thr
    fg2d = binary_opening_compat(fg2d, np.ones((3, 3), dtype=bool))
    fg2d = binary_closing_compat(fg2d, np.ones((9, 9), dtype=bool))
    rows = np.any(fg2d, axis=1)
    cols = np.any(fg2d, axis=0)
    if not rows.any() or not cols.any():
        log("No foreground bounding box found; using full volume", progress)
        return slice(0, vol.shape[0]), slice(0, vol.shape[1]), slice(0, vol.shape[2])

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0 = max(0, int(y0) - int(pad_xy))
    y1 = min(vol.shape[1], int(y1) + int(pad_xy) + 1)
    x0 = max(0, int(x0) - int(pad_xy))
    x1 = min(vol.shape[2], int(x1) + int(pad_xy) + 1)
    log(f"Foreground bbox zyx: z(0:{vol.shape[0]}) y({y0}:{y1}) x({x0}:{x1})", progress)
    return slice(0, vol.shape[0]), slice(y0, y1), slice(x0, x1)


def downsample_zyx(vol: np.ndarray, xy_factor: float) -> np.ndarray:
    xy_factor = max(1.0, float(xy_factor))
    if xy_factor == 1.0:
        return vol
    z, y, x = vol.shape
    ny = max(1, int(round(y / xy_factor)))
    nx = max(1, int(round(x / xy_factor)))
    out = resize(vol, (z, ny, nx), order=1, preserve_range=True, anti_aliasing=True)
    return out.astype(vol.dtype)


def upsample_labels(lbl: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    out = resize(lbl, tuple(target_shape), order=0, preserve_range=True, anti_aliasing=False)
    return np.rint(out).astype(np.int32)


def _make_cellpose_model(params: AnalysisParameters):
    from cellpose import models

    try:
        return models.Cellpose(gpu=params.gpu, model_type=params.cellpose_model)
    except Exception:
        return models.CellposeModel(gpu=params.gpu, model_type=params.cellpose_model)


def _cellpose_eval_filtered(model, image, eval_kwargs: dict):
    """Call Cellpose eval while filtering keyword arguments for version compatibility.

    Cellpose versions differ in the exact keyword arguments accepted by ``eval``.
    This helper filters unsupported keys before calling Cellpose. For 3D ZYX
    grayscale stacks, it also supplies ``z_axis=0`` when the installed Cellpose
    version supports that keyword. This protects users who accidentally run with
    newer Cellpose releases, while the package dependency still pins Cellpose to
    ``>=2,<4`` for the validated workflow.
    """
    import inspect

    sig = inspect.signature(model.eval)
    accepts_var_kwargs = any(
        par.kind == inspect.Parameter.VAR_KEYWORD for par in sig.parameters.values()
    )
    if accepts_var_kwargs:
        filtered = dict(eval_kwargs)
    else:
        filtered = {k: v for k, v in eval_kwargs.items() if k in sig.parameters}

    image_arr = np.asarray(image)
    if bool(filtered.get("do_3D", False)) and image_arr.ndim == 3:
        if "z_axis" in sig.parameters:
            filtered.setdefault("z_axis", 0)
        if "channel_axis" in sig.parameters:
            filtered.setdefault("channel_axis", None)

    try:
        result = model.eval(image, **filtered)
    except TypeError:
        # Last-resort fallback for Cellpose wrappers with non-introspectable
        # signatures. Remove known legacy-only keys and retry once.
        for legacy_key in ("net_avg",):
            filtered.pop(legacy_key, None)
        result = model.eval(image, **filtered)

    if isinstance(result, tuple):
        return result[0]

    if isinstance(result, list):
        # If this is already a list of 2D/3D mask arrays, return it as-is.
        # This protects batched 2D Cellpose calls on older releases.
        if result and all(isinstance(r, np.ndarray) and r.ndim in (2, 3) for r in result):
            return result
        # Otherwise treat it like a tuple-style Cellpose return container.
        if len(result) > 0:
            return result[0]
    return result


def _eval_cellpose_model(model, image_u8: np.ndarray, anisotropy: float, params: AnalysisParameters):
    eval_kwargs = dict(
        channels=[0, 0],
        do_3D=True,
        z_axis=0,
        channel_axis=None,
        anisotropy=float(anisotropy),
        diameter=float(params.diameter),
        stitch_threshold=float(params.stitch_threshold),
        cellprob_threshold=float(params.cellprob_threshold),
        flow_threshold=float(params.flow_threshold),
        min_size=0,
        augment=False,
        net_avg=False,
        batch_size=int(params.cellpose_batch_size),
    )
    return np.asarray(_cellpose_eval_filtered(model, image_u8, eval_kwargs))


def _normalize_cellpose_2d_masks(masks, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """Convert Cellpose 2D output into a ZYX label stack.

    Older Cellpose releases can misinterpret a raw ``(Z, Y, X)`` ndarray passed
    to a 2D run as a single multichannel image, yielding a wrong shape such as
    ``(Z, Y)``. The caller avoids that by passing a list of ``YX`` slices; this
    normalizer still handles several return formats defensively.
    """
    z_n, y_n, x_n = map(int, expected_shape)

    if isinstance(masks, list):
        if len(masks) != z_n:
            raise ValueError(
                f"Cellpose 2D returned a list with {len(masks)} masks; expected {z_n}."
            )
        arr = np.stack([np.asarray(m) for m in masks], axis=0)
    else:
        arr = np.asarray(masks)
        if arr.dtype == object:
            arr = np.stack([np.asarray(m) for m in list(arr)], axis=0)

    if arr.ndim == 2 and z_n == 1 and tuple(arr.shape) == (y_n, x_n):
        arr = arr[None, ...]

    # Some readers/wrappers return YXZ for a list of 2D images.
    if arr.ndim == 3 and tuple(arr.shape) == (y_n, x_n, z_n):
        arr = np.moveaxis(arr, -1, 0)

    if arr.ndim != 3 or tuple(arr.shape) != (z_n, y_n, x_n):
        raise ValueError(
            f"Cellpose 2D returned masks with shape {tuple(arr.shape)}; "
            f"expected ZYX {tuple(expected_shape)}."
        )
    return arr.astype(np.int32, copy=False)


def _eval_cellpose_2d_batch(model, images_u8: np.ndarray, params: AnalysisParameters) -> np.ndarray:
    """Run Cellpose on a Z stack as independent 2D images.

    Important compatibility detail: for Cellpose 2.x, passing a raw ZYX ndarray
    to a 2D evaluation can be interpreted as one image instead of a batch of Z
    images. Therefore we pass a Python list of ``YX`` slices. Cellpose still uses
    its internal ``batch_size`` for neural-network inference, so this remains the
    fast batched path while preserving the correct output shape.
    """
    images_u8 = np.asarray(images_u8)
    if images_u8.ndim != 3:
        raise ValueError(f"Expected ZYX input for 2D Cellpose, got {images_u8.shape}.")

    eval_kwargs = dict(
        channels=[0, 0],
        do_3D=False,
        diameter=float(params.diameter),
        stitch_threshold=0.0,
        cellprob_threshold=float(params.cellprob_threshold),
        flow_threshold=float(params.flow_threshold),
        min_size=0,
        augment=False,
        net_avg=False,
        batch_size=int(params.cellpose_batch_size),
        channel_axis=None,
        z_axis=None,
    )

    slice_list = [np.ascontiguousarray(images_u8[z]) for z in range(images_u8.shape[0])]
    try:
        masks = _cellpose_eval_filtered(model, slice_list, eval_kwargs)
        return _normalize_cellpose_2d_masks(masks, tuple(images_u8.shape))
    except Exception as batch_exc:
        # Extremely defensive fallback: some older Cellpose builds behave poorly
        # for list input. Running one slice at a time is slower but avoids a hard
        # failure and gives a useful warning in the terminal.
        warnings.warn(
            "Batched 2D Cellpose returned an unexpected output; falling back to "
            f"per-slice evaluation. Original error: {batch_exc}",
            RuntimeWarning,
        )
        per_slice = []
        for z, slc in enumerate(slice_list):
            one = _cellpose_eval_filtered(model, slc, eval_kwargs)
            one_arr = np.asarray(one)
            if one_arr.ndim == 3 and one_arr.shape[0] == 1:
                one_arr = one_arr[0]
            if one_arr.ndim != 2:
                raise ValueError(
                    f"Cellpose 2D fallback returned shape {tuple(one_arr.shape)} "
                    f"for z={z}; expected YX."
                ) from batch_exc
            per_slice.append(one_arr)
        return _normalize_cellpose_2d_masks(per_slice, tuple(images_u8.shape))


def _cellpose_2d_worker(images_u8: np.ndarray, params_dict: dict) -> np.ndarray:
    """Process-pool worker for independent 2D Cellpose chunks."""
    params = AnalysisParameters(**params_dict)
    model = _make_cellpose_model(params)
    return _eval_cellpose_2d_batch(model, np.asarray(images_u8), params)


def _stitch_2d_labels_overlap(label_stack: np.ndarray, overlap_threshold: float = 0.10) -> np.ndarray:
    """Stitch per-slice 2D Cellpose labels into 3D objects by adjacent-slice overlap.

    Each 2D slice is first treated independently. Labels in adjacent slices are joined
    if their overlap divided by the smaller 2D object area is at least overlap_threshold.
    """
    labels = np.asarray(label_stack, dtype=np.int32)
    if labels.ndim != 3 or labels.size == 0 or int(labels.max(initial=0)) <= 0:
        return np.zeros_like(labels, dtype=np.int32)

    z_n = labels.shape[0]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    areas: dict[tuple[int, int], int] = {}
    for z in range(z_n):
        vals, counts = np.unique(labels[z], return_counts=True)
        for val, count in zip(vals, counts):
            lab = int(val)
            if lab > 0:
                key = (z, lab)
                parent[key] = key
                areas[key] = int(count)

    def find(a: tuple[int, int]) -> tuple[int, int]:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != a:
            nxt = parent[a]
            parent[a] = root
            a = nxt
        return root

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    threshold = float(max(0.0, overlap_threshold))
    for z in range(z_n - 1):
        a = labels[z]
        b = labels[z + 1]
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            continue
        max_b = int(b.max(initial=0)) + 1
        combined = a[mask].astype(np.int64) * max_b + b[mask].astype(np.int64)
        pair_ids, counts = np.unique(combined, return_counts=True)
        for pair_id, count in zip(pair_ids, counts):
            lab_a = int(pair_id // max_b)
            lab_b = int(pair_id % max_b)
            key_a = (z, lab_a)
            key_b = (z + 1, lab_b)
            if key_a not in areas or key_b not in areas:
                continue
            frac = float(count) / max(1.0, float(min(areas[key_a], areas[key_b])))
            if frac >= threshold:
                union(key_a, key_b)

    root_to_id: dict[tuple[int, int], int] = {}
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for z in range(z_n):
        vals = np.unique(labels[z])
        for val in vals:
            lab = int(val)
            if lab <= 0:
                continue
            root = find((z, lab))
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
            out[z][labels[z] == lab] = root_to_id[root]
    return out


def _segment_nuclei_cellpose_2d_stack(ds_u8: np.ndarray, params: AnalysisParameters, progress: ProgressCallback = None) -> np.ndarray:
    model = _make_cellpose_model(params)
    log(f"Running Cellpose as independent 2D slices on {tuple(ds_u8.shape)}; batch_size={params.cellpose_batch_size}", progress)
    masks2d = _eval_cellpose_2d_batch(model, ds_u8, params)
    return _stitch_2d_labels_overlap(masks2d, overlap_threshold=float(params.cellpose_stitch_overlap))


def _segment_nuclei_cellpose_2d_parallel(ds_u8: np.ndarray, params: AnalysisParameters, progress: ProgressCallback = None) -> np.ndarray:
    if bool(params.gpu):
        log("GPU is enabled; using 2D stack-batch Cellpose instead of multiple GPU processes.", progress)
        return _segment_nuclei_cellpose_2d_stack(ds_u8, params, progress=progress)
    n_jobs = effective_n_jobs(params, ds_u8.shape[0])
    if n_jobs <= 1 or Parallel is None or delayed is None:
        return _segment_nuclei_cellpose_2d_stack(ds_u8, params, progress=progress)
    indices = np.array_split(np.arange(ds_u8.shape[0]), n_jobs)
    chunks = [(idx, ds_u8[idx]) for idx in indices if len(idx)]
    log(f"Running Cellpose 2D in parallel: {len(chunks)} process(es), {ds_u8.shape[0]} z-slice(s)", progress)
    params_dict = asdict(params)
    results = Parallel(n_jobs=len(chunks), backend="loky")(
        delayed(_cellpose_2d_worker)(chunk, params_dict) for _idx, chunk in chunks
    )
    masks2d = np.zeros_like(ds_u8, dtype=np.int32)
    for (idx, _chunk), masks in zip(chunks, results):
        masks2d[idx] = np.asarray(masks, dtype=np.int32)
    return _stitch_2d_labels_overlap(masks2d, overlap_threshold=float(params.cellpose_stitch_overlap))


def segment_nuclei_cellpose_3d(
    nuc_vol: np.ndarray,
    spacing_zyx: SpacingZYX,
    params: AnalysisParameters,
    progress: ProgressCallback = None,
) -> np.ndarray:
    """Run Cellpose nuclei segmentation and filter by physical volume.

    Three modes are available: original 3D Cellpose, batched 2D slice Cellpose with
    overlap-based 3D stitching, and process-parallel 2D slice Cellpose with stitching.
    """
    start = time.time()
    bbox = compute_foreground_bbox_3d(nuc_vol, pad_xy=params.bbox_pad_xy, progress=progress)
    cropped = nuc_vol[bbox]
    ds = downsample_zyx(cropped.astype(np.float32), params.xy_downsample)
    ds_u8 = normalize_uint8(ds)

    mode = str(getattr(params, "cellpose_segmentation_mode", "Cellpose 3D whole volume"))
    z_um, y_um, _x_um = spacing_zyx
    anisotropy = z_um / (y_um * max(1.0, float(params.xy_downsample))) if y_um > 0 else 1.0
    log(
        f"Nuclei segmentation mode: {mode}; input={tuple(ds_u8.shape)}, diameter={params.diameter:g}, "
        f"anisotropy={anisotropy:.3f}",
        progress,
    )

    if "2D parallel" in mode:
        masks = _segment_nuclei_cellpose_2d_parallel(ds_u8, params, progress=progress)
    elif "2D stack" in mode:
        masks = _segment_nuclei_cellpose_2d_stack(ds_u8, params, progress=progress)
    else:
        model = _make_cellpose_model(params)
        masks = _eval_cellpose_model(model, ds_u8, anisotropy, params)

    masks = np.asarray(masks, dtype=np.int32)
    if masks.ndim != 3:
        raise ValueError(f"Cellpose returned masks with shape {masks.shape}; expected 3D ZYX labels.")
    if tuple(masks.shape) != tuple(cropped.shape):
        masks = upsample_labels(masks, cropped.shape)

    full = np.zeros(nuc_vol.shape, dtype=np.int32)
    full[bbox] = masks.astype(np.int32, copy=False)
    out = _filter_labels_by_volume(full, params.min_nucleus_volume_um3, spacing_zyx)
    kept = int(out.max(initial=0))
    log(f"Kept {kept} nuclei after filtering; nuclei step took {time.time() - start:.1f} s", progress)
    return out


# -----------------------------------------------------------------------------
# Context-aware P/Q arm segmentation helpers
# -----------------------------------------------------------------------------


def _smooth_arm_image(arm_img: np.ndarray, sigma: float) -> np.ndarray:
    arm_img = np.asarray(arm_img, dtype=np.float32)
    if sigma and sigma > 0:
        return gaussian(arm_img, sigma=float(sigma), preserve_range=True).astype(np.float32)
    return arm_img


def _finite_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).ravel()
    return arr[np.isfinite(arr)]


def _sample_values(values: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    arr = _finite_1d(values)
    if arr.size <= max_n or max_n <= 0:
        return arr
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(arr.size, size=int(max_n), replace=False)
    return arr[idx]


def _median_mad(values: np.ndarray) -> tuple[float, float, int]:
    arr = _finite_1d(values)
    if arr.size == 0:
        return 0.0, 1.0, 0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= EPS:
        robust_sigma = float(np.std(arr))
    if not np.isfinite(robust_sigma) or robust_sigma <= EPS:
        robust_sigma = 1.0
    return med, robust_sigma, int(arr.size)


def _low_percentile_values(values: np.ndarray, percentile: float) -> np.ndarray:
    arr = _finite_1d(values)
    if arr.size == 0:
        return arr
    p = float(np.clip(percentile, 0.1, 99.9))
    thr = float(np.percentile(arr, p))
    low = arr[arr <= thr]
    return low if low.size >= 32 else arr




def _new_gmm(n_components: int, covariance_type: str = "full", random_state: int = 0) -> GaussianMixture:
    """Create a GaussianMixture with faster initialization while retaining old sklearn compatibility."""
    kwargs = dict(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        random_state=int(random_state),
        n_init=1,
        max_iter=100,
        init_params="k-means++",
    )
    try:
        return GaussianMixture(**kwargs)
    except TypeError:
        kwargs.pop("init_params", None)
        kwargs.pop("n_init", None)
        kwargs.pop("max_iter", None)
        return GaussianMixture(**kwargs)

def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-3)
    z = (np.asarray(x, dtype=np.float32) - float(mu)) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    out = 1.0 / (1.0 + np.exp(-np.clip(x_arr, -60.0, 60.0)))
    if np.ndim(x) == 0:
        return float(out)
    return out.astype(np.float32)


def _background_sample(arm_img: np.ndarray, nuclei_labels: np.ndarray, params: AnalysisParameters) -> tuple[np.ndarray, str]:
    mode = str(params.field_normalization_mode)
    img = np.asarray(arm_img, dtype=np.float32)
    nuclei = np.asarray(nuclei_labels) > 0
    outside = img[~nuclei] if img.shape == nuclei.shape else np.array([], dtype=np.float32)
    inside = img[nuclei] if img.shape == nuclei.shape else img.ravel()

    if mode == "none":
        return np.array([0.0, 1.0], dtype=np.float32), "none"
    if mode == "outside_nuclei":
        sample = _finite_1d(outside)
        if sample.size >= 128:
            return sample, "outside_nuclei"
        return _low_percentile_values(img, params.background_percentile), "outside_nuclei_fallback_whole_low_percentile"
    if mode == "nuclear_low_percentile":
        return _low_percentile_values(inside, params.background_percentile), "nuclear_low_percentile"
    if mode == "whole_image_low_percentile":
        return _low_percentile_values(img, params.background_percentile), "whole_image_low_percentile"

    # mixed_outside_or_low_percentile: prefer outside-nucleus background if enough voxels exist.
    sample = _finite_1d(outside)
    if sample.size >= 128:
        return sample, "outside_nuclei"
    return _low_percentile_values(inside, params.background_percentile), "nuclear_low_percentile_fallback"


def _fit_two_component_prior(norm_img: np.ndarray, nuclei_labels: np.ndarray, params: AnalysisParameters) -> dict:
    nuclei = np.asarray(nuclei_labels) > 0
    vals = _sample_values(norm_img[nuclei], int(params.max_context_sample_voxels), seed=int(params.gmm_random_state))
    vals = vals[np.isfinite(vals)]
    if vals.size < 64 or np.unique(vals).size < 8:
        return {
            "prior_mu_bg": 0.0,
            "prior_sigma_bg": 1.0,
            "prior_mu_sig": 3.0,
            "prior_sigma_sig": 1.0,
            "prior_pi_sig": 0.05,
            "prior_delta_bic": 0.0,
            "prior_n_voxels": int(vals.size),
        }

    x = vals.reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g1 = _new_gmm(1, "full", int(params.gmm_random_state)).fit(x)
            g2 = _new_gmm(2, "full", int(params.gmm_random_state)).fit(x)
        delta_bic = float(g1.bic(x) - g2.bic(x))
        means = g2.means_.ravel()
        order = np.argsort(means)
        bg_idx = int(order[0])
        sig_idx = int(order[-1])
        cov = np.asarray(g2.covariances_)
        if cov.ndim == 3:
            var_bg = float(cov[bg_idx, 0, 0])
            var_sig = float(cov[sig_idx, 0, 0])
        else:
            var_bg = float(np.ravel(cov)[bg_idx])
            var_sig = float(np.ravel(cov)[sig_idx])
        return {
            "prior_mu_bg": float(means[bg_idx]),
            "prior_sigma_bg": float(max(math.sqrt(max(var_bg, EPS)), 1.0e-3)),
            "prior_mu_sig": float(means[sig_idx]),
            "prior_sigma_sig": float(max(math.sqrt(max(var_sig, EPS)), 1.0e-3)),
            "prior_pi_sig": float(np.clip(g2.weights_[sig_idx], 1.0e-4, 0.95)),
            "prior_delta_bic": delta_bic,
            "prior_n_voxels": int(vals.size),
        }
    except Exception:
        return {
            "prior_mu_bg": 0.0,
            "prior_sigma_bg": 1.0,
            "prior_mu_sig": float(np.percentile(vals, 99.0)),
            "prior_sigma_sig": 1.0,
            "prior_pi_sig": 0.05,
            "prior_delta_bic": 0.0,
            "prior_n_voxels": int(vals.size),
        }


def _load_external_prior(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _apply_external_prior(stats: ArmFieldStats, external: dict) -> ArmFieldStats:
    if not external:
        return stats
    key = stats.arm_name.lower()
    candidate = None
    for k in [key, key.upper(), f"{key}_arm", f"{key.upper()}_arm"]:
        if isinstance(external.get(k), dict):
            candidate = external[k]
            break
    if candidate is None:
        # Also accept a flat prior JSON for a single arm.
        if any(k in external for k in ["prior_mu_bg", "mu_bg", "prior_mu_sig", "mu_sig"]):
            candidate = external
    if not isinstance(candidate, dict):
        return stats

    mapping = {
        "prior_mu_bg": ["prior_mu_bg", "mu_bg", "background_mean"],
        "prior_sigma_bg": ["prior_sigma_bg", "sigma_bg", "background_sigma"],
        "prior_mu_sig": ["prior_mu_sig", "mu_sig", "signal_mean"],
        "prior_sigma_sig": ["prior_sigma_sig", "sigma_sig", "signal_sigma"],
        "prior_pi_sig": ["prior_pi_sig", "pi_sig", "signal_prior"],
        "prior_delta_bic": ["prior_delta_bic", "delta_bic"],
    }
    for field_name, keys in mapping.items():
        for k in keys:
            if k in candidate:
                try:
                    setattr(stats, field_name, float(candidate[k]))
                except Exception:
                    pass
                break
    stats.prior_source = f"external:{Path(str(external.get('source_path', 'prior_json'))).name}"
    return stats


def build_arm_contexts(
    p_img: np.ndarray,
    q_img: np.ndarray,
    nuclei_labels: np.ndarray,
    params: AnalysisParameters,
    progress: ProgressCallback = None,
) -> dict[str, ArmRuntimeContext]:
    """Build field-normalized images and population priors for P and Q channels."""
    external = _load_external_prior(params.external_prior_json) if params.use_external_prior_if_available else {}
    if external and isinstance(external, dict):
        external.setdefault("source_path", str(params.external_prior_json))

    contexts: dict[str, ArmRuntimeContext] = {}
    for name, raw in [("p", p_img), ("q", q_img)]:
        raw = _smooth_arm_image(raw, params.arm_smoothing_sigma)
        if str(params.field_normalization_mode) == "none":
            med, mad, n_bg = 0.0, 1.0, int(raw.size)
            actual_mode = "none"
            norm = raw.astype(np.float32)
        else:
            bg_sample, actual_mode = _background_sample(raw, nuclei_labels, params)
            bg_sample = _sample_values(bg_sample, int(params.max_context_sample_voxels), seed=int(params.gmm_random_state))
            med, mad, n_bg = _median_mad(bg_sample)
            norm = ((raw.astype(np.float32) - med) / (mad + EPS)).astype(np.float32)

        prior = _fit_two_component_prior(norm, nuclei_labels, params)
        stats = ArmFieldStats(
            arm_name=name,
            normalization_mode=actual_mode,
            background_median_raw=float(med),
            background_mad_raw=float(mad),
            background_n_voxels=int(n_bg),
            prior_mu_bg=float(prior["prior_mu_bg"]),
            prior_sigma_bg=float(prior["prior_sigma_bg"]),
            prior_mu_sig=float(prior["prior_mu_sig"]),
            prior_sigma_sig=float(prior["prior_sigma_sig"]),
            prior_pi_sig=float(prior["prior_pi_sig"]),
            prior_delta_bic=float(prior["prior_delta_bic"]),
            prior_source="current_field",
            prior_n_voxels=int(prior["prior_n_voxels"]),
        )
        stats = _apply_external_prior(stats, external)
        contexts[name] = ArmRuntimeContext(name, raw, norm, stats)
        log(
            f"{name.upper()} context: bg median={stats.background_median_raw:.3g}, MAD/sigma={stats.background_mad_raw:.3g}, "
            f"prior bg/sig means={stats.prior_mu_bg:.2f}/{stats.prior_mu_sig:.2f}, source={stats.prior_source}",
            progress,
        )
    return contexts


def _method_flags(params: AnalysisParameters) -> dict[str, bool]:
    """Return behavior flags for the streamlined method set.

    The GUI intentionally exposes only three arm-detection modes in this
    version: legacy intensity-only GMM, upgraded GMM with presence gating
    and component scoring, and MRF/CRF refinement. The older experimental
    spatial, hierarchical-prior, and classifier methods remain absent from
    the selector and are forced off here so they cannot be activated by stale
    saved parameter files.
    """
    m = str(params.arm_detection_method).lower()
    legacy = "legacy" in m
    mrf = (("mrf" in m) or ("crf" in m) or bool(params.use_mrf_refinement)) and not legacy
    upgraded = (("upgraded" in m) or ("gate" in m) or mrf) and not legacy
    return {
        "legacy": bool(legacy),
        "upgraded": bool(upgraded),
        "spatial": False,
        "mrf": bool(mrf),
        "hierarchical": False,
        "classifier": False,
        "hysteresis": False,
        "presence_gate": bool(params.enable_presence_gate) and not legacy,
    }


def _fit_gmm(values: np.ndarray, params: AnalysisParameters, max_components: int | None = None) -> Optional[GaussianMixture]:
    vals = _finite_1d(values)
    vals = _sample_values(vals, int(params.max_voxels_per_nucleus_fit), seed=int(params.gmm_random_state))
    if vals.size < 8 or np.unique(vals).size < 2:
        return None
    max_k = max(1, int(max_components if max_components is not None else params.gmm_components))
    max_k = min(max_k, int(vals.size), int(np.unique(vals).size))
    if max_k <= 1:
        return None
    x = vals.reshape(-1, 1)
    if bool(params.auto_choose_gmm_components_by_bic):
        best_gmm = None
        best_bic = float("inf")
        for k in range(1, max_k + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = _new_gmm(k, params.gmm_covariance_type, int(params.gmm_random_state)).fit(x)
                bic = float(g.bic(x))
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = g
            except Exception:
                continue
        if best_gmm is None or best_gmm.n_components <= 1:
            return None
        return best_gmm
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _new_gmm(max_k, params.gmm_covariance_type, int(params.gmm_random_state)).fit(x)
    except Exception:
        return None


def _bic_delta_1_vs_alt(values: np.ndarray, alt_gmm: Optional[GaussianMixture], params: AnalysisParameters) -> float:
    vals = _finite_1d(values)
    vals = _sample_values(vals, int(params.max_voxels_per_nucleus_fit), seed=int(params.gmm_random_state))
    if alt_gmm is None or vals.size < 8 or np.unique(vals).size < 2:
        return 0.0
    x = vals.reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g1 = _new_gmm(1, "full", int(params.gmm_random_state)).fit(x)
        return float(g1.bic(x) - alt_gmm.bic(x))
    except Exception:
        return 0.0


def _gmm_probability_1d(
    norm_img_sub: np.ndarray,
    nucleus_mask: np.ndarray,
    min_sorted_class: int,
    params: AnalysisParameters,
) -> tuple[np.ndarray, dict]:
    vals_all = np.asarray(norm_img_sub[nucleus_mask], dtype=np.float32)
    prob = np.zeros_like(norm_img_sub, dtype=np.float32)
    metrics = {
        "gmm_n_components_used": 0,
        "gmm_delta_bic": 0.0,
        "gmm_means_sorted": "",
    }
    finite = np.isfinite(vals_all)
    vals = vals_all[finite]
    if vals.size < 8 or np.unique(vals).size < 2:
        return prob, metrics

    gmm = _fit_gmm(vals, params)
    if gmm is None or gmm.n_components <= 1:
        return prob, metrics
    metrics["gmm_n_components_used"] = int(gmm.n_components)
    metrics["gmm_delta_bic"] = _bic_delta_1_vs_alt(vals, gmm, params)
    means = gmm.means_.ravel()
    order = np.argsort(means)
    sorted_index = np.zeros_like(order)
    sorted_index[order] = np.arange(gmm.n_components)
    threshold_idx = int(np.clip(min_sorted_class, 0, gmm.n_components - 1))
    signal_components = np.where(sorted_index >= threshold_idx)[0]
    metrics["gmm_means_sorted"] = ";".join(f"{x:.4g}" for x in means[order])

    post = gmm.predict_proba(vals.reshape(-1, 1))
    p_vals = np.sum(post[:, signal_components], axis=1) if signal_components.size else np.zeros(vals.shape, dtype=np.float32)
    full_vals_prob = np.zeros(vals_all.shape, dtype=np.float32)
    full_vals_prob[finite] = p_vals.astype(np.float32)
    prob[nucleus_mask] = full_vals_prob
    return np.clip(prob, 0.0, 1.0), metrics


def _spatial_feature_image(norm_img_sub: np.ndarray, nucleus_mask: np.ndarray, params: AnalysisParameters) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(nucleus_mask)
    if coords.size == 0:
        return np.empty((0, 1), dtype=np.float32), coords

    vals = np.asarray(norm_img_sub[nucleus_mask], dtype=np.float32)
    finite = np.isfinite(vals)
    coords = coords[finite]
    vals = vals[finite]
    if vals.size == 0:
        return np.empty((0, 1), dtype=np.float32), coords

    zyx_min = coords.min(axis=0).astype(np.float32)
    zyx_max = coords.max(axis=0).astype(np.float32)
    span = np.maximum(zyx_max - zyx_min, 1.0)
    norm_coords = ((coords.astype(np.float32) - zyx_min) / span - 0.5) * 2.0

    dist = ndi.distance_transform_edt(nucleus_mask)
    dvals = dist[tuple(coords.T)].astype(np.float32)
    dmax = float(np.max(dvals)) if dvals.size else 1.0
    radial = dvals / max(dmax, EPS)

    local_mean = gaussian(norm_img_sub.astype(np.float32), sigma=1.0, preserve_range=True)
    contrast = (norm_img_sub - local_mean)[tuple(coords.T)].astype(np.float32)
    c_med, c_mad, _ = _median_mad(contrast)
    contrast = np.clip((contrast - c_med) / (c_mad + EPS), -5.0, 5.0)

    grad = ndi.gaussian_gradient_magnitude(norm_img_sub.astype(np.float32), sigma=1.0)
    gvals = grad[tuple(coords.T)].astype(np.float32)
    g_scale = float(np.percentile(gvals, 95.0)) if gvals.size else 1.0
    gvals = np.clip(gvals / max(g_scale, EPS), 0.0, 5.0)

    features = [vals.reshape(-1, 1)]
    if params.spatial_coordinate_weight > 0:
        features.append(norm_coords * float(params.spatial_coordinate_weight))
    if params.spatial_radial_weight > 0:
        features.append(radial.reshape(-1, 1) * float(params.spatial_radial_weight))
    if params.spatial_contrast_weight > 0:
        features.append(contrast.reshape(-1, 1) * float(params.spatial_contrast_weight))
    if params.spatial_gradient_weight > 0:
        features.append(gvals.reshape(-1, 1) * float(params.spatial_gradient_weight))
    return np.hstack(features).astype(np.float32), coords


def _gmm_probability_spatial(
    norm_img_sub: np.ndarray,
    nucleus_mask: np.ndarray,
    min_sorted_class: int,
    params: AnalysisParameters,
) -> tuple[np.ndarray, dict]:
    prob = np.zeros_like(norm_img_sub, dtype=np.float32)
    metrics = {
        "spatial_gmm_n_components_used": 0,
        "spatial_gmm_means_sorted": "",
        "spatial_gmm_delta_bic": 0.0,
    }
    features, coords = _spatial_feature_image(norm_img_sub, nucleus_mask, params)
    if features.shape[0] < 8 or np.unique(features[:, 0]).size < 2:
        return prob, metrics

    fit_features = features
    if params.max_voxels_per_nucleus_fit > 0 and fit_features.shape[0] > params.max_voxels_per_nucleus_fit:
        rng = np.random.default_rng(int(params.gmm_random_state))
        idx = rng.choice(fit_features.shape[0], size=int(params.max_voxels_per_nucleus_fit), replace=False)
        fit_features = fit_features[idx]
    max_k = min(max(2, int(params.gmm_components)), fit_features.shape[0], int(np.unique(fit_features[:, 0]).size))
    if max_k <= 1:
        return prob, metrics
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = _new_gmm(max_k, params.gmm_covariance_type, int(params.gmm_random_state)).fit(fit_features)
    except Exception:
        return prob, metrics

    post = gmm.predict_proba(features)
    intensity_means = gmm.means_[:, 0]
    order = np.argsort(intensity_means)
    sorted_index = np.zeros_like(order)
    sorted_index[order] = np.arange(gmm.n_components)
    threshold_idx = int(np.clip(min_sorted_class, 0, gmm.n_components - 1))
    signal_components = np.where(sorted_index >= threshold_idx)[0]
    p_vals = np.sum(post[:, signal_components], axis=1) if signal_components.size else np.zeros(features.shape[0])
    prob[tuple(coords.T)] = p_vals.astype(np.float32)
    metrics["spatial_gmm_n_components_used"] = int(gmm.n_components)
    metrics["spatial_gmm_means_sorted"] = ";".join(f"{x:.4g}" for x in intensity_means[order])
    # A true BIC delta for the spatial model is not the same as the 1D null model,
    # but this is still useful as a relative model-fit diagnostic.
    try:
        g1 = _new_gmm(1, params.gmm_covariance_type, int(params.gmm_random_state)).fit(fit_features)
        metrics["spatial_gmm_delta_bic"] = float(g1.bic(fit_features) - gmm.bic(fit_features))
    except Exception:
        metrics["spatial_gmm_delta_bic"] = 0.0
    return np.clip(prob, 0.0, 1.0), metrics


def _hierarchical_prior_probability(norm_img_sub: np.ndarray, nucleus_mask: np.ndarray, stats: ArmFieldStats) -> np.ndarray:
    prob = np.zeros_like(norm_img_sub, dtype=np.float32)
    vals = np.asarray(norm_img_sub[nucleus_mask], dtype=np.float32)
    finite = np.isfinite(vals)
    if vals.size == 0:
        return prob
    pi = float(np.clip(stats.prior_pi_sig, 1.0e-4, 0.95))
    p_bg = _normal_pdf(vals[finite], stats.prior_mu_bg, stats.prior_sigma_bg)
    p_sig = _normal_pdf(vals[finite], stats.prior_mu_sig, stats.prior_sigma_sig)
    p = (pi * p_sig) / ((pi * p_sig) + ((1.0 - pi) * p_bg) + EPS)
    vals_prob = np.zeros(vals.shape, dtype=np.float32)
    vals_prob[finite] = p.astype(np.float32)
    prob[nucleus_mask] = vals_prob
    return np.clip(prob, 0.0, 1.0)


def _combine_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    valid = [np.clip(np.asarray(p, dtype=np.float32), 0.0, 1.0) for p in probabilities if p is not None]
    if not valid:
        return np.zeros((1,), dtype=np.float32)
    # Noisy-OR combines multiple independent evidence sources without requiring all to be high.
    prod_not = np.ones_like(valid[0], dtype=np.float32)
    for p in valid:
        prod_not *= (1.0 - p)
    return np.clip(1.0 - prod_not, 0.0, 1.0)


def _hysteresis_mask(prob: np.ndarray, nucleus_mask: np.ndarray, low: float, high: float) -> np.ndarray:
    low = float(np.clip(low, 0.0, 1.0))
    high = float(np.clip(max(high, low), 0.0, 1.0))
    candidate = (prob >= low) & nucleus_mask
    seeds = (prob >= high) & nucleus_mask
    if not np.any(seeds):
        return np.zeros_like(nucleus_mask, dtype=bool)
    lbl, n = ndi.label(candidate)
    if n == 0:
        return np.zeros_like(nucleus_mask, dtype=bool)
    seed_labels = np.unique(lbl[seeds])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return np.zeros_like(nucleus_mask, dtype=bool)
    return np.isin(lbl, seed_labels)


def _mrf_refine(prob: np.ndarray, norm_img_sub: np.ndarray, nucleus_mask: np.ndarray, initial_mask: np.ndarray, params: AnalysisParameters) -> np.ndarray:
    if int(params.mrf_iterations) <= 0:
        return initial_mask.astype(bool)
    p = np.clip(prob.astype(np.float32), 1.0e-4, 1.0 - 1.0e-4)
    unary = _logit(p)
    grad = ndi.gaussian_gradient_magnitude(norm_img_sub.astype(np.float32), sigma=1.0)
    g_scale = float(np.percentile(grad[nucleus_mask], 95.0)) if np.any(nucleus_mask) else 1.0
    edge_conf = np.exp(-grad / max(float(params.mrf_edge_sigma) * g_scale, EPS)).astype(np.float32)
    kernel = np.zeros((3, 3, 3), dtype=np.float32)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0

    mask = initial_mask.astype(bool) & nucleus_mask
    seed = (p >= float(params.probability_high_threshold)) & nucleus_mask
    candidate = (p >= max(0.05, float(params.probability_low_threshold) * 0.5)) & nucleus_mask
    lam = float(params.mrf_lambda)
    for _ in range(int(params.mrf_iterations)):
        weighted_mask = mask.astype(np.float32) * edge_conf
        num = ndi.convolve(weighted_mask, kernel, mode="constant", cval=0.0)
        den = ndi.convolve(edge_conf, kernel, mode="constant", cval=0.0) + EPS
        neighbor_fraction = num / den
        smooth_term = lam * (2.0 * neighbor_fraction - 1.0) * edge_conf
        score = unary + smooth_term
        mask = (score > 0.0) & candidate
        mask[seed] = True
        mask &= nucleus_mask
    return mask.astype(bool)


def _apply_optional_morphology(mask: np.ndarray, params: AnalysisParameters) -> np.ndarray:
    out = mask.astype(bool)
    if not bool(params.apply_binary_morphology):
        return out
    if int(params.arm_opening_radius) > 0:
        out = binary_opening_compat(out, ball(int(params.arm_opening_radius)))
    if int(params.arm_closing_radius) > 0:
        out = binary_closing_compat(out, ball(int(params.arm_closing_radius)))
    return out.astype(bool)


def _component_scores(mask: np.ndarray, prob: np.ndarray, norm_img_sub: np.ndarray, nucleus_mask: np.ndarray, params: AnalysisParameters, spacing_zyx: SpacingZYX) -> list[dict]:
    lbl, n = ndi.label(mask.astype(bool))
    if n == 0:
        return []
    voxel_um3 = float(np.prod(spacing_zyx))
    min_vol = max(float(params.min_arm_volume_um3), voxel_um3)
    scores = []
    for lab in range(1, n + 1):
        comp = lbl == lab
        n_vox = int(np.sum(comp))
        volume = n_vox * voxel_um3
        if volume + EPS < float(params.min_arm_volume_um3):
            continue
        mean_prob = float(np.mean(prob[comp])) if n_vox else 0.0
        comp_mean = float(np.mean(norm_img_sub[comp])) if n_vox else 0.0
        ring = ndi.binary_dilation(comp, structure=ball(2)) & nucleus_mask & (~comp)
        if np.any(ring):
            local_bg = float(np.median(norm_img_sub[ring]))
        else:
            rest = nucleus_mask & (~comp)
            local_bg = float(np.median(norm_img_sub[rest])) if np.any(rest) else 0.0
        contrast = comp_mean - local_bg
        boundary = find_boundaries(comp, mode="outer") & nucleus_mask
        boundary_irregularity = float(np.sum(boundary)) / max(float(n_vox) ** (2.0 / 3.0), 1.0)
        score = (
            float(params.component_weight_probability) * mean_prob
            + float(params.component_weight_contrast) * math.tanh(contrast / 2.0)
            + float(params.component_weight_volume) * math.tanh(math.log1p(volume / min_vol) / 2.0)
            - float(params.component_weight_boundary_penalty) * min(boundary_irregularity / 6.0, 1.0)
        )
        scores.append(
            {
                "label": int(lab),
                "n_voxels": n_vox,
                "volume_um3": float(volume),
                "mean_prob": mean_prob,
                "mean_norm_intensity": comp_mean,
                "local_contrast": float(contrast),
                "boundary_irregularity": float(boundary_irregularity),
                "score": float(score),
            }
        )
    return scores


def _select_components(mask: np.ndarray, prob: np.ndarray, norm_img_sub: np.ndarray, nucleus_mask: np.ndarray, params: AnalysisParameters, spacing_zyx: SpacingZYX) -> tuple[np.ndarray, dict]:
    lbl, n = ndi.label(mask.astype(bool))
    metrics = {
        "n_candidate_components": int(n),
        "n_kept_components": 0,
        "best_component_score": float("nan"),
        "best_component_volume_um3": 0.0,
        "best_component_mean_prob": float("nan"),
    }
    if n == 0:
        return np.zeros_like(mask, dtype=bool), metrics

    mode = str(params.component_selection)
    if bool(params.keep_largest_arm_component) and mode == "none":
        mode = "largest"
    voxel_um3 = float(np.prod(spacing_zyx))
    min_vox = max(1, int(math.ceil(float(params.min_arm_volume_um3) / voxel_um3)))

    if mode == "none":
        out = mask.astype(bool)
        metrics["n_kept_components"] = int(ndi.label(out)[1])
        return out, metrics

    counts = np.bincount(lbl.ravel())
    valid = np.where(counts >= min_vox)[0]
    valid = valid[valid > 0]
    if valid.size == 0:
        return np.zeros_like(mask, dtype=bool), metrics

    if mode == "largest":
        keep_lab = int(valid[np.argmax(counts[valid])])
        out = lbl == keep_lab
        metrics["n_kept_components"] = 1
        metrics["best_component_volume_um3"] = float(counts[keep_lab] * voxel_um3)
        metrics["best_component_mean_prob"] = float(np.mean(prob[out])) if np.any(out) else float("nan")
        metrics["best_component_score"] = metrics["best_component_mean_prob"]
        return out, metrics

    if mode == "all_after_size_filter":
        out = np.isin(lbl, valid)
        metrics["n_kept_components"] = int(valid.size)
        metrics["best_component_volume_um3"] = float(np.max(counts[valid]) * voxel_um3)
        metrics["best_component_mean_prob"] = float(np.mean(prob[out])) if np.any(out) else float("nan")
        metrics["best_component_score"] = metrics["best_component_mean_prob"]
        return out, metrics

    scores = _component_scores(mask, prob, norm_img_sub, nucleus_mask, params, spacing_zyx)
    if not scores:
        return np.zeros_like(mask, dtype=bool), metrics
    best = max(scores, key=lambda d: d["score"])
    metrics["best_component_score"] = float(best["score"])
    metrics["best_component_volume_um3"] = float(best["volume_um3"])
    metrics["best_component_mean_prob"] = float(best["mean_prob"])

    threshold = float(params.component_score_threshold)
    if mode == "all_passing_score":
        keep_labels = [d["label"] for d in scores if d["score"] >= threshold]
    else:  # best_score and any unknown mode
        keep_labels = [best["label"]] if best["score"] >= threshold else []
    if not keep_labels:
        return np.zeros_like(mask, dtype=bool), metrics
    out = np.isin(lbl, keep_labels)
    metrics["n_kept_components"] = int(len(keep_labels))
    return out.astype(bool), metrics


def _presence_metrics(prob: np.ndarray, norm_img_sub: np.ndarray, nucleus_mask: np.ndarray, delta_bic: float, params: AnalysisParameters, spacing_zyx: SpacingZYX) -> dict:
    vals = np.asarray(norm_img_sub[nucleus_mask], dtype=np.float32)
    pvals = np.asarray(prob[nucleus_mask], dtype=np.float32)
    metrics = {
        "delta_bic_or_llr": float(delta_bic),
        "expected_signal_volume_um3": 0.0,
        "expected_signal_fraction": 0.0,
        "signal_snr": 0.0,
        "top1pct_mean_norm_intensity": float("nan"),
        "mean_posterior_over_expected_signal": float("nan"),
        "presence_probability": float("nan"),
        "presence_accepted": 0,
        "rejection_reason": "",
    }
    if vals.size == 0 or pvals.size == 0:
        metrics["rejection_reason"] = "empty_nucleus"
        return metrics
    voxel_um3 = float(np.prod(spacing_zyx))
    expected_vox = float(np.sum(np.clip(pvals, 0.0, 1.0)))
    metrics["expected_signal_volume_um3"] = expected_vox * voxel_um3
    metrics["expected_signal_fraction"] = expected_vox / max(float(np.sum(nucleus_mask)), 1.0)
    finite = vals[np.isfinite(vals)]
    if finite.size:
        p99 = float(np.percentile(finite, 99.0))
        p95 = float(np.percentile(finite, 95.0))
        med = float(np.median(finite))
        low = finite[finite <= np.percentile(finite, 50.0)]
        _m, low_sigma, _n = _median_mad(low)
        metrics["signal_snr"] = float((p99 - med) / max(low_sigma, EPS))
        top_n = max(1, int(math.ceil(0.01 * finite.size)))
        metrics["top1pct_mean_norm_intensity"] = float(np.mean(np.sort(finite)[-top_n:]))
        high_p = pvals[pvals >= float(params.hard_probability_threshold)]
        metrics["mean_posterior_over_expected_signal"] = float(np.mean(high_p)) if high_p.size else float(np.mean(pvals))
        # Use p95 as a backup for very small nuclei where p99 is unstable.
        metrics["signal_snr_p95"] = float((p95 - med) / max(low_sigma, EPS))
    return metrics


def _classifier_presence_probability(metrics: dict, params: AnalysisParameters) -> float:
    min_vol = max(float(params.min_arm_volume_um3), EPS)
    vol_ratio = float(metrics.get("expected_signal_volume_um3", 0.0)) / min_vol
    delta = float(metrics.get("delta_bic_or_llr", 0.0))
    snr = float(metrics.get("signal_snr", 0.0))
    mean_post = float(metrics.get("mean_posterior_over_expected_signal", 0.0))
    if not np.isfinite(mean_post):
        mean_post = 0.0
    linear = (
        float(params.classifier_bias)
        + 1.20 * (snr - float(params.classifier_peak_z_threshold))
        + 0.012 * (delta - float(params.min_delta_bic))
        + 2.00 * (mean_post - 0.50)
        + 0.75 * math.log1p(max(vol_ratio, 0.0))
    )
    return float(_sigmoid(linear))


def _apply_presence_gate(metrics: dict, params: AnalysisParameters, use_classifier: bool) -> tuple[bool, str, float]:
    reasons = []
    if float(metrics.get("delta_bic_or_llr", 0.0)) < float(params.min_delta_bic):
        reasons.append("delta_bic_below_threshold")
    if float(metrics.get("signal_snr", 0.0)) < float(params.min_signal_snr):
        reasons.append("snr_below_threshold")
    if float(metrics.get("expected_signal_volume_um3", 0.0)) < float(params.min_arm_volume_um3):
        reasons.append("expected_volume_below_min")
    frac = float(metrics.get("expected_signal_fraction", 0.0))
    if frac < float(params.min_signal_fraction):
        reasons.append("expected_fraction_below_min")
    if frac > float(params.max_signal_fraction):
        reasons.append("expected_fraction_above_max")
    mean_post = float(metrics.get("mean_posterior_over_expected_signal", 0.0))
    if np.isfinite(mean_post) and mean_post < float(params.min_mean_posterior):
        reasons.append("mean_posterior_below_min")

    clf_prob = _classifier_presence_probability(metrics, params)
    if use_classifier and clf_prob < float(params.classifier_threshold):
        reasons.append("classifier_probability_below_threshold")
    return len(reasons) == 0, ";".join(reasons), clf_prob


def _legacy_segment_arm(
    raw_or_norm_img_sub: np.ndarray,
    nucleus_mask: np.ndarray,
    min_sorted_class: int,
    params: AnalysisParameters,
    spacing_zyx: SpacingZYX,
) -> ArmSegmentationResult:
    """Original one-dimensional GMM segmentation, retained as a selectable method."""
    prob, gmm_metrics = _gmm_probability_1d(raw_or_norm_img_sub, nucleus_mask, min_sorted_class, params)
    if int(gmm_metrics.get("gmm_n_components_used", 0)) <= 0:
        return ArmSegmentationResult(np.zeros_like(nucleus_mask, dtype=bool), prob, {**gmm_metrics, "presence_accepted": 0, "rejection_reason": "gmm_failed"})
    # Legacy used hard class assignment. A 0.5 posterior cutoff closely matches that behavior.
    seg = (prob >= 0.5) & nucleus_mask
    old_apply = params.apply_binary_morphology
    # Preserve legacy behavior with opening/closing radii if the user set them.
    if int(params.arm_opening_radius) > 0:
        seg = binary_opening_compat(seg, ball(int(params.arm_opening_radius)))
    if int(params.arm_closing_radius) > 0:
        seg = binary_closing_compat(seg, ball(int(params.arm_closing_radius)))
    seg, component_metrics = _select_components(seg, prob, raw_or_norm_img_sub, nucleus_mask, params, spacing_zyx)
    met = _presence_metrics(prob, raw_or_norm_img_sub, nucleus_mask, float(gmm_metrics.get("gmm_delta_bic", 0.0)), params, spacing_zyx)
    met.update(gmm_metrics)
    met.update(component_metrics)
    met["presence_accepted"] = int(np.any(seg))
    met["rejection_reason"] = "" if np.any(seg) else "empty_after_legacy_postprocess"
    met["mean_posterior_in_final_mask"] = float(np.mean(prob[seg])) if np.any(seg) else float("nan")
    return ArmSegmentationResult(seg.astype(bool), prob, met)


def segment_arm_within_nucleus(
    raw_img_sub: np.ndarray,
    norm_img_sub: np.ndarray,
    nucleus_mask: np.ndarray,
    min_sorted_class: int,
    params: AnalysisParameters,
    spacing_zyx: SpacingZYX,
    stats: ArmFieldStats,
) -> ArmSegmentationResult:
    """Segment one arm signal inside one nucleus using the selected method."""
    if int(np.sum(nucleus_mask)) == 0:
        zero = np.zeros_like(nucleus_mask, dtype=np.float32)
        return ArmSegmentationResult(np.zeros_like(nucleus_mask, dtype=bool), zero, {"presence_accepted": 0, "rejection_reason": "empty_nucleus"})

    flags = _method_flags(params)
    if flags["legacy"]:
        # Deliberately use raw/smoothed intensities, not field-normalized context, to reproduce the old method.
        return _legacy_segment_arm(raw_img_sub, nucleus_mask, min_sorted_class, params, spacing_zyx)

    prob_gmm, gmm_metrics = _gmm_probability_1d(norm_img_sub, nucleus_mask, min_sorted_class, params)
    probabilities = [prob_gmm]
    metrics: dict = dict(gmm_metrics)

    if flags["hierarchical"]:
        prob_h = _hierarchical_prior_probability(norm_img_sub, nucleus_mask, stats)
        probabilities.append(prob_h)
        metrics["hierarchical_prior_mu_bg"] = float(stats.prior_mu_bg)
        metrics["hierarchical_prior_sigma_bg"] = float(stats.prior_sigma_bg)
        metrics["hierarchical_prior_mu_sig"] = float(stats.prior_mu_sig)
        metrics["hierarchical_prior_sigma_sig"] = float(stats.prior_sigma_sig)
        metrics["hierarchical_prior_pi_sig"] = float(stats.prior_pi_sig)
        metrics["hierarchical_prior_delta_bic_field"] = float(stats.prior_delta_bic)
        metrics["hierarchical_prior_source"] = str(stats.prior_source)

    if flags["spatial"]:
        prob_s, spatial_metrics = _gmm_probability_spatial(norm_img_sub, nucleus_mask, min_sorted_class, params)
        probabilities.append(prob_s)
        metrics.update(spatial_metrics)

    prob = _combine_probabilities(probabilities)
    if prob.shape != norm_img_sub.shape:
        prob = np.zeros_like(norm_img_sub, dtype=np.float32)

    # Prefer the per-nucleus 1D delta BIC for presence. If unavailable, use spatial or field prior diagnostics.
    delta = float(metrics.get("gmm_delta_bic", 0.0))
    if delta <= 0 and flags["spatial"]:
        delta = float(metrics.get("spatial_gmm_delta_bic", 0.0))
    if delta <= 0 and flags["hierarchical"]:
        delta = float(stats.prior_delta_bic)
    presence = _presence_metrics(prob, norm_img_sub, nucleus_mask, delta, params, spacing_zyx)
    presence.update(metrics)

    if flags["presence_gate"]:
        accepted, reason, clf_prob = _apply_presence_gate(presence, params, use_classifier=flags["classifier"])
        presence["presence_probability"] = float(clf_prob)
        presence["presence_accepted"] = int(accepted)
        presence["rejection_reason"] = reason
        if not accepted:
            return ArmSegmentationResult(np.zeros_like(nucleus_mask, dtype=bool), prob, presence)
    elif flags["classifier"]:
        clf_prob = _classifier_presence_probability(presence, params)
        presence["presence_probability"] = float(clf_prob)
        accepted = clf_prob >= float(params.classifier_threshold)
        presence["presence_accepted"] = int(accepted)
        presence["rejection_reason"] = "" if accepted else "classifier_probability_below_threshold"
        if not accepted:
            return ArmSegmentationResult(np.zeros_like(nucleus_mask, dtype=bool), prob, presence)
    else:
        presence["presence_probability"] = float(_classifier_presence_probability(presence, params))
        presence["presence_accepted"] = 1
        presence["rejection_reason"] = ""

    if flags["hysteresis"]:
        seg = _hysteresis_mask(prob, nucleus_mask, params.probability_low_threshold, params.probability_high_threshold)
    else:
        seg = (prob >= float(params.hard_probability_threshold)) & nucleus_mask

    if flags["mrf"]:
        seg = _mrf_refine(prob, norm_img_sub, nucleus_mask, seg, params)

    seg = _apply_optional_morphology(seg, params)
    seg, component_metrics = _select_components(seg, prob, norm_img_sub, nucleus_mask, params, spacing_zyx)
    presence.update(component_metrics)
    if not np.any(seg):
        presence["presence_accepted"] = 0
        if not presence.get("rejection_reason"):
            presence["rejection_reason"] = "empty_after_component_filter"
    else:
        presence["presence_accepted"] = 1
        presence["rejection_reason"] = ""
    presence["mean_posterior_in_final_mask"] = float(np.mean(prob[seg])) if np.any(seg) else float("nan")
    return ArmSegmentationResult(seg.astype(bool), prob, presence)


# -----------------------------------------------------------------------------
# Measurement helpers
# -----------------------------------------------------------------------------


def shell_index_map_for_nucleus(mask: np.ndarray, n_shells: int = 5) -> np.ndarray:
    n_shells = max(1, int(n_shells))
    dist = ndi.distance_transform_edt(mask)
    inside = dist[mask > 0]
    if inside.size == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    qs = np.quantile(inside, np.linspace(0, 1, n_shells + 1))
    shell_map = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(n_shells):
        lo, hi = qs[i], qs[i + 1]
        if i == n_shells - 1:
            shell_map[(mask > 0) & (dist >= lo) & (dist <= hi)] = i + 1
        else:
            shell_map[(mask > 0) & (dist >= lo) & (dist < hi)] = i + 1
    return shell_map


def min_edge_distance(mask_a: np.ndarray, mask_b: np.ndarray, spacing_zyx: SpacingZYX) -> float:
    """Minimum edge-to-mask distance in physical units; returns 0 for overlap/contact."""
    if not np.any(mask_a) or not np.any(mask_b):
        return float("nan")
    if np.any(mask_a & mask_b):
        return 0.0
    boundary_a = find_boundaries(mask_a, mode="inner")
    if not np.any(boundary_a):
        boundary_a = mask_a
    dist_to_b = ndi.distance_transform_edt(~mask_b.astype(bool), sampling=spacing_zyx)
    return float(np.nanmin(dist_to_b[boundary_a]))


def centroid_um(mask: np.ndarray, spacing_zyx: SpacingZYX, offset_zyx: Sequence[int] = (0, 0, 0)) -> tuple[float, float, float]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    c = (pts.mean(axis=0) + np.asarray(offset_zyx, dtype=np.float64)) * np.asarray(spacing_zyx, dtype=np.float64)
    return tuple(map(float, c))


def overlap_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    ov = float(np.logical_and(mask_a, mask_b).sum())
    va = float(mask_a.sum())
    vb = float(mask_b.sum())
    fa = ov / va if va > 0 else float("nan")
    fb = ov / vb if vb > 0 else float("nan")
    return fa, fb




def _mask_surface_area_um2(mask: np.ndarray, spacing_zyx: SpacingZYX) -> float:
    """Estimate the physical 3D surface area of a binary object using marching cubes."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or int(mask.sum()) == 0:
        return float("nan")
    # Padding creates a closed surface even if the object touches the local subvolume edge.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    try:
        verts, faces, _normals, _values = marching_cubes(
            padded.astype(np.float32),
            level=0.5,
            spacing=tuple(float(v) for v in spacing_zyx),
        )
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return float("nan")


def _shape_metrics(mask: np.ndarray, spacing_zyx: SpacingZYX, prefix: str) -> dict:
    """Return common 3D object shape measurements for a binary volume.

    Columns are prefixed, for example ``p_shape_sphericity`` or
    ``nucleus_shape_surface_area_um2``. Volumes use the physical voxel size.
    Principal-axis lengths are PCA-based size descriptors in microns.
    """
    mask = np.asarray(mask, dtype=bool)
    voxel_um3 = float(np.prod(spacing_zyx))
    n_vox = int(mask.sum())
    volume = float(n_vox) * voxel_um3
    out = {
        f"{prefix}_voxel_count": n_vox,
        f"{prefix}_volume_um3": volume,
        f"{prefix}_surface_area_um2": float("nan"),
        f"{prefix}_sphericity": float("nan"),
        f"{prefix}_compactness": float("nan"),
        f"{prefix}_equivalent_sphere_diameter_um": float("nan"),
        f"{prefix}_bbox_z_um": float("nan"),
        f"{prefix}_bbox_y_um": float("nan"),
        f"{prefix}_bbox_x_um": float("nan"),
        f"{prefix}_bbox_volume_um3": float("nan"),
        f"{prefix}_extent": float("nan"),
        f"{prefix}_pca_major_axis_um": float("nan"),
        f"{prefix}_pca_intermediate_axis_um": float("nan"),
        f"{prefix}_pca_minor_axis_um": float("nan"),
        f"{prefix}_elongation_major_to_minor": float("nan"),
        f"{prefix}_elongation_major_to_intermediate": float("nan"),
        f"{prefix}_flatness_intermediate_to_minor": float("nan"),
    }
    if n_vox <= 0:
        return out

    area = _mask_surface_area_um2(mask, spacing_zyx)
    out[f"{prefix}_surface_area_um2"] = area
    if np.isfinite(area) and area > EPS and volume > EPS:
        sphericity = (math.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / area
        # Discrete marching-cubes estimates for tiny objects can be slightly above 1.
        # Clamp the physical interpretive range while preserving NaN for invalid cases.
        out[f"{prefix}_sphericity"] = float(np.clip(sphericity, 0.0, 1.0))
        out[f"{prefix}_compactness"] = float(np.clip((36.0 * math.pi * (volume ** 2.0)) / (area ** 3.0), 0.0, 1.0))
    if volume > EPS:
        out[f"{prefix}_equivalent_sphere_diameter_um"] = float((6.0 * volume / math.pi) ** (1.0 / 3.0))

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims_vox = (maxs - mins + 1).astype(np.float64)
    dims_um = dims_vox * np.asarray(spacing_zyx, dtype=np.float64)
    out[f"{prefix}_bbox_z_um"] = float(dims_um[0])
    out[f"{prefix}_bbox_y_um"] = float(dims_um[1])
    out[f"{prefix}_bbox_x_um"] = float(dims_um[2])
    bbox_volume = float(np.prod(dims_um))
    out[f"{prefix}_bbox_volume_um3"] = bbox_volume
    out[f"{prefix}_extent"] = float(volume / bbox_volume) if bbox_volume > EPS else float("nan")

    if coords.shape[0] >= 3:
        pts_um = coords.astype(np.float64) * np.asarray(spacing_zyx, dtype=np.float64)[None, :]
        pts_um -= pts_um.mean(axis=0, keepdims=True)
        try:
            cov = np.cov(pts_um, rowvar=False, bias=True)
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eig = np.maximum(eig, 0.0)
            axes = 4.0 * np.sqrt(eig)
            out[f"{prefix}_pca_major_axis_um"] = float(axes[0])
            out[f"{prefix}_pca_intermediate_axis_um"] = float(axes[1])
            out[f"{prefix}_pca_minor_axis_um"] = float(axes[2])
            out[f"{prefix}_elongation_major_to_minor"] = float(axes[0] / max(axes[2], EPS))
            out[f"{prefix}_elongation_major_to_intermediate"] = float(axes[0] / max(axes[1], EPS))
            out[f"{prefix}_flatness_intermediate_to_minor"] = float(axes[1] / max(axes[2], EPS))
        except Exception:
            pass
    return out


def _nucleus_radial_map_3d(nucleus_mask: np.ndarray, spacing_zyx: SpacingZYX) -> tuple[np.ndarray, float, float]:
    """3D extension of the HiTIPS distance-transform radial coordinate.

    The returned radial coordinate is 0 near the nuclear center and 1 near the
    nuclear surface. It is computed from the 3D Euclidean distance transform
    inside the nucleus, using physical voxel spacing.
    """
    nucleus_mask = np.asarray(nucleus_mask, dtype=bool)
    radial = np.full(nucleus_mask.shape, np.nan, dtype=np.float32)
    if not np.any(nucleus_mask):
        return radial, float("nan"), float("nan")
    dist = ndi.distance_transform_edt(nucleus_mask, sampling=tuple(float(v) for v in spacing_zyx)).astype(np.float32)
    inside = dist[nucleus_mask]
    if inside.size == 0:
        return radial, float("nan"), float("nan")
    d_max = float(np.nanmax(inside))
    d_min = float(np.nanmin(inside))
    denom = max(d_max - d_min, EPS)
    radial[nucleus_mask] = np.clip((d_max - dist[nucleus_mask]) / denom, 0.0, 1.0)
    return radial, d_max, d_min


def _radial_position_metrics(territory_mask: np.ndarray, nucleus_mask: np.ndarray, spacing_zyx: SpacingZYX, prefix: str) -> dict:
    territory_mask = np.asarray(territory_mask, dtype=bool)
    nucleus_mask = np.asarray(nucleus_mask, dtype=bool)
    radial, d_max, d_min = _nucleus_radial_map_3d(nucleus_mask, spacing_zyx)
    out = {
        f"{prefix}_radial_centroid": float("nan"),
        f"{prefix}_radial_min": float("nan"),
        f"{prefix}_radial_max": float("nan"),
        f"{prefix}_radial_mean": float("nan"),
        f"{prefix}_radial_median": float("nan"),
        f"{prefix}_nucleus_distance_transform_max_um": d_max,
        f"{prefix}_nucleus_distance_transform_boundary_um": d_min,
    }
    valid_mask = territory_mask & nucleus_mask
    if not np.any(valid_mask):
        return out
    vals = radial[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size:
        out[f"{prefix}_radial_min"] = float(np.nanmin(vals))
        out[f"{prefix}_radial_max"] = float(np.nanmax(vals))
        out[f"{prefix}_radial_mean"] = float(np.nanmean(vals))
        out[f"{prefix}_radial_median"] = float(np.nanmedian(vals))
    coords = np.argwhere(valid_mask).astype(np.float64)
    if coords.size:
        centroid = coords.mean(axis=0)
        try:
            cval = ndi.map_coordinates(radial, centroid.reshape(3, 1), order=1, mode="nearest")[0]
            out[f"{prefix}_radial_centroid"] = float(cval) if np.isfinite(cval) else float("nan")
        except Exception:
            # Fallback to nearest voxel if interpolation fails.
            c_round = np.clip(np.rint(centroid).astype(int), 0, np.asarray(radial.shape) - 1)
            cval = radial[tuple(c_round)]
            out[f"{prefix}_radial_centroid"] = float(cval) if np.isfinite(cval) else float("nan")
    return out

def analyze_one_nucleus_from_subvolume(
    label_id: int,
    nucleus_mask: np.ndarray,
    p_result: ArmSegmentationResult,
    q_result: ArmSegmentationResult,
    spacing_zyx: SpacingZYX,
    params: AnalysisParameters,
    p_stats: ArmFieldStats,
    q_stats: ArmFieldStats,
    offset_zyx: Sequence[int] = (0, 0, 0),
) -> dict:
    p_mask = p_result.mask
    q_mask = q_result.mask
    voxel_um3 = float(np.prod(spacing_zyx))
    nuc_vol = float(nucleus_mask.sum()) * voxel_um3
    p_vol = float(p_mask.sum()) * voxel_um3
    q_vol = float(q_mask.sum()) * voxel_um3
    ov_mask = p_mask & q_mask
    ov_vol = float(ov_mask.sum()) * voxel_um3
    p_frac = p_vol / nuc_vol if nuc_vol > 0 else float("nan")
    q_frac = q_vol / nuc_vol if nuc_vol > 0 else float("nan")
    ov_p_frac, ov_q_frac = overlap_fraction(p_mask, q_mask)

    shell_map = shell_index_map_for_nucleus(nucleus_mask, n_shells=params.n_shells)
    shell_fracs: dict[str, float] = {}
    for arm_name, arm_mask in [("p", p_mask), ("q", q_mask)]:
        total = float(arm_mask.sum())
        for s in range(1, params.n_shells + 1):
            v = float(np.logical_and(arm_mask, shell_map == s).sum())
            shell_fracs[f"{arm_name}_shell_{s}_frac"] = v / total if total > 0 else float("nan")

    p_cent = centroid_um(p_mask, spacing_zyx, offset_zyx)
    q_cent = centroid_um(q_mask, spacing_zyx, offset_zyx)

    radius = max(0, int(params.contact_dilation_radius))
    if radius > 0:
        contact = bool(np.any(ndi.binary_dilation(p_mask, structure=ball(radius)) & q_mask))
    else:
        contact = bool(np.any(p_mask & q_mask))

    edge_dist = min_edge_distance(p_mask, q_mask, spacing_zyx)

    shape_metrics = {}
    shape_metrics.update(_shape_metrics(nucleus_mask, spacing_zyx, "nucleus_shape"))
    shape_metrics.update(_shape_metrics(p_mask, spacing_zyx, "p_shape"))
    shape_metrics.update(_shape_metrics(q_mask, spacing_zyx, "q_shape"))
    shape_metrics.update(_shape_metrics(ov_mask, spacing_zyx, "pq_overlap_shape"))

    radial_metrics = {}
    radial_metrics.update(_radial_position_metrics(p_mask, nucleus_mask, spacing_zyx, "p"))
    radial_metrics.update(_radial_position_metrics(q_mask, nucleus_mask, spacing_zyx, "q"))
    radial_metrics.update(_radial_position_metrics(ov_mask, nucleus_mask, spacing_zyx, "pq_overlap"))

    out = {
        "nucleus_id": int(label_id),
        "arm_detection_method": str(params.arm_detection_method),
        "nucleus_volume_um3": nuc_vol,
        "p_volume_um3": p_vol,
        "q_volume_um3": q_vol,
        "p_fraction_of_nucleus": p_frac,
        "q_fraction_of_nucleus": q_frac,
        "pq_overlap_volume_um3": ov_vol,
        "pq_overlap_fraction_of_p": ov_p_frac,
        "pq_overlap_fraction_of_q": ov_q_frac,
        "pq_contact": int(contact),
        "pq_min_edge_distance_um": edge_dist,
        "p_centroid_z_um": p_cent[0],
        "p_centroid_y_um": p_cent[1],
        "p_centroid_x_um": p_cent[2],
        "q_centroid_z_um": q_cent[0],
        "q_centroid_y_um": q_cent[1],
        "q_centroid_x_um": q_cent[2],
        "p_field_background_median_raw": float(p_stats.background_median_raw),
        "p_field_background_mad_raw": float(p_stats.background_mad_raw),
        "q_field_background_median_raw": float(q_stats.background_median_raw),
        "q_field_background_mad_raw": float(q_stats.background_mad_raw),
    }
    out.update(shell_fracs)
    out.update(shape_metrics)
    out.update(radial_metrics)
    for prefix, result in [("p", p_result), ("q", q_result)]:
        for key, val in result.metrics.items():
            col = f"{prefix}_{key}"
            if isinstance(val, (np.floating, float)):
                out[col] = float(val)
            elif isinstance(val, (np.integer, int)):
                out[col] = int(val)
            else:
                out[col] = val
    return out


def _label_ids(nuclei_labels: np.ndarray, limit_nuclei: int = 0) -> list[int]:
    ids = [int(x) for x in np.unique(nuclei_labels) if int(x) > 0]
    if limit_nuclei and limit_nuclei > 0:
        ids = ids[: int(limit_nuclei)]
    return ids


def segment_arms_for_nuclei(
    nuclei_labels: np.ndarray,
    p_img: np.ndarray,
    q_img: np.ndarray,
    spacing_zyx: SpacingZYX,
    params: AnalysisParameters,
    limit_nuclei: int = 0,
    return_rows: bool = False,
    progress: ProgressCallback = None,
):
    """Segment P/Q masks for each nucleus and optionally compute measurements.

    The expensive per-nucleus work is embarrassingly parallel. This function first
    computes global field context once, then dispatches one independent job per
    nucleus using the selected joblib backend. Result arrays are merged in the main
    thread to avoid write conflicts.
    """
    if nuclei_labels.shape != p_img.shape or nuclei_labels.shape != q_img.shape:
        raise ValueError(
            f"Shape mismatch: nuclei={nuclei_labels.shape}, P={p_img.shape}, Q={q_img.shape}. "
            "All selected channels must have the same ZYX shape."
        )

    contexts = build_arm_contexts(p_img, q_img, nuclei_labels, params, progress=progress)
    p_ctx = contexts["p"]
    q_ctx = contexts["q"]

    p_union = np.zeros_like(nuclei_labels, dtype=bool)
    q_union = np.zeros_like(nuclei_labels, dtype=bool)
    p_prob_union = np.zeros_like(nuclei_labels, dtype=np.float32)
    q_prob_union = np.zeros_like(nuclei_labels, dtype=np.float32)
    p_labels = np.zeros_like(nuclei_labels, dtype=np.int32)
    q_labels = np.zeros_like(nuclei_labels, dtype=np.int32)
    rows: list[dict] = []

    ids = _label_ids(nuclei_labels, limit_nuclei=limit_nuclei)
    if not ids:
        if return_rows:
            return rows, p_union, q_union, p_labels, q_labels, p_prob_union, q_prob_union, contexts
        return p_union, q_union, p_labels, q_labels, p_prob_union, q_prob_union

    # find_objects is much faster than regionprops for bounding boxes. It assumes
    # positive integer labels; missing/non-dense labels are handled by None checks.
    objects = ndi.find_objects(np.asarray(nuclei_labels, dtype=np.int32))
    tasks = []
    for label_id in ids:
        idx = int(label_id) - 1
        if idx < 0 or idx >= len(objects):
            continue
        slc = objects[idx]
        if slc is None:
            continue
        tasks.append((int(label_id), slc))

    total = len(tasks)
    if progress is not None:
        log(f"Segmenting arms in {total} nuclei", progress)

    def _one(task):
        label_id, slc = task
        z0 = int(slc[0].start)
        y0 = int(slc[1].start)
        x0 = int(slc[2].start)
        nuc_sub = nuclei_labels[slc] == int(label_id)
        p_result = segment_arm_within_nucleus(
            p_ctx.raw_img[slc],
            p_ctx.normalized_img[slc],
            nuc_sub,
            int(params.p_min_sorted_class),
            params,
            spacing_zyx,
            p_ctx.stats,
        )
        q_result = segment_arm_within_nucleus(
            q_ctx.raw_img[slc],
            q_ctx.normalized_img[slc],
            nuc_sub,
            int(params.q_min_sorted_class),
            params,
            spacing_zyx,
            q_ctx.stats,
        )
        row = None
        if return_rows:
            row = analyze_one_nucleus_from_subvolume(
                int(label_id),
                nuc_sub,
                p_result,
                q_result,
                spacing_zyx,
                params,
                p_ctx.stats,
                q_ctx.stats,
                offset_zyx=(z0, y0, x0),
            )
        return label_id, slc, p_result.mask, q_result.mask, p_result.probability, q_result.probability, row

    results = _parallel_map(_one, tasks, params, progress=progress, task_name="per-nucleus P/Q segmentation")

    for label_id, slc, p_mask, q_mask, p_prob, q_prob, row in results:
        p_mask = np.asarray(p_mask, dtype=bool)
        q_mask = np.asarray(q_mask, dtype=bool)
        p_union[slc] |= p_mask
        q_union[slc] |= q_mask
        p_prob_union[slc] = np.maximum(p_prob_union[slc], np.asarray(p_prob, dtype=np.float32))
        q_prob_union[slc] = np.maximum(q_prob_union[slc], np.asarray(q_prob, dtype=np.float32))
        p_view = p_labels[slc]
        q_view = q_labels[slc]
        p_view[p_mask] = int(label_id)
        q_view[q_mask] = int(label_id)
        if return_rows and row is not None:
            rows.append(row)

    if return_rows:
        return rows, p_union, q_union, p_labels, q_labels, p_prob_union, q_prob_union, contexts
    return p_union, q_union, p_labels, q_labels, p_prob_union, q_prob_union


def make_population_summary(df: pd.DataFrame) -> dict:
    def col_mean(name: str) -> float:
        return float(df[name].mean()) if len(df) and name in df else float("nan")

    return {
        "n_nuclei": int(len(df)),
        "arm_detection_method": str(df["arm_detection_method"].iloc[0]) if len(df) and "arm_detection_method" in df else "",
        "mean_nucleus_volume_um3": col_mean("nucleus_volume_um3"),
        "mean_p_fraction_of_nucleus": col_mean("p_fraction_of_nucleus"),
        "mean_q_fraction_of_nucleus": col_mean("q_fraction_of_nucleus"),
        "mean_pq_overlap_fraction_of_p": col_mean("pq_overlap_fraction_of_p"),
        "mean_pq_overlap_fraction_of_q": col_mean("pq_overlap_fraction_of_q"),
        "pq_contact_frequency": col_mean("pq_contact"),
        "mean_pq_min_edge_distance_um": col_mean("pq_min_edge_distance_um"),
        "mean_nucleus_shape_sphericity": col_mean("nucleus_shape_sphericity"),
        "mean_p_shape_sphericity": col_mean("p_shape_sphericity"),
        "mean_q_shape_sphericity": col_mean("q_shape_sphericity"),
        "mean_p_shape_surface_area_um2": col_mean("p_shape_surface_area_um2"),
        "mean_q_shape_surface_area_um2": col_mean("q_shape_surface_area_um2"),
        "mean_p_shape_equivalent_sphere_diameter_um": col_mean("p_shape_equivalent_sphere_diameter_um"),
        "mean_q_shape_equivalent_sphere_diameter_um": col_mean("q_shape_equivalent_sphere_diameter_um"),
        "mean_p_radial_centroid": col_mean("p_radial_centroid"),
        "mean_q_radial_centroid": col_mean("q_radial_centroid"),
        "mean_p_radial_min": col_mean("p_radial_min"),
        "mean_q_radial_min": col_mean("q_radial_min"),
        "mean_p_radial_max": col_mean("p_radial_max"),
        "mean_q_radial_max": col_mean("q_radial_max"),
    }


def _safe_filename_token(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    token = "".join(keep).strip("_")
    return token or "scene"


def _context_to_jsonable(contexts: dict[str, ArmRuntimeContext]) -> dict:
    return {name: asdict(ctx.stats) for name, ctx in contexts.items()}


def run_full_analysis(
    data_czyx: np.ndarray,
    spacing_zyx: SpacingZYX,
    output_dir: str | Path,
    params: AnalysisParameters,
    source_image_path: str | Path | None = None,
    scene_name: str | None = None,
    existing_nuclei_labels: np.ndarray | None = None,
    progress: ProgressCallback = None,
) -> AnalysisOutputs:
    """Run nuclei segmentation, arm segmentation, measurements, masks, CSVs, and plots."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if data_czyx.ndim != 4:
        raise ValueError(f"Expected CZYX data, got shape {data_czyx.shape}.")
    n_channels = data_czyx.shape[0]
    for channel_name, channel_index in {
        "nucleus": params.nuc_channel,
        "P": params.p_channel,
        "Q": params.q_channel,
    }.items():
        if channel_index < 1 or channel_index > n_channels:
            raise ValueError(f"{channel_name} channel {channel_index} is outside the available range 1..{n_channels}.")

    nuc_img = np.asarray(data_czyx[params.nuc_channel - 1])
    p_img = np.asarray(data_czyx[params.p_channel - 1])
    q_img = np.asarray(data_czyx[params.q_channel - 1])

    if existing_nuclei_labels is not None:
        nuclei_labels = np.asarray(existing_nuclei_labels, dtype=np.int32)
        log(f"Using existing nuclei labels with shape {nuclei_labels.shape}", progress)
    else:
        nuclei_labels = segment_nuclei_cellpose_3d(nuc_img, spacing_zyx, params, progress=progress)

    nuclei_labels_tif = output_dir / "nuclei_labels_3d.tif"
    write_tiff(nuclei_labels_tif, nuclei_labels)

    n_ids = len(_label_ids(nuclei_labels, limit_nuclei=params.limit_nuclei))
    log(f"Analyzing {n_ids} nuclei with arm method: {params.arm_detection_method}", progress)
    rows, p_union, q_union, p_labels, q_labels, p_prob, q_prob, contexts = segment_arms_for_nuclei(
        nuclei_labels,
        p_img,
        q_img,
        spacing_zyx,
        params,
        limit_nuclei=params.limit_nuclei,
        return_rows=True,
        progress=progress,
    )
    overlap_union = p_union & q_union

    df = filter_result_columns(pd.DataFrame(rows))
    per_nucleus_csv = output_dir / "series7_chrX_arm_measurements_per_nucleus.csv"
    df.to_csv(per_nucleus_csv, index=False)

    summary = make_population_summary(df)
    population_summary_csv = output_dir / "series7_chrX_arm_measurements_population_summary.csv"
    pd.DataFrame([summary]).to_csv(population_summary_csv, index=False)

    p_mask_tif = output_dir / "p_arm_mask_3d.tif"
    q_mask_tif = output_dir / "q_arm_mask_3d.tif"
    overlap_mask_tif = output_dir / "pq_overlap_mask_3d.tif"
    write_tiff(p_mask_tif, p_union.astype(np.uint8) * 255)
    write_tiff(q_mask_tif, q_union.astype(np.uint8) * 255)
    write_tiff(overlap_mask_tif, overlap_union.astype(np.uint8) * 255)

    if params.save_probability_maps:
        write_tiff(output_dir / "p_arm_probability_3d.tif", p_prob.astype(np.float32))
        write_tiff(output_dir / "q_arm_probability_3d.tif", q_prob.astype(np.float32))

    if params.save_label_masks:
        write_tiff(output_dir / "p_arm_labels_by_nucleus_3d.tif", p_labels)
        write_tiff(output_dir / "q_arm_labels_by_nucleus_3d.tif", q_labels)
        write_tiff(output_dir / "pq_overlap_labels_by_nucleus_3d.tif", np.where(overlap_union, nuclei_labels, 0))

    if params.save_qc:
        write_tiff(output_dir / "qc_nuc_maxproj.tif", normalize_uint8(nuc_img.max(axis=0)))
        write_tiff(output_dir / "qc_p_maxproj.tif", normalize_uint8(p_img.max(axis=0)))
        write_tiff(output_dir / "qc_q_maxproj.tif", normalize_uint8(q_img.max(axis=0)))
        write_tiff(output_dir / "qc_nuclei_labels_maxproj.tif", safe_label_dtype(nuclei_labels.max(axis=0)))

    context_json = _context_to_jsonable(contexts)
    with open(output_dir / "arm_intensity_context.json", "w", encoding="utf-8") as f:
        json.dump(context_json, f, indent=2)

    metadata = {
        "source_image_path": str(source_image_path) if source_image_path is not None else None,
        "scene_name": scene_name,
        "spacing_zyx_um": list(map(float, spacing_zyx)),
        "data_shape_czyx": list(map(int, data_czyx.shape)),
        "parameters": asdict(params),
        "arm_intensity_context": context_json,
        "elapsed_seconds": float(time.time() - start),
    }
    with open(output_dir / "analysis_parameters.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # A compact configuration copy is saved in every result folder so the exact
    # parameter set can be reloaded later from the GUI.
    run_configuration = {
        "plugin": "napari-pq-arm-analyzer",
        "plugin_version": "0.3.5",
        "source_image_path": str(source_image_path) if source_image_path is not None else None,
        "scene_name": scene_name,
        "spacing_zyx_um": list(map(float, spacing_zyx)),
        "parameters": asdict(params),
    }
    with open(output_dir / "pq_arm_analyzer_configuration.json", "w", encoding="utf-8") as f:
        json.dump(run_configuration, f, indent=2)

    from .plotting import create_plots

    create_plots(output_dir, df=df, population_summary=pd.DataFrame([summary]))
    log(f"Analysis finished in {time.time() - start:.1f} s", progress)

    return AnalysisOutputs(
        output_dir=output_dir,
        plot_dir=plot_dir,
        per_nucleus_csv=per_nucleus_csv,
        population_summary_csv=population_summary_csv,
        nuclei_labels_tif=nuclei_labels_tif,
        p_mask_tif=p_mask_tif,
        q_mask_tif=q_mask_tif,
        overlap_mask_tif=overlap_mask_tif,
        n_nuclei=int(len(df)),
        summary=summary,
    )
