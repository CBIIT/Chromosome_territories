# Author: Adib Keikhosravi, Ph.D.
# Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI
# National Institutes of Health
# Email: adib.keikhosravi@nih.gov
# License: MIT License

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analysis import (
    ARM_DETECTION_METHODS,
    COMPONENT_SELECTION_MODES,
    FIELD_NORMALIZATION_MODES,
    NUCLEI_SEGMENTATION_MODES,
    PARALLEL_BACKENDS,
    AnalysisOutputs,
    AnalysisParameters,
    get_scene_names,
    load_scene_channels,
    run_full_analysis,
    segment_arms_for_nuclei,
    segment_nuclei_cellpose_3d,
)
from .image_io import SUPPORTED_IMAGE_FILTER

PLUGIN_VERSION = "0.3.8"


def _token(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep).strip("_") or "scene"


from .help_text import PARAMETER_INFO

# Use the expanded help text module so hover/click help remains comprehensive
# and consistent with the full user manual. If the module cannot be imported,
# the fallback dictionary above is still available.
try:
    from .help_text import PARAMETER_INFO as PARAMETER_INFO  # type: ignore[no-redef]
except Exception:
    pass


class PQArmAnalyzerWidget(QWidget):
    """Dock widget for interactive P/Q arm segmentation and measurement."""

    def __init__(self, viewer=None):
        super().__init__()
        if viewer is None:
            try:
                import napari

                viewer = napari.current_viewer()
            except Exception:
                viewer = None
        if viewer is None:
            raise RuntimeError(
                "P/Q Arm Analyzer could not find an active napari viewer. "
                "Open the widget from napari or pass viewer explicitly."
            )
        self.viewer = viewer
        self._image_path: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._scene_name: Optional[str] = None
        self._scene_index: Optional[int] = None
        self._scene_names: list[str] = []
        self._data_czyx: Optional[np.ndarray] = None
        self._spacing_zyx = (1.0, 1.0, 1.0)
        self._nuclei_labels: Optional[np.ndarray] = None
        self._nuclei_dirty = True
        self._main_worker = None
        self._preview_worker = None
        self._preview_busy = False
        self._preview_pending = False
        self._loading_config = False
        self._info_dialogs: dict[str, QDialog] = {}

        self._arm_preview_timer = QTimer(self)
        self._arm_preview_timer.setSingleShot(True)
        self._arm_preview_timer.setInterval(650)
        self._arm_preview_timer.timeout.connect(self._start_arm_preview)

        self._nuclei_preview_timer = QTimer(self)
        self._nuclei_preview_timer.setSingleShot(True)
        self._nuclei_preview_timer.setInterval(1500)
        self._nuclei_preview_timer.timeout.connect(self._start_nuclei_preview)

        self._build_ui()
        self._connect_parameter_signals()
        self._sync_gmm_class_ranges()
        self._remove_unwanted_preview_layers()
        self.append_status("Load an image, select a scene for preview, then tune nuclei/P/Q arm parameters.")

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.controls_layout = QVBoxLayout(body)
        scroll.setWidget(body)
        root.addWidget(scroll)

        self._build_file_group()
        self._build_channel_group()
        self._build_cellpose_group()
        self._build_parallel_group()
        self._build_arm_method_group()
        self._build_arm_context_group()
        self._build_arm_gate_group()
        self._build_arm_boundary_group()
        self._build_measurement_output_group()
        self._build_action_group()
        self.controls_layout.addStretch(1)

        self.status_box = QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(140)
        root.addWidget(self.status_box)

    def _new_group(self, title: str) -> QFormLayout:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.controls_layout.addWidget(group)
        return form

    def _spin(self, value: int, minimum: int, maximum: int, step: int = 1) -> QSpinBox:
        w = QSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setValue(value)
        return w

    def _dspin(self, value: float, minimum: float, maximum: float, step: float = 0.1, decimals: int = 3) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(value)
        return w

    def _combo(self, items: list[str], current: str | None = None) -> QComboBox:
        w = QComboBox()
        w.addItems(items)
        if current and current in items:
            w.setCurrentText(current)
        return w

    def _info_button(self, key: str, title: str, info: str) -> QToolButton:
        btn = QToolButton()
        btn.setText("i")
        btn.setToolTip(info)
        btn.setFixedSize(18, 18)
        btn.setStyleSheet(
            "QToolButton { border: 1px solid #777; border-radius: 9px; "
            "font-weight: bold; color: #1557a6; background: #eef5ff; }"
            "QToolButton:hover { background: #dbeaff; }"
        )
        btn.clicked.connect(lambda _checked=False, k=key, t=title, txt=info: self._show_info_dialog(k, t, txt))
        return btn

    def _show_info_dialog(self, key: str, title: str, info: str) -> None:
        key = str(key)
        existing = self._info_dialogs.get(key)
        if existing is not None:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._info_dialogs.pop(key, None)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Parameter help: {title}")
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.setWindowModality(Qt.NonModal)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        safe_title = html.escape(title)
        safe_info = html.escape(info).replace("\n\n", "</p><p>").replace("\n", "<br>")
        browser.setHtml(f"<h2>{safe_title}</h2><p>{safe_info}</p>")
        layout.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        layout.addWidget(close_btn)
        dlg.resize(760, 600)
        self._info_dialogs[key] = dlg
        dlg.destroyed.connect(lambda *_args, k=key: self._forget_info_dialog(k))
        dlg.show()

    def _forget_info_dialog(self, key: str) -> None:
        self._info_dialogs.pop(key, None)

    def _add_info_row(self, form: QFormLayout, label: str | None, widget: QWidget, info_key: str, info: str | None = None) -> None:
        text = info or PARAMETER_INFO.get(info_key, "No detailed help text is available for this parameter.")
        try:
            widget.setToolTip(text)
        except Exception:
            pass
        if label is None:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(widget)
            row_layout.addWidget(self._info_button(info_key, getattr(widget, "text", lambda: info_key)() or info_key, text))
            row_layout.addStretch(1)
            form.addRow(row)
        else:
            label_widget = QWidget()
            label_layout = QHBoxLayout(label_widget)
            label_layout.setContentsMargins(0, 0, 0, 0)
            label_layout.addWidget(QLabel(label))
            label_layout.addWidget(self._info_button(info_key, label, text))
            label_layout.addStretch(1)
            form.addRow(label_widget, widget)

    def _add_plain_row(self, form: QFormLayout, widget: QWidget) -> None:
        form.addRow(widget)

    # ------------------------------------------------------------------
    # UI groups
    # ------------------------------------------------------------------
    def _build_file_group(self) -> None:
        form = self._new_group("Image, scenes, configuration, and output")
        self.image_label = QLabel("No image loaded")
        self.image_label.setWordWrap(True)
        self.scene_combo = QComboBox()
        self.scene_list = QListWidget()
        self.scene_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.scene_list.setMinimumHeight(110)
        self.output_label = QLabel("No output folder selected")
        self.output_label.setWordWrap(True)

        self.load_image_btn = QPushButton("Load image...")
        self.load_scene_btn = QPushButton("Load selected scene for preview")
        self.check_all_scenes_btn = QPushButton("Check all scenes")
        self.clear_scene_checks_btn = QPushButton("Clear scene checks")
        self.choose_output_btn = QPushButton("Choose output folder...")
        self.save_config_btn = QPushButton("Save configuration...")
        self.load_config_btn = QPushButton("Load configuration...")

        row = QHBoxLayout()
        row.addWidget(self.load_image_btn)
        row.addWidget(self.load_scene_btn)
        form.addRow(row)
        self._add_info_row(form, "Image", self.image_label, "image_path")
        self._add_info_row(form, "Preview scene", self.scene_combo, "preview_scene")
        self._add_info_row(form, "Scenes to analyze", self.scene_list, "scene_checklist")
        scene_btn_row = QHBoxLayout()
        scene_btn_row.addWidget(self.check_all_scenes_btn)
        scene_btn_row.addWidget(self.clear_scene_checks_btn)
        form.addRow(scene_btn_row)
        hint = QLabel("If no scenes are checked, Analyze uses only the currently loaded preview scene. If scenes are checked, Analyze runs all checked scenes into separate result folders.")
        hint.setWordWrap(True)
        form.addRow(hint)
        form.addRow(self.choose_output_btn)
        self._add_info_row(form, "Output", self.output_label, "output_folder")
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(self.save_config_btn)
        cfg_row.addWidget(self.load_config_btn)
        form.addRow(cfg_row)

    def _build_channel_group(self) -> None:
        form = self._new_group("Channels, 1-based")
        self.nuc_channel_spin = self._spin(1, 1, 99)
        self.p_channel_spin = self._spin(2, 1, 99)
        self.q_channel_spin = self._spin(3, 1, 99)
        self._add_info_row(form, "Nucleus channel", self.nuc_channel_spin, "nuc_channel")
        self._add_info_row(form, "P-arm channel", self.p_channel_spin, "p_channel")
        self._add_info_row(form, "Q-arm channel", self.q_channel_spin, "q_channel")

    def _build_cellpose_group(self) -> None:
        form = self._new_group("Cellpose nuclei segmentation")
        self.cellpose_model_combo = self._combo(["nuclei", "cyto3", "cyto2", "cyto"], "nuclei")
        self.cellpose_mode_combo = self._combo(NUCLEI_SEGMENTATION_MODES, "Cellpose 3D whole volume")
        self.gpu_check = QCheckBox("Use GPU")
        self.diameter_spin = self._dspin(30.0, 1.0, 500.0, step=1.0, decimals=1)
        self.xy_downsample_spin = self._dspin(2.0, 1.0, 8.0, step=0.25, decimals=2)
        self.min_nucleus_volume_spin = self._dspin(5.0, 0.0, 1_000_000.0, step=1.0, decimals=2)
        self.cellpose_batch_spin = self._spin(8, 1, 256)
        self.cellpose_overlap_spin = self._dspin(0.10, 0.0, 1.0, step=0.05, decimals=3)
        self.auto_cellpose_check = QCheckBox("Auto-run nuclei preview when Cellpose parameters change (slow)")
        self.auto_cellpose_check.setChecked(False)

        self._add_info_row(form, "Model", self.cellpose_model_combo, "cellpose_model")
        self._add_info_row(form, "Nuclei segmentation mode", self.cellpose_mode_combo, "cellpose_mode")
        self._add_info_row(form, None, self.gpu_check, "gpu")
        self._add_info_row(form, "Cell / nucleus diameter", self.diameter_spin, "diameter")
        self._add_info_row(form, "XY downsample factor", self.xy_downsample_spin, "xy_downsample")
        self._add_info_row(form, "Minimum nucleus volume (um^3)", self.min_nucleus_volume_spin, "min_nucleus_volume")
        self._add_info_row(form, "2D batch size", self.cellpose_batch_spin, "cellpose_batch")
        self._add_info_row(form, "2D z-stitch overlap fraction", self.cellpose_overlap_spin, "cellpose_overlap")
        self._add_info_row(form, None, self.auto_cellpose_check, "auto_cellpose")

    def _build_parallel_group(self) -> None:
        form = self._new_group("Parallel execution")
        self.parallel_jobs_spin = self._spin(0, 0, 256)
        self.parallel_backend_combo = self._combo(PARALLEL_BACKENDS, "threading")
        self._add_info_row(form, "Worker count", self.parallel_jobs_spin, "parallel_jobs")
        self._add_info_row(form, "Backend for per-nucleus work", self.parallel_backend_combo, "parallel_backend")

    def _build_arm_method_group(self) -> None:
        form = self._new_group("Arm detection method and GMM")
        self.method_combo = self._combo(ARM_DETECTION_METHODS, "Upgraded 1D GMM + gate + scoring")
        self.gmm_components_spin = self._spin(4, 1, 8)
        self.auto_bic_check = QCheckBox("Auto-choose 1D GMM components by BIC")
        self.auto_bic_check.setChecked(False)
        self.p_min_class_spin = self._spin(3, 0, 7)
        self.q_min_class_spin = self._spin(3, 0, 7)

        self._add_info_row(form, "Detection method", self.method_combo, "method")
        self._add_info_row(form, "Max GMM components", self.gmm_components_spin, "gmm_components")
        self._add_info_row(form, None, self.auto_bic_check, "auto_bic")
        self._add_info_row(form, "P: keep sorted class index >=", self.p_min_class_spin, "p_min_class")
        self._add_info_row(form, "Q: keep sorted class index >=", self.q_min_class_spin, "q_min_class")

    def _build_arm_context_group(self) -> None:
        form = self._new_group("Field context and intensity normalization")
        self.norm_mode_combo = self._combo(FIELD_NORMALIZATION_MODES, "mixed_outside_or_low_percentile")
        self.bg_percentile_spin = self._dspin(35.0, 0.1, 99.9, step=1.0, decimals=1)
        self.max_context_voxels_spin = self._spin(250000, 1000, 50_000_000, step=10000)
        self._add_info_row(form, "Normalization mode", self.norm_mode_combo, "norm_mode")
        self._add_info_row(form, "Background percentile", self.bg_percentile_spin, "background_percentile")
        self._add_info_row(form, "Max voxels for field context", self.max_context_voxels_spin, "max_context_voxels")

    def _build_arm_gate_group(self) -> None:
        form = self._new_group("Presence gate and probability thresholds")
        self.enable_gate_check = QCheckBox("Enable explicit presence/absence gate")
        self.enable_gate_check.setChecked(True)
        self.min_delta_bic_spin = self._dspin(10.0, -1_000_000.0, 1_000_000.0, step=5.0, decimals=2)
        self.min_snr_spin = self._dspin(2.0, -100.0, 100.0, step=0.25, decimals=2)
        self.min_signal_fraction_spin = self._dspin(0.0005, 0.0, 1.0, step=0.0005, decimals=5)
        self.max_signal_fraction_spin = self._dspin(0.65, 0.0, 1.0, step=0.05, decimals=3)
        self.min_mean_posterior_spin = self._dspin(0.35, 0.0, 1.0, step=0.05, decimals=3)
        self.hard_prob_spin = self._dspin(0.50, 0.0, 1.0, step=0.05, decimals=3)

        self._add_info_row(form, None, self.enable_gate_check, "enable_gate")
        self._add_info_row(form, "Minimum Delta BIC / LLR", self.min_delta_bic_spin, "min_delta_bic")
        self._add_info_row(form, "Minimum signal SNR", self.min_snr_spin, "min_snr")
        self._add_info_row(form, "Minimum expected signal fraction", self.min_signal_fraction_spin, "min_signal_fraction")
        self._add_info_row(form, "Maximum expected signal fraction", self.max_signal_fraction_spin, "max_signal_fraction")
        self._add_info_row(form, "Minimum mean posterior", self.min_mean_posterior_spin, "min_mean_posterior")
        self._add_info_row(form, "Probability threshold", self.hard_prob_spin, "prob_threshold")

    def _build_arm_boundary_group(self) -> None:
        form = self._new_group("Boundary cleanup, component scoring, and MRF/CRF")
        self.arm_smoothing_spin = self._dspin(0.0, 0.0, 10.0, step=0.1, decimals=2)
        self.apply_morphology_check = QCheckBox("Apply binary opening/closing after detection")
        self.apply_morphology_check.setChecked(False)
        self.opening_radius_spin = self._spin(0, 0, 20)
        self.closing_radius_spin = self._spin(0, 0, 20)
        self.min_arm_volume_spin = self._dspin(0.1, 0.0, 1_000_000.0, step=0.1, decimals=3)
        self.component_combo = self._combo(COMPONENT_SELECTION_MODES, "all_passing_score")
        self.component_score_spin = self._dspin(0.30, -100.0, 100.0, step=0.05, decimals=3)
        self.comp_prob_w_spin = self._dspin(1.0, -10.0, 10.0, step=0.1, decimals=3)
        self.comp_contrast_w_spin = self._dspin(0.6, -10.0, 10.0, step=0.1, decimals=3)
        self.comp_volume_w_spin = self._dspin(0.35, -10.0, 10.0, step=0.1, decimals=3)
        self.comp_boundary_w_spin = self._dspin(0.20, -10.0, 10.0, step=0.1, decimals=3)
        self.mrf_iterations_spin = self._spin(5, 0, 100)
        self.mrf_lambda_spin = self._dspin(1.0, 0.0, 100.0, step=0.1, decimals=3)
        self.mrf_edge_sigma_spin = self._dspin(1.0, 0.001, 100.0, step=0.1, decimals=3)

        self._add_info_row(form, "Arm Gaussian smoothing sigma", self.arm_smoothing_spin, "arm_smoothing")
        self._add_info_row(form, None, self.apply_morphology_check, "apply_morphology")
        self._add_info_row(form, "Binary opening radius", self.opening_radius_spin, "opening_radius")
        self._add_info_row(form, "Binary closing radius", self.closing_radius_spin, "closing_radius")
        self._add_info_row(form, "Minimum arm volume (um^3)", self.min_arm_volume_spin, "min_arm_volume")
        self._add_info_row(form, "Component selection", self.component_combo, "component_selection")
        self._add_info_row(form, "Component score threshold", self.component_score_spin, "component_score")
        self._add_info_row(form, "Component weight: probability", self.comp_prob_w_spin, "comp_prob_w")
        self._add_info_row(form, "Component weight: contrast", self.comp_contrast_w_spin, "comp_contrast_w")
        self._add_info_row(form, "Component weight: volume", self.comp_volume_w_spin, "comp_volume_w")
        self._add_info_row(form, "Component boundary penalty", self.comp_boundary_w_spin, "comp_boundary_w")
        self._add_info_row(form, "MRF iterations", self.mrf_iterations_spin, "mrf_iterations")
        self._add_info_row(form, "MRF lambda", self.mrf_lambda_spin, "mrf_lambda")
        self._add_info_row(form, "MRF edge sigma", self.mrf_edge_sigma_spin, "mrf_edge_sigma")

    def _build_measurement_output_group(self) -> None:
        form = self._new_group("Measurements, preview, and output")
        self.contact_radius_spin = self._spin(1, 0, 20)
        self.n_shells_spin = self._spin(5, 1, 20)
        self.preview_limit_spin = self._spin(0, 0, 100000)
        self.analysis_limit_spin = self._spin(0, 0, 100000)
        self.auto_preview_check = QCheckBox("Live-update P/Q preview when arm parameters change")
        self.auto_preview_check.setChecked(True)
        self.save_qc_check = QCheckBox("Save QC max projections")
        self.save_qc_check.setChecked(True)
        self.save_label_masks_check = QCheckBox("Save arm label masks by nucleus")
        self.save_label_masks_check.setChecked(True)
        self.save_prob_maps_check = QCheckBox("Save probability maps")
        self.save_prob_maps_check.setChecked(True)

        self._add_info_row(form, "Contact dilation radius (voxels)", self.contact_radius_spin, "contact_radius")
        self._add_info_row(form, "Radial shells", self.n_shells_spin, "n_shells")
        self._add_info_row(form, "Preview nucleus limit, 0=all", self.preview_limit_spin, "preview_limit")
        self._add_info_row(form, "Analysis nucleus limit, 0=all", self.analysis_limit_spin, "analysis_limit")
        self._add_info_row(form, None, self.auto_preview_check, "auto_preview")
        self._add_info_row(form, None, self.save_qc_check, "save_qc")
        self._add_info_row(form, None, self.save_label_masks_check, "save_label_masks")
        self._add_info_row(form, None, self.save_prob_maps_check, "save_prob_maps")

    def _build_action_group(self) -> None:
        form = self._new_group("Actions")
        self.preview_nuclei_btn = QPushButton("Preview nuclei masks")
        self.preview_arms_btn = QPushButton("Preview P/Q arm masks")
        self.analyze_btn = QPushButton("Analyze and save all outputs")
        self.reuse_nuclei_check = QCheckBox("Analyze using current nuclei preview if available")
        self.reuse_nuclei_check.setChecked(True)

        form.addRow(self.preview_nuclei_btn)
        form.addRow(self.preview_arms_btn)
        self._add_info_row(form, None, self.reuse_nuclei_check, "reuse_nuclei")
        form.addRow(self.analyze_btn)

    # ------------------------------------------------------------------
    # Signal wiring and parameters
    # ------------------------------------------------------------------
    def _connect_parameter_signals(self) -> None:
        self.load_image_btn.clicked.connect(self._load_image_clicked)
        self.load_scene_btn.clicked.connect(self._load_selected_scene_clicked)
        self.check_all_scenes_btn.clicked.connect(self._check_all_scenes)
        self.clear_scene_checks_btn.clicked.connect(self._clear_scene_checks)
        self.choose_output_btn.clicked.connect(self._choose_output_clicked)
        self.save_config_btn.clicked.connect(self._save_configuration_clicked)
        self.load_config_btn.clicked.connect(self._load_configuration_clicked)
        self.preview_nuclei_btn.clicked.connect(self._start_nuclei_preview)
        self.preview_arms_btn.clicked.connect(self._start_arm_preview)
        self.analyze_btn.clicked.connect(self._start_analysis)

        for widget in [self.nuc_channel_spin, self.p_channel_spin, self.q_channel_spin]:
            widget.valueChanged.connect(self._channels_changed)

        cellpose_widgets = [
            self.diameter_spin,
            self.xy_downsample_spin,
            self.min_nucleus_volume_spin,
            self.cellpose_mode_combo,
            self.cellpose_batch_spin,
            self.cellpose_overlap_spin,
            self.cellpose_model_combo,
            self.gpu_check,
        ]
        for widget in cellpose_widgets:
            signal = getattr(widget, "valueChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "stateChanged", None)
            if signal is not None:
                signal.connect(self._cellpose_params_changed)

        arm_widgets = [
            self.method_combo,
            self.gmm_components_spin,
            self.auto_bic_check,
            self.p_min_class_spin,
            self.q_min_class_spin,
            self.norm_mode_combo,
            self.bg_percentile_spin,
            self.max_context_voxels_spin,
            self.enable_gate_check,
            self.min_delta_bic_spin,
            self.min_snr_spin,
            self.min_signal_fraction_spin,
            self.max_signal_fraction_spin,
            self.min_mean_posterior_spin,
            self.hard_prob_spin,
            self.arm_smoothing_spin,
            self.apply_morphology_check,
            self.opening_radius_spin,
            self.closing_radius_spin,
            self.min_arm_volume_spin,
            self.component_combo,
            self.component_score_spin,
            self.comp_prob_w_spin,
            self.comp_contrast_w_spin,
            self.comp_volume_w_spin,
            self.comp_boundary_w_spin,
            self.mrf_iterations_spin,
            self.mrf_lambda_spin,
            self.mrf_edge_sigma_spin,
            self.contact_radius_spin,
            self.n_shells_spin,
            self.preview_limit_spin,
            self.parallel_jobs_spin,
            self.parallel_backend_combo,
        ]
        for widget in arm_widgets:
            signal = getattr(widget, "valueChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "stateChanged", None)
            if signal is not None:
                signal.connect(self._arm_params_changed)
        self.gmm_components_spin.valueChanged.connect(self._sync_gmm_class_ranges)

    def _params(self) -> AnalysisParameters:
        return AnalysisParameters(
            nuc_channel=int(self.nuc_channel_spin.value()),
            p_channel=int(self.p_channel_spin.value()),
            q_channel=int(self.q_channel_spin.value()),
            cellpose_model=str(self.cellpose_model_combo.currentText()),
            gpu=bool(self.gpu_check.isChecked()),
            diameter=float(self.diameter_spin.value()),
            xy_downsample=float(self.xy_downsample_spin.value()),
            min_nucleus_volume_um3=float(self.min_nucleus_volume_spin.value()),
            bbox_pad_xy=32,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
            stitch_threshold=0.0,
            cellpose_segmentation_mode=str(self.cellpose_mode_combo.currentText()),
            cellpose_batch_size=int(self.cellpose_batch_spin.value()),
            cellpose_stitch_overlap=float(self.cellpose_overlap_spin.value()),
            parallel_n_jobs=int(self.parallel_jobs_spin.value()),
            parallel_backend=str(self.parallel_backend_combo.currentText()),
            arm_detection_method=str(self.method_combo.currentText()),
            gmm_components=int(self.gmm_components_spin.value()),
            auto_choose_gmm_components_by_bic=bool(self.auto_bic_check.isChecked()),
            p_min_sorted_class=int(self.p_min_class_spin.value()),
            q_min_sorted_class=int(self.q_min_class_spin.value()),
            gmm_covariance_type="full",
            gmm_random_state=0,
            max_voxels_per_nucleus_fit=50000,
            field_normalization_mode=str(self.norm_mode_combo.currentText()),
            background_percentile=float(self.bg_percentile_spin.value()),
            max_context_sample_voxels=int(self.max_context_voxels_spin.value()),
            external_prior_json="",
            use_external_prior_if_available=False,
            arm_smoothing_sigma=float(self.arm_smoothing_spin.value()),
            apply_binary_morphology=bool(self.apply_morphology_check.isChecked()),
            arm_opening_radius=int(self.opening_radius_spin.value()),
            arm_closing_radius=int(self.closing_radius_spin.value()),
            min_arm_volume_um3=float(self.min_arm_volume_spin.value()),
            keep_largest_arm_component=False,
            enable_presence_gate=bool(self.enable_gate_check.isChecked()),
            min_delta_bic=float(self.min_delta_bic_spin.value()),
            min_signal_snr=float(self.min_snr_spin.value()),
            min_signal_fraction=float(self.min_signal_fraction_spin.value()),
            max_signal_fraction=float(self.max_signal_fraction_spin.value()),
            min_mean_posterior=float(self.min_mean_posterior_spin.value()),
            hard_probability_threshold=float(self.hard_prob_spin.value()),
            use_hysteresis=False,
            probability_low_threshold=0.35,
            probability_high_threshold=0.75,
            component_selection=str(self.component_combo.currentText()),
            component_score_threshold=float(self.component_score_spin.value()),
            component_weight_probability=float(self.comp_prob_w_spin.value()),
            component_weight_contrast=float(self.comp_contrast_w_spin.value()),
            component_weight_volume=float(self.comp_volume_w_spin.value()),
            component_weight_boundary_penalty=float(self.comp_boundary_w_spin.value()),
            spatial_coordinate_weight=0.0,
            spatial_radial_weight=0.0,
            spatial_contrast_weight=0.0,
            spatial_gradient_weight=0.0,
            use_mrf_refinement=str(self.method_combo.currentText()) == "MRF/CRF refinement",
            mrf_iterations=int(self.mrf_iterations_spin.value()),
            mrf_lambda=float(self.mrf_lambda_spin.value()),
            mrf_edge_sigma=float(self.mrf_edge_sigma_spin.value()),
            classifier_threshold=0.50,
            classifier_bias=0.0,
            classifier_peak_z_threshold=3.0,
            contact_dilation_radius=int(self.contact_radius_spin.value()),
            n_shells=int(self.n_shells_spin.value()),
            limit_nuclei=int(self.analysis_limit_spin.value()),
            save_qc=bool(self.save_qc_check.isChecked()),
            save_label_masks=bool(self.save_label_masks_check.isChecked()),
            save_probability_maps=bool(self.save_prob_maps_check.isChecked()),
        )

    def _apply_params_to_widgets(self, p: dict) -> None:
        self._loading_config = True
        try:
            def set_spin(widget, key):
                if key in p:
                    widget.setValue(p[key])

            def set_check(widget, key):
                if key in p:
                    widget.setChecked(bool(p[key]))

            def set_combo(widget, key):
                if key in p:
                    txt = str(p[key])
                    idx = widget.findText(txt)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)

            set_spin(self.nuc_channel_spin, "nuc_channel")
            set_spin(self.p_channel_spin, "p_channel")
            set_spin(self.q_channel_spin, "q_channel")
            set_combo(self.cellpose_model_combo, "cellpose_model")
            set_check(self.gpu_check, "gpu")
            set_spin(self.diameter_spin, "diameter")
            set_spin(self.xy_downsample_spin, "xy_downsample")
            set_spin(self.min_nucleus_volume_spin, "min_nucleus_volume_um3")
            set_combo(self.cellpose_mode_combo, "cellpose_segmentation_mode")
            set_spin(self.cellpose_batch_spin, "cellpose_batch_size")
            set_spin(self.cellpose_overlap_spin, "cellpose_stitch_overlap")
            set_spin(self.parallel_jobs_spin, "parallel_n_jobs")
            set_combo(self.parallel_backend_combo, "parallel_backend")
            set_combo(self.method_combo, "arm_detection_method")
            set_spin(self.gmm_components_spin, "gmm_components")
            set_check(self.auto_bic_check, "auto_choose_gmm_components_by_bic")
            self._sync_gmm_class_ranges()
            set_spin(self.p_min_class_spin, "p_min_sorted_class")
            set_spin(self.q_min_class_spin, "q_min_sorted_class")
            set_combo(self.norm_mode_combo, "field_normalization_mode")
            set_spin(self.bg_percentile_spin, "background_percentile")
            set_spin(self.max_context_voxels_spin, "max_context_sample_voxels")
            set_spin(self.arm_smoothing_spin, "arm_smoothing_sigma")
            set_check(self.apply_morphology_check, "apply_binary_morphology")
            set_spin(self.opening_radius_spin, "arm_opening_radius")
            set_spin(self.closing_radius_spin, "arm_closing_radius")
            set_spin(self.min_arm_volume_spin, "min_arm_volume_um3")
            set_check(self.enable_gate_check, "enable_presence_gate")
            set_spin(self.min_delta_bic_spin, "min_delta_bic")
            set_spin(self.min_snr_spin, "min_signal_snr")
            set_spin(self.min_signal_fraction_spin, "min_signal_fraction")
            set_spin(self.max_signal_fraction_spin, "max_signal_fraction")
            set_spin(self.min_mean_posterior_spin, "min_mean_posterior")
            set_spin(self.hard_prob_spin, "hard_probability_threshold")
            set_combo(self.component_combo, "component_selection")
            set_spin(self.component_score_spin, "component_score_threshold")
            set_spin(self.comp_prob_w_spin, "component_weight_probability")
            set_spin(self.comp_contrast_w_spin, "component_weight_contrast")
            set_spin(self.comp_volume_w_spin, "component_weight_volume")
            set_spin(self.comp_boundary_w_spin, "component_weight_boundary_penalty")
            set_spin(self.mrf_iterations_spin, "mrf_iterations")
            set_spin(self.mrf_lambda_spin, "mrf_lambda")
            set_spin(self.mrf_edge_sigma_spin, "mrf_edge_sigma")
            set_spin(self.contact_radius_spin, "contact_dilation_radius")
            set_spin(self.n_shells_spin, "n_shells")
            set_spin(self.analysis_limit_spin, "limit_nuclei")
            set_check(self.save_qc_check, "save_qc")
            set_check(self.save_label_masks_check, "save_label_masks")
            set_check(self.save_prob_maps_check, "save_probability_maps")
        finally:
            self._loading_config = False
        self._sync_gmm_class_ranges()
        self._nuclei_dirty = True

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def append_status(self, message: str) -> None:
        self.status_box.appendPlainText(str(message))

    def _worker_is_running(self, worker) -> bool:
        if worker is None:
            return False
        flag = getattr(worker, "is_running", False)
        return bool(flag() if callable(flag) else flag)

    def _set_main_busy(self, busy: bool, message: str | None = None) -> None:
        for btn in [
            self.load_image_btn,
            self.load_scene_btn,
            self.preview_nuclei_btn,
            self.analyze_btn,
            self.save_config_btn,
            self.load_config_btn,
        ]:
            btn.setEnabled(not busy)
        if message:
            self.append_status(message)

    def _selected_images(self):
        if self._data_czyx is None:
            raise RuntimeError("No scene has been loaded yet.")
        params = self._params()
        return (
            np.asarray(self._data_czyx[params.nuc_channel - 1]),
            np.asarray(self._data_czyx[params.p_channel - 1]),
            np.asarray(self._data_czyx[params.q_channel - 1]),
        )

    def _sync_gmm_class_ranges(self, *_args) -> None:
        max_idx = max(0, int(self.gmm_components_spin.value()) - 1)
        for spin in [self.p_min_class_spin, self.q_min_class_spin]:
            old = int(spin.value())
            spin.blockSignals(True)
            spin.setRange(0, max_idx)
            spin.setValue(min(old, max_idx))
            spin.blockSignals(False)

    def _contrast_limits(self, data: np.ndarray):
        arr = np.asarray(data)
        if arr.size == 0:
            return (0, 1)
        lo, hi = np.nanpercentile(arr.astype(np.float32), [0.5, 99.8])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))
            if hi <= lo:
                hi = lo + 1.0
        return (float(lo), float(hi))

    def _get_layer(self, name: str):
        try:
            return self.viewer.layers[name]
        except Exception:
            return None

    def _remove_layer(self, name: str) -> None:
        try:
            layer = self.viewer.layers[name]
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _remove_unwanted_preview_layers(self) -> None:
        for name in ["P arm probability preview", "Q arm probability preview", "Nuclei mask preview"]:
            self._remove_layer(name)

    def _layer_type_name(self, layer) -> str:
        return layer.__class__.__name__.lower() if layer is not None else ""

    def _set_image_layer(self, name: str, data: np.ndarray, colormap: str = "gray", opacity: float = 1.0) -> None:
        layer = self._get_layer(name)
        if layer is not None and "image" not in self._layer_type_name(layer):
            self._remove_layer(name)
            layer = None
        if layer is not None:
            layer.data = data
            layer.contrast_limits = self._contrast_limits(data)
            layer.opacity = opacity
            layer.visible = True
            try:
                layer.colormap = colormap
            except Exception:
                pass
        else:
            self.viewer.add_image(
                data,
                name=name,
                colormap=colormap,
                blending="additive",
                opacity=opacity,
                contrast_limits=self._contrast_limits(data),
            )

    def _set_mask_image_layer(self, name: str, data: np.ndarray, colormap: str, opacity: float = 0.45) -> None:
        mask = np.asarray(data)
        if mask.dtype == bool:
            mask = mask.astype(np.float32)
        else:
            mask = (mask > 0).astype(np.float32)
        self._set_image_layer(name, mask, colormap=colormap, opacity=opacity)

    def _set_labels_layer(self, name: str, data: np.ndarray, opacity: float = 0.45) -> None:
        layer = self._get_layer(name)
        labels = np.asarray(data)
        if layer is not None and "labels" not in self._layer_type_name(layer):
            self._remove_layer(name)
            layer = None
        if layer is not None:
            layer.data = labels
            layer.opacity = opacity
            layer.visible = True
        else:
            self.viewer.add_labels(labels, name=name, opacity=opacity)

    def _show_channel_layers(self) -> None:
        if self._data_czyx is None:
            return
        self._remove_unwanted_preview_layers()
        nuc_img, p_img, q_img = self._selected_images()
        self._set_image_layer("Nucleus channel", nuc_img, colormap="gray", opacity=0.85)
        self._set_image_layer("P arm channel", p_img, colormap="magenta", opacity=0.65)
        self._set_image_layer("Q arm channel", q_img, colormap="cyan", opacity=0.65)
        try:
            self.viewer.dims.ndisplay = 3
        except Exception:
            pass

    def _update_channel_ranges(self) -> None:
        if self._data_czyx is None:
            return
        c = int(self._data_czyx.shape[0])
        defaults = [1, min(2, c), min(3, c)]
        for spin, default in zip([self.nuc_channel_spin, self.p_channel_spin, self.q_channel_spin], defaults):
            spin.blockSignals(True)
            spin.setRange(1, max(1, c))
            if spin.value() > c:
                spin.setValue(default)
            spin.blockSignals(False)

    # ------------------------------------------------------------------
    # File, scene, and configuration handling
    # ------------------------------------------------------------------
    def _populate_scenes(self, scenes: list[str]) -> None:
        self._scene_names = [str(s) for s in scenes]
        self.scene_combo.clear()
        self.scene_combo.addItems(self._scene_names)
        self.scene_list.clear()
        for i, name in enumerate(self._scene_names):
            item = QListWidgetItem(f"{i + 1}: {name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            self.scene_list.addItem(item)

    def _checked_scene_indices(self) -> list[int]:
        indices: list[int] = []
        for i in range(self.scene_list.count()):
            if self.scene_list.item(i).checkState() == Qt.Checked:
                indices.append(i)
        return indices

    def _set_checked_scene_indices(self, indices: list[int]) -> None:
        wanted = {int(i) for i in indices}
        for i in range(self.scene_list.count()):
            self.scene_list.item(i).setCheckState(Qt.Checked if i in wanted else Qt.Unchecked)

    def _check_all_scenes(self) -> None:
        self._set_checked_scene_indices(list(range(self.scene_list.count())))

    def _clear_scene_checks(self) -> None:
        self._set_checked_scene_indices([])

    def _load_image_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image file", "", SUPPORTED_IMAGE_FILTER)
        if not path:
            return
        self._open_image_path(Path(path))

    def _open_image_path(self, path: Path) -> None:
        self._image_path = Path(path)
        self.image_label.setText(str(self._image_path))
        try:
            scenes = get_scene_names(self._image_path)
        except Exception as exc:
            self.append_status(f"ERROR while reading scenes: {exc}")
            return
        self._populate_scenes(scenes)
        default_out = self._image_path.parent / f"{self._image_path.stem}_pq_analysis"
        if self._output_dir is None:
            self._output_dir = default_out
            self.output_label.setText(str(default_out))
        self.append_status(f"Found {len(scenes)} scene(s). Load one scene for preview, and optionally check scenes for batch analysis.")

    def _choose_output_clicked(self) -> None:
        start = str(self._output_dir or (self._image_path.parent if self._image_path else Path.home()))
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if folder:
            self._output_dir = Path(folder)
            self.output_label.setText(str(self._output_dir))

    def _load_selected_scene_clicked(self) -> None:
        if self._image_path is None:
            self.append_status("Load an image first.")
            return
        scene_index = int(self.scene_combo.currentIndex())
        self._set_main_busy(True, f"Loading scene {scene_index + 1}...")

        def job():
            data, spacing, scene_name = load_scene_channels(self._image_path, scene_index)
            return scene_index, data, spacing, scene_name

        worker = thread_worker(job)()
        worker.returned.connect(self._on_scene_loaded)
        worker.errored.connect(self._on_worker_error)
        self._main_worker = worker
        worker.start()

    def _on_scene_loaded(self, result) -> None:
        self._scene_index, self._data_czyx, self._spacing_zyx, self._scene_name = result
        self._nuclei_labels = None
        self._nuclei_dirty = True
        self._update_channel_ranges()
        self._show_channel_layers()
        self._set_main_busy(False)
        self.append_status(
            f"Loaded scene '{self._scene_name}' with CZYX shape {tuple(self._data_czyx.shape)} "
            f"and spacing ZYX={tuple(round(v, 4) for v in self._spacing_zyx)} um."
        )

    def _configuration_dict(self) -> dict:
        return {
            "plugin": "napari-pq-arm-analyzer",
            "plugin_version": PLUGIN_VERSION,
            "image_path": str(self._image_path) if self._image_path else None,
            "output_dir": str(self._output_dir) if self._output_dir else None,
            "preview_scene_index": self._scene_index,
            "selected_scene_indices": self._checked_scene_indices(),
            "parameters": asdict(self._params()),
            "ui_options": {
                "preview_nucleus_limit": int(self.preview_limit_spin.value()),
                "auto_live_arm_preview": bool(self.auto_preview_check.isChecked()),
                "auto_cellpose_preview": bool(self.auto_cellpose_check.isChecked()),
                "reuse_current_nuclei": bool(self.reuse_nuclei_check.isChecked()),
            },
        }

    def _save_configuration_clicked(self) -> None:
        start_dir = str(self._output_dir or (self._image_path.parent if self._image_path else Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, "Save P/Q analyzer configuration", start_dir, "JSON configuration (*.json);;All files (*)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._configuration_dict(), f, indent=2)
        self.append_status(f"Saved configuration: {path}")

    def _load_configuration_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load P/Q analyzer configuration", "", "JSON configuration (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            self.append_status(f"ERROR loading configuration: {exc}")
            return
        params = cfg.get("parameters", cfg)
        if isinstance(params, dict):
            self._apply_params_to_widgets(params)
        ui = cfg.get("ui_options", {}) if isinstance(cfg, dict) else {}
        self._loading_config = True
        try:
            if "preview_nucleus_limit" in ui:
                self.preview_limit_spin.setValue(int(ui["preview_nucleus_limit"]))
            if "auto_live_arm_preview" in ui:
                self.auto_preview_check.setChecked(bool(ui["auto_live_arm_preview"]))
            if "auto_cellpose_preview" in ui:
                self.auto_cellpose_check.setChecked(bool(ui["auto_cellpose_preview"]))
            if "reuse_current_nuclei" in ui:
                self.reuse_nuclei_check.setChecked(bool(ui["reuse_current_nuclei"]))
        finally:
            self._loading_config = False
        if cfg.get("output_dir"):
            self._output_dir = Path(cfg["output_dir"])
            self.output_label.setText(str(self._output_dir))
        img_path = cfg.get("image_path")
        if img_path and Path(img_path).exists():
            self._open_image_path(Path(img_path))
            selected = cfg.get("selected_scene_indices", [])
            if isinstance(selected, list):
                self._set_checked_scene_indices([int(i) for i in selected])
            preview_idx = cfg.get("preview_scene_index")
            if preview_idx is not None and 0 <= int(preview_idx) < self.scene_combo.count():
                self.scene_combo.setCurrentIndex(int(preview_idx))
        self.append_status(f"Loaded configuration: {path}")
        self.append_status("After loading a configuration, load a preview scene and rerun nuclei preview before live P/Q preview.")

    # ------------------------------------------------------------------
    # Parameter changes and live preview
    # ------------------------------------------------------------------
    def _channels_changed(self, *_args) -> None:
        if self._loading_config:
            return
        if self._data_czyx is None:
            return
        self._nuclei_labels = None
        self._nuclei_dirty = True
        self._show_channel_layers()
        self.append_status("Channels changed; nuclei preview was cleared.")

    def _cellpose_params_changed(self, *_args) -> None:
        if self._loading_config:
            return
        self._nuclei_dirty = True
        if self._data_czyx is None:
            return
        if self.auto_cellpose_check.isChecked():
            self._nuclei_preview_timer.start()
        else:
            self.append_status("Cellpose parameters changed; click 'Preview nuclei masks' to update nuclei.")

    def _arm_params_changed(self, *_args) -> None:
        if self._loading_config:
            return
        if self.sender() is self.gmm_components_spin:
            self._sync_gmm_class_ranges()
        if self.auto_preview_check.isChecked():
            self._arm_preview_timer.start()

    # ------------------------------------------------------------------
    # Nuclei and arm previews
    # ------------------------------------------------------------------
    def _start_nuclei_preview(self) -> None:
        if self._data_czyx is None:
            self.append_status("Load a scene first.")
            return
        if self._worker_is_running(self._main_worker):
            self.append_status("A long-running task is already active.")
            return

        nuc_img, _p_img, _q_img = self._selected_images()
        spacing = self._spacing_zyx
        params = self._params()
        self._set_main_busy(True, "Running Cellpose nuclei preview...")

        def job():
            return segment_nuclei_cellpose_3d(nuc_img, spacing, params)

        worker = thread_worker(job)()
        worker.returned.connect(self._on_nuclei_preview_returned)
        worker.errored.connect(self._on_worker_error)
        self._main_worker = worker
        worker.start()

    def _on_nuclei_preview_returned(self, labels: np.ndarray) -> None:
        self._nuclei_labels = np.asarray(labels, dtype=np.int32)
        self._nuclei_dirty = False
        self._remove_unwanted_preview_layers()
        self._set_labels_layer("Nuclei labels preview", self._nuclei_labels, opacity=0.35)
        self._set_main_busy(False)
        n = int(len(np.unique(self._nuclei_labels)) - (1 if np.any(self._nuclei_labels == 0) else 0))
        self.append_status(f"Nuclei preview updated: {n} nuclei.")
        if self.auto_preview_check.isChecked():
            self._arm_preview_timer.start()

    def _start_arm_preview(self) -> None:
        if self._data_czyx is None:
            self.append_status("Load a scene first.")
            return
        if self._nuclei_labels is None:
            self.append_status("Run the nuclei preview before previewing P/Q arms.")
            return
        if self._preview_busy:
            self._preview_pending = True
            return

        _nuc_img, p_img, q_img = self._selected_images()
        labels = self._nuclei_labels
        spacing = self._spacing_zyx
        params = self._params()
        preview_limit = int(self.preview_limit_spin.value())
        self._preview_busy = True
        self.preview_arms_btn.setEnabled(False)
        self.append_status(
            f"Updating P/Q arm preview using '{params.arm_detection_method}'"
            + (f" for first {preview_limit} nuclei..." if preview_limit else " for all nuclei...")
        )

        def job():
            return segment_arms_for_nuclei(
                labels,
                p_img,
                q_img,
                spacing,
                params,
                limit_nuclei=preview_limit,
                return_rows=False,
            )

        worker = thread_worker(job)()
        worker.returned.connect(self._on_arm_preview_returned)
        worker.errored.connect(self._on_arm_preview_error)
        self._preview_worker = worker
        worker.start()

    def _on_arm_preview_returned(self, result) -> None:
        p_mask, q_mask, _p_labels, _q_labels, _p_prob, _q_prob = result
        self._remove_unwanted_preview_layers()
        overlap = np.asarray(p_mask, dtype=bool) & np.asarray(q_mask, dtype=bool)
        self._set_mask_image_layer("P arm mask preview", np.asarray(p_mask, dtype=bool), colormap="magenta", opacity=0.45)
        self._set_mask_image_layer("Q arm mask preview", np.asarray(q_mask, dtype=bool), colormap="cyan", opacity=0.45)
        self._set_mask_image_layer("P/Q overlap preview", overlap.astype(bool), colormap="yellow", opacity=0.55)
        self._preview_busy = False
        self.preview_arms_btn.setEnabled(True)
        self.append_status("P/Q arm preview updated. Only binary P/Q/overlap masks are shown in the layer panel.")
        if self._preview_pending:
            self._preview_pending = False
            self._arm_preview_timer.start(100)

    def _on_arm_preview_error(self, exc) -> None:
        self._preview_busy = False
        self.preview_arms_btn.setEnabled(True)
        self.append_status(f"ERROR during P/Q arm preview: {exc}")
        if self._preview_pending:
            self._preview_pending = False
            self._arm_preview_timer.start(100)

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------
    def _scene_output_dir(self, base: Path, scene_index: int, scene_name: str) -> Path:
        return Path(base) / f"scene_{scene_index + 1:03d}_{_token(scene_name)}"

    def _start_analysis(self) -> None:
        if self._image_path is None:
            self.append_status("Load an image first.")
            return
        if self._output_dir is None:
            self._choose_output_clicked()
            if self._output_dir is None:
                return
        if self._worker_is_running(self._main_worker):
            self.append_status("A long-running task is already active.")
            return

        checked = self._checked_scene_indices()
        params = self._params()
        base_outdir = Path(self._output_dir)

        if checked:
            current_existing = None
            current_index = self._scene_index
            if self.reuse_nuclei_check.isChecked() and self._nuclei_labels is not None and not self._nuclei_dirty:
                current_existing = np.asarray(self._nuclei_labels, dtype=np.int32)
            selected_names = [self._scene_names[i] if i < len(self._scene_names) else f"Scene {i + 1}" for i in checked]
            batch_cfg = self._configuration_dict()
            batch_cfg["selected_scene_names"] = selected_names
            self._set_main_busy(True, f"Running batch analysis for {len(checked)} checked scene(s). Base output: {base_outdir}")

            def job():
                base_outdir.mkdir(parents=True, exist_ok=True)
                with open(base_outdir / "batch_pq_arm_analyzer_configuration.json", "w", encoding="utf-8") as f:
                    json.dump(batch_cfg, f, indent=2)
                outputs = []
                for idx in checked:
                    data, spacing, scene_name = load_scene_channels(self._image_path, idx)
                    scene_out = self._scene_output_dir(base_outdir, idx, scene_name)
                    existing = current_existing if (current_existing is not None and current_index == idx) else None
                    outputs.append(
                        run_full_analysis(
                            data,
                            spacing,
                            scene_out,
                            params,
                            source_image_path=self._image_path,
                            scene_name=scene_name,
                            existing_nuclei_labels=existing,
                        )
                    )
                return outputs

            worker = thread_worker(job)()
            worker.returned.connect(self._on_batch_analysis_returned)
            worker.errored.connect(self._on_worker_error)
            self._main_worker = worker
            worker.start()
            return

        if self._data_czyx is None:
            self.append_status("No scenes are checked, so Analyze uses the current preview scene. Load a scene for preview first, or check scenes for batch analysis.")
            return

        data = self._data_czyx
        spacing = self._spacing_zyx
        outdir = base_outdir
        existing = None
        if self.reuse_nuclei_check.isChecked() and self._nuclei_labels is not None and not self._nuclei_dirty:
            existing = np.asarray(self._nuclei_labels, dtype=np.int32)
            self.append_status("Analyze will reuse the current nuclei preview.")
        elif self._nuclei_labels is not None and self._nuclei_dirty:
            self.append_status("Nuclei parameters changed after preview; Analyze will rerun Cellpose.")

        single_cfg = self._configuration_dict()
        self._set_main_busy(True, f"Running full analysis with '{params.arm_detection_method}'. Outputs will be saved to: {outdir}")

        def job():
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir / "pq_arm_analyzer_configuration.json", "w", encoding="utf-8") as f:
                json.dump(single_cfg, f, indent=2)
            return run_full_analysis(
                data,
                spacing,
                outdir,
                params,
                source_image_path=self._image_path,
                scene_name=self._scene_name,
                existing_nuclei_labels=existing,
            )

        worker = thread_worker(job)()
        worker.returned.connect(self._on_analysis_returned)
        worker.errored.connect(self._on_worker_error)
        self._main_worker = worker
        worker.start()

    def _on_analysis_returned(self, outputs: AnalysisOutputs) -> None:
        self._set_main_busy(False)
        self.append_status(f"Analysis complete: {outputs.n_nuclei} nuclei measured.")
        self.append_status(f"Per-nucleus CSV: {outputs.per_nucleus_csv}")
        self.append_status(f"Per-object CSV: {outputs.per_object_csv}")
        self.append_status(f"Plots: {outputs.plot_dir}")
        try:
            from tifffile import imread

            self._nuclei_labels = np.asarray(imread(outputs.nuclei_labels_tif), dtype=np.int32)
            p_mask = np.asarray(imread(outputs.p_mask_tif)) > 0
            q_mask = np.asarray(imread(outputs.q_mask_tif)) > 0
            ov_mask = np.asarray(imread(outputs.overlap_mask_tif)) > 0
            self._nuclei_dirty = False
            self._remove_unwanted_preview_layers()
            self._set_labels_layer("Nuclei labels preview", self._nuclei_labels, opacity=0.35)
            self._set_mask_image_layer("P arm mask preview", p_mask.astype(bool), colormap="magenta", opacity=0.45)
            self._set_mask_image_layer("Q arm mask preview", q_mask.astype(bool), colormap="cyan", opacity=0.45)
            self._set_mask_image_layer("P/Q overlap preview", ov_mask.astype(bool), colormap="yellow", opacity=0.55)
        except Exception as exc:
            self.append_status(f"Analysis saved successfully, but layers were not reloaded: {exc}")

    def _on_batch_analysis_returned(self, outputs_list: list[AnalysisOutputs]) -> None:
        self._set_main_busy(False)
        self.append_status(f"Batch analysis complete: {len(outputs_list)} scene(s) analyzed.")
        for out in outputs_list:
            self.append_status(f"  {out.output_dir} ({out.n_nuclei} nuclei)")

    def _on_worker_error(self, exc) -> None:
        self._set_main_busy(False)
        self.append_status(f"ERROR: {exc}")


def make_widget(viewer=None):
    """Factory used by napari plugin discovery."""
    return PQArmAnalyzerWidget(viewer=viewer)
