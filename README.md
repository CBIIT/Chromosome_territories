# napari P/Q Arm Analyzer

![P/Q Arm Analyzer UI](docs/images/plugin_screenshot.png)

Interactive napari plugin for 3D nuclei segmentation, P/Q-arm detection, batch scene analysis, mask export, CSV measurement export, and summary plot generation from multi-channel microscopy images.

**Author and repository owner**  
Adib Keikhosravi, Ph.D.  
Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI  
National Institutes of Health  
Email: adib.keikhosravi@nih.gov

**License:** MIT License. See [`LICENSE`](LICENSE).

---

## Current version

**v0.3.5**

This version is a streamlined routine-analysis version. It keeps only the three P/Q arm-detection methods used for practical tuning:

```text
Legacy 1D GMM
Upgraded 1D GMM + gate + scoring
MRF/CRF refinement
```

It also includes scene checkboxes for batch analysis, configuration save/load, simplified napari visualization layers, detailed parameter help buttons, expanded 3D shape/radial measurements, and a public result table with internal tuning/debug columns removed.

---

## Main screenshot

The README image points to:

```text
docs/images/plugin_screenshot.png
```

Replace that file with a screenshot exported from the plugin, keeping the same filename, and GitHub will automatically show it at the top of this README.

---

## Full manual

A complete DOCX manual is included here:

```text
docs/manual/PQ_Arm_Analyzer_User_Manual_v0.3.5.docx
```

The manual contains the full step-by-step workflow, detailed explanations of every visible GUI parameter, tuning recipes, output-file descriptions, result-column explanations, mathematical formulas, and detailed appendices on MRF/CRF refinement and component selection.

---

## What the plugin does

The plugin provides an interactive workflow in napari:

1. Load a multi-channel microscopy image.
2. Select a scene, series, field, position, or tile.
3. Choose the nucleus, P-arm, and Q-arm channels.
4. Segment nuclei with Cellpose.
5. Detect P and Q arm territories inside each nucleus.
6. Preview the masks in napari.
7. Tune parameters using the live preview and the parameter help buttons.
8. Save a reusable configuration JSON.
9. Analyze one scene or multiple checked scenes.
10. Export masks, measurements, plots, QC files, and configuration files.

---

# Installation

This section gives a complete installation path starting from a clean machine or clean conda environment.

The plugin is a napari plugin, so it should be installed into the same Python environment that will launch napari.

## 1. Install prerequisites

You need:

```text
Git
Conda or Mamba
Python 3.9 or newer
A working graphical desktop session for napari
```

Recommended: use a fresh conda environment rather than installing into an existing analysis environment.

### Install Miniconda or Anaconda

Install Miniconda or Anaconda if you do not already have conda available. After installation, open a new terminal and confirm:

```bash
conda --version
```

If you prefer mamba, install it into your base conda environment:

```bash
conda install -n base -c conda-forge mamba -y
```

All commands below use `conda`, but you can replace `conda` with `mamba` for faster solving.

---

## 2. Create a fresh environment

Recommended environment name:

```text
pq-arm-analyzer
```

Create and activate it:

```bash
conda create -n pq-arm-analyzer python=3.9 -y
conda activate pq-arm-analyzer
```

Upgrade the basic Python build tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Confirm that the terminal is using the environment you just created:

```bash
python --version
python -m pip --version
```

---

## 3. Install napari with a Qt backend

napari needs a Qt backend. The recommended pip route is:

```bash
python -m pip install "napari[pyqt5]"
```

If that command fails on your system, try installing napari and PyQt separately:

```bash
python -m pip install napari pyqt5
```

You can verify napari launches before installing the plugin:

```bash
napari
```

Close napari after confirming that the main window opens.

---

## 4. Decide whether you need CPU or GPU Cellpose

The plugin uses Cellpose for nuclei segmentation. Cellpose uses PyTorch internally.

### CPU-only installation

For CPU-only use, no special GPU setup is required. Installing the plugin will install Cellpose and its Python dependencies.

### GPU installation

For GPU acceleration, install a PyTorch build compatible with your NVIDIA driver and CUDA setup before installing the plugin. The exact PyTorch command depends on your workstation and CUDA version. After installing the GPU-enabled PyTorch build, confirm:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('torch:', torch.__version__)"
```

You want:

```text
CUDA available: True
```

If CUDA is not available, the plugin can still run with CPU Cellpose, but nuclei segmentation will usually be slower.

---

## 5. Clone the repository from GitHub

After this repository has been uploaded to GitHub, clone it with:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/napari-pq-arm-analyzer.git
cd napari-pq-arm-analyzer
```

Replace `<YOUR_GITHUB_USERNAME>` with the GitHub account or organization name that owns the repository.

For example:

```bash
git clone https://github.com/adib-keikhosravi/napari-pq-arm-analyzer.git
cd napari-pq-arm-analyzer
```

If your repository is private, make sure you are logged in with GitHub credentials or have configured SSH access.

### Alternative: install from a downloaded ZIP

If you have a ZIP file instead of a GitHub clone:

```bash
unzip napari_pq_arm_analyzer_repository_v0_3_5.zip
cd napari-pq-arm-analyzer
```

---

## 6. Install the plugin

From the repository root, install the plugin into the active conda environment.

Recommended editable installation:

```bash
python -m pip install -e .
```

Editable mode means that Python uses the code directly from this folder. This is convenient for development and for updating the plugin without repeatedly copying files.

### Install with optional file readers

For broader microscopy file support, install optional reader packages:

```bash
python -m pip install -e ".[extra-readers]"
```

This installs optional support packages for formats such as LIF, ND2, and IMS when available:

```text
readlif
nd2reader
imaris-ims-file-reader
```

If one optional reader fails to install on your system, you can still install the base plugin:

```bash
python -m pip install -e .
```

Then install optional readers individually as needed:

```bash
python -m pip install readlif
python -m pip install nd2reader
python -m pip install imaris-ims-file-reader
```

---

## 7. Verify the installation

From the same activated conda environment, run:

```bash
python -c "import napari_pq_arm_analyzer; print(napari_pq_arm_analyzer.__version__)"
```

Then start napari:

```bash
napari
```

Open the plugin from the napari menu:

```text
Plugins > P/Q Arm Analyzer > P/Q Arm Analyzer
```

If the plugin appears in the menu, the installation was successful.

---

## 8. Test with a small image or one scene first

Before running a large batch:

1. Open napari.
2. Open the plugin.
3. Click **Load image...**.
4. Select a file.
5. Load one scene for preview.
6. Select the nucleus, P-arm, and Q-arm channels.
7. Run **Preview nuclei masks**.
8. Run **Preview P/Q arm masks**.
9. Save outputs to a test folder.

After this test works, use the scene checkboxes for batch analysis.

---

## 9. Updating the plugin later

If the repository was installed from GitHub:

```bash
conda activate pq-arm-analyzer
cd napari-pq-arm-analyzer
git pull
python -m pip install -e .
```

If optional readers are used:

```bash
python -m pip install -e ".[extra-readers]"
```

Restart napari after updating.

---

## 10. Clean reinstall

If napari is showing an old plugin version or the plugin menu is inconsistent, reinstall cleanly:

```bash
conda activate pq-arm-analyzer
python -m pip uninstall -y napari-pq-arm-analyzer
cd napari-pq-arm-analyzer
python -m pip install -e .
napari
```

For optional readers:

```bash
python -m pip install -e ".[extra-readers]"
```

---

## 11. Creating the GitHub repository from the ZIP file

If you received the repository as a ZIP file and want to upload it to GitHub:

```bash
unzip napari_pq_arm_analyzer_repository_v0_3_5.zip
cd napari-pq-arm-analyzer

git init
git add .
git commit -m "Initial commit: napari P/Q Arm Analyzer"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/napari-pq-arm-analyzer.git
git push -u origin main
```

After pushing to GitHub, users can install with the clone workflow above.

---

# Supported image formats

The plugin uses AICSImageIO first when possible and includes fallback reader logic for common microscopy formats. Supported paths include:

```text
.lif
.czi
.nd2
.ims
.ome.tif / .ome.tiff
.tif / .tiff
.lsm
.zarr
```

Files with multiple scenes, fields, series, positions, or tiles are exposed in the scene dropdown and the scene checkbox list. You can load one scene for interactive preview and check multiple scenes for batch analysis.

Optional reader packages may be required for some files:

```bash
python -m pip install readlif nd2reader imaris-ims-file-reader
```

---

# Repository layout

```text
napari-pq-arm-analyzer/
  README.md
  LICENSE
  CHANGELOG.md
  pyproject.toml
  MANIFEST.in
  src/napari_pq_arm_analyzer/
    __init__.py
    _widget.py
    analysis.py
    help_text.py
    image_io.py
    plotting.py
    napari.yaml
  docs/manual/
    PQ_Arm_Analyzer_User_Manual_v0.3.5.docx
  docs/images/
    plugin_screenshot.png
  examples/
    launch_widget.py
```

---

# GUI layers shown in napari

This version intentionally keeps the layer panel simple. It does **not** add P-arm probability preview, Q-arm probability preview, or Nuclei mask preview layers.

Shown layers are:

```text
Nucleus channel
P arm channel
Q arm channel
Nuclei labels preview
P arm mask preview
Q arm mask preview
P/Q overlap preview
```

The P, Q, and overlap masks are shown as napari Image layers, so colormap changes behave like raw channel color changes. The nuclei preview remains a Labels layer because it stores object identities.

---

# Configuration save/load

The GUI includes:

```text
Save configuration...
Load configuration...
```

A configuration JSON stores visible parameter values, output folder, image path, selected preview scene, checked batch scenes, and useful UI options.

Every final analysis folder receives:

```text
pq_arm_analyzer_configuration.json
analysis_parameters.json
```

Batch analysis also writes:

```text
batch_pq_arm_analyzer_configuration.json
```

Use configuration files to reuse the same parameters across future images or to document exactly how a result folder was generated.

---

# Arm-detection methods

## Legacy 1D GMM

Baseline intensity-only method. It fits Gaussian intensity classes inside each nucleus and keeps selected bright classes. It is useful for comparison and very clean images, but it can segment noise because it always tries to split intensities into classes.

## Upgraded 1D GMM + gate + scoring

Recommended default. It adds field-level robust normalization, an explicit presence/absence gate, probability thresholding, and connected-component scoring. This is usually the best starting method when some nuclei may have weak signal or no convincing P/Q signal.

## MRF/CRF refinement

Starts from the upgraded GMM probability map and adds spatial refinement. Neighboring voxels are encouraged to agree, but strong intensity edges reduce smoothing across boundaries. Use this when the upgraded method finds the correct general region but boundaries are ragged, speckled, or have small holes.

---

# Quick-start settings for a new image set

A practical starting point for large multi-slice images is:

```text
Nuclei segmentation mode: Cellpose 2D stack batch + overlap stitch
XY downsample factor: 2.0 for preview, lower if boundaries need more precision
2D batch size: 8, reduce if memory is limited
Worker count: 0 or a moderate number
Backend: threading
Preview nucleus limit: a modest number while tuning
Arm method: Upgraded 1D GMM + gate + scoring
Max GMM components: 2
P/Q sorted class index: 1
Normalization mode: mixed_outside_or_low_percentile
Presence gate: enabled
Probability threshold: 0.50
Component selection: best_score
Binary morphology: off
```

After the preview looks good, set **Analysis nucleus limit** to 0 for the final run.

---

# Main output files

A single-scene analysis writes:

```text
nuclei_labels_3d.tif
p_arm_mask_3d.tif
q_arm_mask_3d.tif
pq_overlap_mask_3d.tif
p_arm_probability_3d.tif                 # if Save probability maps is enabled
q_arm_probability_3d.tif                 # if Save probability maps is enabled
p_arm_labels_by_nucleus_3d.tif           # if Save arm label masks is enabled
q_arm_labels_by_nucleus_3d.tif           # if Save arm label masks is enabled
pq_overlap_labels_by_nucleus_3d.tif      # if Save arm label masks is enabled
series7_chrX_arm_measurements_per_nucleus.csv
series7_chrX_arm_measurements_population_summary.csv
analysis_parameters.json
pq_arm_analyzer_configuration.json
arm_intensity_context.json
qc_nuc_maxproj.tif                       # if Save QC max projections is enabled
qc_p_maxproj.tif                         # if Save QC max projections is enabled
qc_q_maxproj.tif                         # if Save QC max projections is enabled
qc_nuclei_labels_maxproj.tif             # if Save QC max projections is enabled
plots/*.png
plots/plot_summary.txt
```

Batch analysis writes one result folder per checked scene, plus a batch configuration file in the base output folder.

---

# Public result table

The public per-nucleus CSV keeps final analysis/reporting columns such as:

- nucleus ID and method name;
- nucleus, P, Q, and overlap volumes;
- P and Q fractions of nuclear volume;
- P/Q overlap normalized to P and Q;
- P/Q contact and edge distance;
- P and Q centroids;
- selected radial-shell fractions;
- 3D shape metrics such as surface area, sphericity, compactness, equivalent sphere diameter, PCA axis lengths, elongation, and flatness;
- continuous 3D radial coordinates for P, Q, and P/Q overlap.

Internal tuning/debug columns are intentionally omitted from the public CSV. These include field-background medians/MADs, GMM means, BIC/LLR diagnostics, presence-gate diagnostics, detailed component diagnostics, selected bounding-box dimensions, selected raw distance-transform values, and the requested shell-1 fractions.

See the manual for the complete result-column dictionary and formulas.

---

# Plots

The plot set includes summary figures for arm volume fraction, P/Q overlap, contact frequency, minimum edge distance, P versus Q volume, mean arm volume versus overlap, centroid separation, radial centroid position, and shape/sphericity when the required columns are present.

Plots based only on removed internal diagnostic columns are not generated.

---

# Parameter help buttons

Each visible parameter row has a round **i** help button.

- Hovering over the button shows detailed help.
- Clicking the button opens a fixed help window.
- Re-clicking the same parameter help button raises the existing help window instead of opening duplicate windows.

The help text is stored in:

```text
src/napari_pq_arm_analyzer/help_text.py
```

---

# Hidden backend defaults

To simplify routine use, the following advanced/internal parameters are not shown in the GUI and are fixed to conservative defaults internally:

```text
Foreground bbox XY pad
Cellprob threshold
Flow threshold
3D stitch threshold
Covariance type
Random state
Max voxels/nucleus for fitting
```

They remain recorded in JSON output files for reproducibility.

---

# Parameter tuning philosophy

The most reliable way to tune the plugin is to treat segmentation as a sequence of decisions rather than one threshold.

First tune nuclei, because all arm measurements are made inside nucleus labels. Next tune the GMM class index and probability threshold so the approximate P/Q signal region is correct. Then tune the presence gate so nuclei without convincing signal are rejected. Tune component selection after that, because it decides which connected 3D objects survive. Use MRF/CRF refinement only after the intensity-based result is mostly correct and the remaining problem is boundary quality, small gaps, or speckled edges.

A typical tuning order is:

```text
1. Confirm scene and channel selection.
2. Tune Cellpose nuclei segmentation.
3. Tune GMM class index and probability threshold.
4. Tune presence gate thresholds.
5. Tune component selection mode and score threshold.
6. Add MRF/CRF refinement if boundaries need spatial cleanup.
7. Save configuration and run checked scenes.
```

---

# Result-column interpretation

The per-nucleus CSV is designed to contain final analysis/reporting columns, not every internal diagnostic variable. For each nucleus, the table reports volumes, fractions, overlap, contact, edge distance, centroids, radial shell fractions, 3D shape metrics, PCA-based shape axes, continuous 3D radial positions, and method metadata.

Important examples:

- `p_fraction_of_nucleus` and `q_fraction_of_nucleus` normalize arm volume by nuclear volume.
- `pq_overlap_fraction_of_p` and `pq_overlap_fraction_of_q` answer different questions when P and Q volumes differ.
- `pq_contact` is a binary contact call, while `pq_min_edge_distance_um` is a continuous distance.
- Shape columns such as sphericity and surface area help identify rough, fragmented, or over-smoothed masks.
- Radial columns summarize where the P/Q masks lie inside each nucleus on a 0-to-1 center-to-periphery scale.

---

# Visual validation checklist

Before running a large batch, validate one or a few representative scenes:

- Scroll through Z planes and confirm the correct scene and channels.
- Check that nuclei are neither split nor merged.
- Overlay P/Q masks on the raw channels.
- Inspect strong-signal, weak-signal, and no-signal nuclei.
- Confirm that saved masks and QC projections match the preview.
- Review the saved configuration JSON alongside the CSV output.

---

# Development and testing

Useful checks from the repository root:

```bash
python -m compileall src
python -m pip install -e .
napari
```

The plugin is written as a standard napari npe2 plugin with the manifest in:

```text
src/napari_pq_arm_analyzer/napari.yaml
```

---

# Troubleshooting installation

## The plugin does not appear in napari

Try:

```bash
conda activate pq-arm-analyzer
python -m pip uninstall -y napari-pq-arm-analyzer
cd napari-pq-arm-analyzer
python -m pip install -e .
napari
```

Also confirm that you launched napari from the same environment where the plugin was installed:

```bash
which python
which napari
python -m pip show napari-pq-arm-analyzer
```

On Windows, use:

```bat
where python
where napari
python -m pip show napari-pq-arm-analyzer
```

## napari opens but crashes with a Qt error

Install or reinstall a Qt backend:

```bash
python -m pip install --upgrade "napari[pyqt5]" pyqt5
```

Then restart the terminal and relaunch napari.

## LIF, ND2, or IMS files do not open

Install optional readers:

```bash
python -m pip install readlif nd2reader imaris-ims-file-reader
```

Then restart napari.

## Cellpose is slow

Use Cellpose 2D stack-batch mode, increase XY downsampling for preview, limit preview nuclei, and use GPU if available.

## GPU is not detected

Check PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

If this prints `False`, install a GPU-enabled PyTorch build compatible with your system, then reinstall or restart napari.

## First Cellpose run is slow

Cellpose may download or initialize model files the first time it runs. Later runs are usually faster after model files are cached.

---

# Uninstalling

To remove the plugin from the current environment:

```bash
conda activate pq-arm-analyzer
python -m pip uninstall -y napari-pq-arm-analyzer
```

To remove the entire environment:

```bash
conda deactivate
conda env remove -n pq-arm-analyzer
```

---

# Attribution and license

Author and repository owner:

```text
Adib Keikhosravi, Ph.D.
Staff Scientist,
Laboratory of Receptor Biology and Gene Expression, CCR, NCI
National Institutes of Health
Email: adib.keikhosravi@nih.gov
```

This repository is protected by the MIT License. See [`LICENSE`](LICENSE).
