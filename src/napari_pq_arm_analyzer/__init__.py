# Author: Adib Keikhosravi, Ph.D.
# Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI
# National Institutes of Health
# Email: adib.keikhosravi@nih.gov
# License: MIT License

"""napari plugin for P/Q arm segmentation and measurement."""

__version__ = "0.3.4"

try:
    from ._widget import PQArmAnalyzerWidget, make_widget
except Exception:
    # Keep package metadata importable in environments where napari/Qt is not installed yet.
    PQArmAnalyzerWidget = None  # type: ignore
    make_widget = None  # type: ignore
