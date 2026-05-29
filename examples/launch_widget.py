# Author: Adib Keikhosravi, Ph.D.
# Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI
# National Institutes of Health
# Email: adib.keikhosravi@nih.gov
# License: MIT License

"""Launch napari with the P/Q Arm Analyzer widget for local testing."""

import napari
from napari_pq_arm_analyzer._widget import PQArmAnalyzerWidget

viewer = napari.Viewer()
viewer.window.add_dock_widget(PQArmAnalyzerWidget(viewer), name="P/Q Arm Analyzer", area="right")
napari.run()
