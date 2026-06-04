# Changelog

## v0.3.6

- Updated default routine-analysis parameters: Max GMM components = 4, P/Q sorted class index thresholds = 3, minimum nucleus volume = 5 um^3, component selection = all_passing_score, and preview nucleus limit = 0.
- Pinned the package dependency to Cellpose >=2,<4 so editable installs do not accidentally use the Cellpose v4 API that requires different 3D-axis arguments.
- Added a defensive Cellpose call path that supplies z_axis=0 for ZYX 3D stacks when a newer Cellpose API is present.
- Updated README and manual references for the new defaults and installation behavior.

## v0.3.5

- Expanded the user manual into a complete conceptual, mathematical, and practical reference.
- Added detailed explanations for all remaining public CSV result columns.
- Added a complete MRF/CRF and component-selection appendix.
- Expanded the GitHub README with installation, workflow, parameter categories, outputs, result-table description, and troubleshooting.
- Generalized help text so it does not mention one specific image size.
- Kept hidden backend parameters out of the GUI and manual.

## v0.3.4

- Added author and MIT license text to repository files.
- Removed duplicate help windows per parameter.
- Removed selected internal diagnostic columns from the public CSV output.
- Added configuration save/load and scene checkbox batch analysis.
