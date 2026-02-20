# 3D Crustal Displacement Retrieval near Denali Fault

## Overview
This repository implements a physics-informed, GCP-free framework for reconstructing relative three-dimensional (3D) ground deformation from single-geometry InSAR observations.

The workflow is demonstrated over a ~70 × 70 km segment of the Denali Fault, Alaska, and integrates:

Sentinel-1 LOS InSAR measurements

Very-high-resolution (50 cm) Pleiades stereo imagery

DEM differencing and COSI-Corr optical correlation

Persistent Scatterer InSAR products as virtual geodetic constraints

A physics-informed U-Net model for patch-based 3D deformation prior generation

Least-squares geodetic adjustment for physical consistency

Monte Carlo dropout and posterior covariance analysis for uncertainty quantification

Within a ~20 × 20 km stereo-imaged anchor region, relative east (E), north (N), and vertical (U) displacement components are inferred through cross-sensor fusion. The framework is subsequently scaled to the broader domain using a deep-learning–derived 3D deformation prior constrained by InSAR observations and terrain features.

Across the full study region, reconstructed displacement gradients exhibit ranges of approximately:

−30 to +30 mm (E–W)

−40 to +40 mm (N–S)

−40 to +40 mm (Vertical)

Independent evaluation against peripheral GNSS observations in the IGS14/NNR reference frame yields average RMSE values of:

~7.1 mm (East)

~25.0 mm (North)

~9.8 mm (Vertical)

The larger uncertainty in the north–south component reflects the geometric sensitivity limitations of near-polar Sentinel-1 orbits.

The repository includes preprocessing scripts, model architecture definitions, training workflows, uncertainty estimation modules, and validation routines necessary to reproduce the reported results.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the output directory:
   ```bash
   export OUTPUT_DIR=/path/to/your/outputs
   ```

## Required Inputs
To execute the full code, the following input files are required (available to reviewers upon request):
- `clipped_dinsar.tif`: Line-of-sight displacement data.
- `alaska_ps.csv`: PSInSAR displacement data (columns: longitude, latitude, deff).
- `dem1.tif`: Digital elevation model.
- `filtered_east-west-iqr_4326.tif`: East-west displacement from Pleiades.
- `filtered_North_South-iqr_4326.tif`: North-south displacement from Pleiades.
- `smoothed_vertical_deformation.tif`: Vertical displacement from Pleiades.
- `Map_a.tif`: Background map for visualization.
These files can be generated from raw Pleiades stereo imagery, Sentinel-1A SLC images, or requested from the corresponding author (alizadehzahra@email.kntu.ac.ir).

### Execution and Output
The full source code and input datasets are stored in a private, password-protected repository on Zenodo with the [DOI:10.5281/zenodo.15616826](https://doi.org/10.5281/zenodo.15616826). Access is restricted until the article is published. Request access and password from the first author (alizadehzahra@email.kntu.ac.ir) during review. Post-acceptance, data will be made public.

## Contributors
- Zahra Alizadeh Zakaria (K.N. Toosi University of Technology, alizadehzahra@email.kntu.ac.ir)
- Farshid Farnood Ahmadi (University of Tabriz, farnood@tabrizu.ac.ir)
- Hamid Ebadi (K.N. Toosi University of Technology, ebadi@email.kntu.ac.ir)

## License
GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Data Availability
The input datasets (e.g., Pleiades stereo imagery from 25 September 2021 and 28 July 2022, Sentinel-1A SLC images from September 2021–August 2022) are large-volume and not included due to size constraints. Sample outputs are provided, but full datasets are available to reviewers upon request with specifications:
- **Pleiades Stereo Imagery:** 50 cm panchromatic resolution, stereo pairs aligned to UTM Zone 5N.
- **Sentinel-1A SLC Images:** IW mode, descending orbit, incidence angle 35° ± 0.5°, azimuth angle 190° ± 0.5°, 17 scenes (Sep 2021–Aug 2022).
- **Geological Constraints:** Slope, Terrain Ruggedness Index (TRI) from Pleiades DEMs, fault trace from Stanford dataset.
Request datasets from the corresponding author (alizadehzahra@eail.kntu.ac.ir) during review. Post-acceptance, key data will be public.
## License
GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
