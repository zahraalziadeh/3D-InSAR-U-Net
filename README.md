# 3D Crustal Displacement Retrieval near Denali Fault

## Overview
This repository contains the source code and outputs for a novel GCP-free methodology to retrieve 3D crustal displacement fields near the Denali Fault, Alaska, integrating Pleiades stereo imagery, InSAR techniques, and a U-Net deep learning model with geodetic optimization. The study, submitted to *Journal of Geophysical Research: Solid Earth (JGR)*, targets a 20 km × 20 km segment, achieving displacement ranges of ±20 mm with RMSEs of 1.17 mm (east-west), 1.46 mm (north-south), and 1.52 mm (vertical).

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
The full source code and input datasets are stored in a private, password-protected repository on Zenodo with the DOI: 10.5281/zenodo.15616826. Access is restricted until the article is published. Request access and password from the first author (alizadehzahra@email.kntu.ac.ir) during review. Post-acceptance, data will be made public.

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
