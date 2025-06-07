# 3D Crustal Displacement Retrieval near Denali Fault

## Overview
This repository contains the source code for a novel GCP-free methodology to retrieve 3D crustal displacement fields near the Denali Fault, Alaska, integrating Pleiades stereo imagery, InSAR techniques, and a U-Net deep learning model with geodetic optimization. The study, submitted to *Journal of Geophysical Research: Solid Earth (JGR)*, targets a 20 km × 20 km segment, achieving displacement ranges of ±20 mm with RMSEs of 1.17 mm (east-west), 1.46 mm (north-south), and 1.52 mm (vertical).

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the output directory:
   ```bash
   export OUTPUT_DIR=/path/to/your/outputs
   ```
3. Run the script:
   ```bash
   python 3d_los.py
   ```
   - Input files must be provided or requested.
   - Outputs are saved in `outputs/section_1`.

## Required Inputs
To execute the code, the following input files are required in the working directory:
- `clipped_dinsar.tif`: Line-of-sight displacement data.
- `alaska_ps.csv`: PSInSAR displacement data (columns: longitude, latitude, deff).
- `dem1.tif`: Digital elevation model.
- `filtered_east-west-iqr_4326.tif`: East-west displacement from Pleiades.
- `filtered_North_South-iqr_4326.tif`: North-south displacement from Pleiades.
- `smoothed_vertical_deformation.tif`: Vertical displacement from Pleiades.
- `Map_a.tif`: Background map for visualization.
These files can be generated from raw Pleiades stereo imagery, Sentinel-1A SLC images, or requested from the first author (alizadehzahra@email.kntu.ac.ir).

## Next Steps
### Stage 2: Testing and Optimization
1. **Prepare Outputs:**
   - Ensure `outputs/section_1`, `section_3`, and `section_4` directories exist:
     ```bash
     mkdir -p outputs/section_{1,3,4}
     ```
   - Run the script to generate intermediate files (e.g., `.npy` files) in `outputs/section_1`:
     ```bash
     python3 3d_los.py
     ```

## Data Availability
The input datasets (e.g., Pleiades stereo imagery from 25 September 2021 and 28 July 2022, Sentinel-1A SLC images from September 2021–August 2022) are large-volume and not included due to size constraints. Users should provide these with specifications:
- **Pleiades Stereo Imagery:** 50 cm panchromatic resolution, stereo pairs aligned to UTM Zone 5N.
- **Sentinel-1A SLC Images:** IW mode, descending orbit, incidence angle 35° ± 0.5°, azimuth angle 190° ± 0.5°, 17 scenes (Sep 2021–Aug 2022).

## Contributors
- Zahra Alizadeh Zakaria (K.N. Toosi University of Technology, alizadehzahra@eail.kntu.ac.ir)
- Farshid Farnood Ahmadi (University of Tabriz, farnood@tabrizu.ac.ir)
- Hamid Ebadi (K.N. Toosi University of Technology)

## License
GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
```
