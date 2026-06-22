# Regional 3D Surface Deformation Inference from Single-Geometry InSAR

## Overview

This repository provides a physics-informed framework for probabilistic regional-scale inference of three-dimensional (3D) surface deformation fields from single-geometry InSAR observations.

The workflow integrates:

* Sentinel-1 line-of-sight (LOS) InSAR measurements
* Very-high-resolution (50 cm) Pleiades stereo imagery
* DEM differencing and COSI-Corr optical image correlation
* Persistent Scatterer InSAR (PS-InSAR) observations as geodetic constraints
* A physics-informed U-Net for deformation prior generation
* Bayesian geodetic fusion for physical consistency
* Monte Carlo Dropout and posterior covariance propagation for uncertainty quantification

The central Denali Fault, Alaska, serves as a case study for demonstrating the regional applicability of the proposed framework.

Within an approximately 130 km² stereo-constrained anchor region, relative east (E), north (N), and vertical (U) displacement components are derived through the integration of optical, radar, and geodetic observations. These locally constrained displacement fields provide supervisory information for a physics-informed deep-learning model that is subsequently applied to a broader regional domain (~70 × 70 km) where only single-geometry Sentinel-1 observations are available.

The framework enables spatially continuous regional 3D deformation inference in areas lacking multi-geometry observations while providing spatially explicit uncertainty estimates through probabilistic modelling and geodetic covariance propagation.

Independent validation against GNSS observations indicates regionally coherent deformation estimates with component-dependent accuracy ranging from millimeter to centimeter levels.

The repository contains preprocessing scripts, model architectures, training workflows, regional inference modules, uncertainty estimation routines, geodetic fusion procedures, and validation tools required to reproduce the results reported in the associated manuscript.

---

## Reproducibility Statement

All scripts required for:

* Data preprocessing
* PS-InSAR integration
* Stereo-derived deformation prior generation
* U-Net training
* Regional-scale inference
* Bayesian geodetic fusion
* Uncertainty quantification
* GNSS validation

are included in this repository.

The workflow reproduces the experiments presented in the manuscript:

**Physics-Constrained Regional Inference of Three-Dimensional Ground Deformation from Single-Geometry InSAR Using Locally Constrained Stereo-Derived Priors**

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Set an output directory:

```bash
export OUTPUT_DIR=/path/to/your/outputs
```

---

## Required Inputs

The workflow requires the following input files:

* `clipped_dinsar.tif` – LOS displacement measurements
* `alaska_ps.csv` – PS-InSAR observations (longitude, latitude, deformation rate)
* `dem1.tif` – Digital elevation model
* `filtered_east-west-iqr_4326.tif` – East-west displacement prior
* `filtered_North_South-iqr_4326.tif` – North-south displacement prior
* `smoothed_vertical_deformation.tif` – Vertical displacement prior
* `Map_a.tif` – Background visualization layer

These products can be generated from raw Sentinel-1 and Pleiades observations following the procedures described in the manuscript.

---

## Code and Data Availability

The source code and supporting datasets have been archived on Zenodo:

DOI: 10.5281/zenodo.15616826

During peer review, repository access is restricted. Editors and reviewers may obtain access upon request.

Upon publication, the repository will be made publicly accessible.

---

## Data Availability

Due to licensing restrictions and data volume, the original remote-sensing datasets are not distributed directly through this repository.

Data sources include:

* Pleiades stereo imagery (25 September 2021 and 28 July 2022)
* Sentinel-1A IW-mode SLC observations (September 2021 to August 2022)
* DEM-derived terrain products (slope, aspect, TRI)
* Geological fault datasets

Sample outputs and derived products are provided where permitted.

For review purposes, access requests may be directed to the corresponding author.

---

## Contributors

* Zahra Alizadeh Zakaria
  K. N. Toosi University of Technology
  [alizadehzahra@email.kntu.ac.ir](mailto:alizadehzahra@email.kntu.ac.ir)

* Farshid Farnood Ahmadi
  University of Tabriz
  [farnood@tabrizu.ac.ir](mailto:farnood@tabrizu.ac.ir)

* Hamid Ebadi
  K. N. Toosi University of Technology
  [ebadi@email.kntu.ac.ir](mailto:ebadi@email.kntu.ac.ir)

---

## License

This project is distributed under the GNU General Public License v3.0 (GPL-3.0).

See the LICENSE file for details.
