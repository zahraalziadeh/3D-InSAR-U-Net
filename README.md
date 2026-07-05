[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15616826.svg)](https://doi.org/10.5281/zenodo.15616826)

![Python](https://img.shields.io/badge/Python-3.10+-blue)

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
# Regional 3D Surface Deformation Inference from Single-Geometry InSAR

## Overview

This repository accompanies the research presented in the manuscript:

**Physics-Constrained Regional Inference of Three-Dimensional Ground Deformation from Single-Geometry InSAR Using Locally Constrained Stereo-Derived Priors**

The project presents a physics-informed computational framework for probabilistic regional-scale inference of three-dimensional (3D) ground deformation from single-geometry Sentinel-1 InSAR observations.

The proposed workflow integrates

- Sentinel-1 line-of-sight (LOS) measurements
- Very-high-resolution (0.5 m) Pleiades stereo imagery
- DEM differencing
- COSI-Corr optical image correlation
- Persistent Scatterer InSAR (PS-InSAR)
- Physics-informed U-Net
- Bayesian geodetic fusion
- Monte Carlo Dropout
- Posterior covariance propagation

The framework was developed and evaluated over the central Denali Fault, Alaska.

Within an approximately **130 km²** stereo-constrained anchor region, east (E), north (N), and vertical (U) displacement components are derived by integrating optical, radar, and geodetic observations. These locally constrained deformation fields provide supervisory information for a physics-informed U-Net that is subsequently transferred to a much larger regional domain (**~70 × 70 km**) where only single-geometry Sentinel-1 observations and DEM-derived terrain information are available.

The proposed framework enables spatially continuous regional 3D deformation mapping while explicitly quantifying predictive uncertainty through probabilistic inference and Bayesian geodetic modelling.

Independent GNSS validation demonstrates regionally coherent deformation estimates with component-dependent accuracies ranging from millimeter to centimeter levels.

---

# Repository Contents

The repository documents the complete computational workflow, including

- data preprocessing
- DEM preparation
- PS-InSAR integration
- stereo-derived deformation prior generation
- Physics-informed U-Net architecture
- model training
- regional inference
- Bayesian geodetic fusion
- uncertainty quantification
- GNSS validation

---

# Reproducibility

A permanently archived and citable version of the complete research software and processed datasets is available on **Zenodo**:

**https://doi.org/10.5281/zenodo.15616826**

The Zenodo archive represents the exact version used to generate the experiments and results presented in the accompanying manuscript.

This GitHub repository serves as the public project repository, providing documentation, workflow description, software updates, and project maintenance.

---

# Installation

Install the required Python packages

```bash
pip install -r requirements.txt
```

Optionally define an output directory

```bash
export OUTPUT_DIR=/path/to/output
```

---

# Required Inputs

The workflow requires

- `clipped_dinsar.tif`
  Sentinel-1 LOS displacement

- `alaska_ps.csv`
  Persistent Scatterer InSAR observations

- `dem1.tif`
  Digital elevation model

- `filtered_east-west-iqr_4326.tif`
  East displacement prior

- `filtered_North_South-iqr_4326.tif`
  North displacement prior

- `smoothed_vertical_deformation.tif`
  Vertical displacement prior

- `Map_a.tif`
  Background visualization layer

These datasets can be generated following the processing methodology described in the accompanying manuscript.

---

# Code and Data Availability

The complete source code, processed datasets, trained models, and supplementary materials supporting this work are permanently archived on **Zenodo**:

**https://doi.org/10.5281/zenodo.15616826**

The archived Zenodo release constitutes the citable version associated with the published research.

The GitHub repository is maintained as the public development and documentation repository for the project.

---

# Data Availability

The original remote sensing datasets are **not redistributed** through this repository because of licensing restrictions.

Primary data sources include

- Pleiades stereo imagery
- Sentinel-1A IW SLC observations
- DEM-derived terrain products
- Geological fault datasets
- GNSS observations

The original Pleiades imagery was obtained through the ESA Earth Observation Gateway and remains subject to third-party licensing restrictions.

Sentinel-1 SAR data are freely available through the Copernicus Data Space Ecosystem.

GNSS observations are publicly available through the UNAVCO data archive.

Processed datasets used in this study are available through the archived Zenodo release.

---

# Contributors

### Zahra Alizadeh Zakaria

PhD Candidate

Department of Geodesy and Geomatics Engineering

K. N. Toosi University of Technology

Email:
alizadehzahra@email.kntu.ac.ir

ORCID:
https://orcid.org/0009-0002-1344-7007

---

### Farshid Farnood Ahmadi

Department of Geomatics Engineering

University of Tabriz

Email:
farnood@tabrizu.ac.ir

ORCID:
https://orcid.org/0009-0001-9664-9546

---

### Hamid Ebadi

Department of Geodesy and Geomatics Engineering

K. N. Toosi University of Technology

Email:
ebadi@email.kntu.ac.ir

ORCID:
https://orcid.org/0000-0002-7017-1927

---

# Citation

If you use this repository in your research, please cite both

- the accompanying journal article

and

- the archived Zenodo release

to ensure proper attribution and reproducibility.

---

# License

This project is released under the **GNU General Public License v3.0 (GPL-3.0).**

See the **LICENSE** file for additional information.
