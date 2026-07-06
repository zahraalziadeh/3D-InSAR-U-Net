[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15616826.svg)](https://doi.org/10.5281/zenodo.15616826)

![Python](https://img.shields.io/badge/Python-3.8-blue)

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)

# Regional 3D Surface Deformation Inference from Single-Geometry InSAR

## Overview

This repository accompanies the manuscript

**Physics-Constrained Regional Inference of Three-Dimensional Ground Deformation from Single-Geometry InSAR Using Locally Constrained Stereo-Derived Priors**

The proposed framework estimates regional three-dimensional ground deformation from single-geometry Sentinel-1 InSAR observations by integrating

- Sentinel-1 LOS displacement
- Pleiades stereo imagery
- DEM differencing
- COSI-Corr
- Persistent Scatterer InSAR
- Physics-informed U-Net
- Bayesian geodetic fusion
- Monte Carlo Dropout
- Posterior uncertainty propagation

The workflow was developed and evaluated over the central Denali Fault, Alaska.

---

# Repository Contents

```
README.md
LICENSE
requirements.txt

3d_los.py
3d_los.ipynb
validation_gnss.ipynb

example/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/zahraalziadeh/3D-InSAR-U-Net.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# Required Input Data

The workflow requires

- clipped_dinsar.tif
- alaska_ps.csv
- dem1.tif
- filtered_east-west-iqr_4326.tif
- filtered_North_South-iqr_4326.tif
- smoothed_vertical_deformation.tif
- Map_a.tif

These datasets are available through the archived Zenodo release.

---

# Quick Start

Download the processed datasets from Zenodo.

Place the datasets in the project directory.

Run

```bash
python 3d_los.py
```

For validation

```bash
jupyter notebook validation_gnss.ipynb
```

---

# Reproducibility

The complete archived version of the software, processed datasets, trained models, and supplementary material is permanently available through Zenodo:

https://doi.org/10.5281/zenodo.15616826

The GitHub repository serves as the public source-code repository.

---

# Code Availability

All Python source code developed for this study is openly available in this GitHub repository.

The archived Zenodo release contains the exact version used to generate the published results.

---

# Data Availability

The original Pleiades imagery cannot be redistributed because of third-party licensing restrictions.

Sentinel-1 SAR data are available through the Copernicus Data Space Ecosystem.

GNSS observations are available through the UNAVCO archive.

Processed datasets required to reproduce the published analyses are available through Zenodo.

---

# Citation

If you use this repository, please cite both

- the accompanying journal article
- the Zenodo archive

---

# License

GNU General Public License v3.0
