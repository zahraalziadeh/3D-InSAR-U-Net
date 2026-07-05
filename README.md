# Regional 3D Surface Deformation Inference from Single-Geometry InSAR

## Overview

This repository provides a physics-informed framework for probabilistic regional-scale inference of three-dimensional (3D) surface deformation fields from single-geometry InSAR observations.

The proposed workflow integrates:

* Sentinel-1 line-of-sight (LOS) InSAR measurements
* Very-high-resolution Pleiades stereo imagery (0.5 m)
* DEM differencing and COSI-Corr optical image correlation
* Persistent Scatterer InSAR (PS-InSAR) observations as geodetic constraints
* A physics-informed U-Net for deformation prior generation
* Bayesian geodetic fusion for enforcing physical consistency
* Monte Carlo Dropout and posterior covariance propagation for uncertainty quantification

The central Denali Fault, Alaska, serves as the case study for demonstrating the regional applicability of the proposed framework.

Within an approximately 130 km² stereo-constrained anchor region, east (E), north (N), and vertical (U) displacement components are derived through the integration of optical, radar, and geodetic observations. These locally constrained deformation fields provide supervisory information for a physics-informed deep-learning model that is subsequently transferred to a broader regional domain (~70 × 70 km), where only single-geometry Sentinel-1 observations and DEM-derived terrain information are available.

The framework enables spatially continuous regional 3D deformation inference in areas lacking multi-geometry SAR observations while providing spatially explicit uncertainty estimates through probabilistic modelling and Bayesian geodetic covariance propagation.

Independent GNSS validation demonstrates regionally coherent deformation estimates with component-dependent accuracies ranging from millimeter to centimeter levels.

The repository contains preprocessing scripts, model architectures, training workflows, regional inference modules, Bayesian geodetic fusion routines, uncertainty estimation procedures, and validation tools required to reproduce the experiments presented in the associated manuscript.

---

## Reproducibility Statement

This repository contains all scripts required for:

* Data preprocessing
* PS-InSAR integration
* Stereo-derived deformation prior generation
* Physics-informed U-Net training
* Regional-scale inference
* Bayesian geodetic fusion
* Monte Carlo uncertainty estimation
* Posterior covariance propagation
* GNSS validation

The implementation reproduces the experiments presented in the manuscript:

**Physics-Constrained Regional Inference of Three-Dimensional Ground Deformation from Single-Geometry InSAR Using Locally Constrained Stereo-Derived Priors**

---

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Optionally specify an output directory:

```bash
export OUTPUT_DIR=/path/to/your/outputs
```

---

## Required Inputs

The workflow requires the following input files:

* `clipped_dinsar.tif` – Sentinel-1 LOS displacement map
* `alaska_ps.csv` – PS-InSAR observations (longitude, latitude, deformation rate)
* `dem1.tif` – Digital elevation model
* `filtered_east-west-iqr_4326.tif` – East displacement prior
* `filtered_North_South-iqr_4326.tif` – North displacement prior
* `smoothed_vertical_deformation.tif` – Vertical displacement prior
* `Map_a.tif` – Background visualization layer

These datasets can be generated from raw Sentinel-1 and Pleiades observations following the procedures described in the accompanying manuscript.

---

## Code and Data Availability

The complete implementation of the proposed framework is publicly available through this GitHub repository:

https://github.com/zahraalziadeh/3D-InSAR-U-Net

A permanent archived version is also available through Zenodo:

https://doi.org/10.5281/zenodo.15616826

The repository includes preprocessing scripts, stereo-derived deformation prior generation, physics-informed U-Net training, regional inference, Bayesian geodetic fusion, uncertainty quantification, and GNSS validation required to reproduce the published experiments.

---

## Data Availability

The original remote-sensing datasets are not redistributed through this repository because of licensing restrictions and data volume.

The study uses the following primary datasets:

* Pleiades stereo imagery (25 September 2021 and 28 July 2022)
* Sentinel-1A IW-mode SLC observations (September 2021 to August 2022)
* DEM-derived terrain products (slope, aspect, TRI)
* Geological fault datasets
* GNSS observations for independent validation

Processed datasets and representative outputs are provided where redistribution is permitted.

The original Pleiades imagery was obtained through the ESA Earth Observation Gateway and remains subject to third-party licensing restrictions. Sentinel-1 SAR data are openly available through the Copernicus Data Space Ecosystem, while GNSS observations are publicly available through the UNAVCO data repository.

Researchers can reproduce the proposed workflow using the publicly available code together with the original data sources referenced above.

---

## Contributors

### Zahra Alizadeh Zakaria

PhD Candidate, Department of Geodesy and Geomatics Engineering  
K. N. Toosi University of Technology, Tehran, Iran

Email: alizadehzahra@email.kntu.ac.ir

ORCID: https://orcid.org/0009-0002-1344-7007

---

### Farshid Farnood Ahmadi

Department of Geomatics Engineering  
University of Tabriz, Tabriz, Iran

Email: farnood@tabrizu.ac.ir

ORCID: https://orcid.org/0009-0001-9664-9546

---

### Hamid Ebadi

Department of Geodesy and Geomatics Engineering  
K. N. Toosi University of Technology, Tehran, Iran

Email: ebadi@email.kntu.ac.ir

ORCID: https://orcid.org/0000-0002-7017-1927

---

## Citation

If you use this repository in your research, please cite the accompanying manuscript together with the archived Zenodo release.

---

## License

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0).**

See the **LICENSE** file for details.
