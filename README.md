# jpsi-upsilon-invariant-mass

Reproducible Python pipeline to reconstruct the dimuon invariant-mass spectrum
and reproduce the **J/ψ** and **ϒ** resonance peaks using the  
**CERN Subatomic Particles Dataset** (CMS DoubleMu 2011) available on Kaggle.

This repository accompanies an academic article focused on **reproducibility**
and provides an open, transparent analysis workflow from raw variables to final figures.

---

## Overview

This project:
- Loads the dataset (Excel/CSV).
- Applies physics-motivated selections:
  - Opposite-charge dimuon pairs: \(Q_1 + Q_2 = 0\)
  - Physical cut: \(m_{\mathrm{inv}}^2 > 0\)
- Computes the dimuon invariant mass:
  \[
  m_{\mathrm{inv}}^2 = (E_1 + E_2)^2 - |\vec{p}_1 + \vec{p}_2|^2
  \]
- Produces:
  - Global histogram (0–12 GeV)
  - Zoomed histograms around J/ψ and ϒ
  - Optional local **signal + background** fits
    (Gaussian + linear background)
  - Optional normalized distributions:
    \[
    z = \frac{m - \mu}{\sigma}
    \]

---

## Dataset

- **Name:** CERN Subatomic Particles Dataset  
- **Source:** Kaggle (derived from CMS DoubleMu 2011)  
- This repository does **not** include the dataset file.
  Download it from Kaggle and place it locally.

---

## Requirements

Create and activate a virtual environment (recommended), then:

```bash
pip install -r requirements.txt
