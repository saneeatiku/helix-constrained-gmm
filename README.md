# Helix-Constrained Gaussian Mixture Model

A 48-component Gaussian Mixture Model (GMM) for visitor positions `(x, y, z)` on a
double-helix observation tower, trained with a Generalized EM (ECM-type) algorithm.
Each component mean is **constrained to lie on a helix** and each covariance is
**aligned with the stair's local frame**. Written in Python (NumPy, SciPy).

Developed with Team Ascend for the *Mathematical Foundations of AI & ML*
continuous assessment (MSc AI & ML, University of Birmingham, 2026).

> **Data note:** the scenario provides only a scatter plot, not a coordinate file.
> All experiments use **synthetic data** generated from the assumed geometry.
> The results illustrate the algorithm and are **not** results on real visitor data.

## Idea

- Two intertwined helical staircases (height 80 ft, centreline radius 12 ft,
  stair width 4 ft, 2 turns each, 180° phase offset).
- Component `k` has one free angle `θ_k`, so its mean is `h_s(θ_k)`, always on the helix.
- Covariance is diagonal in the local frame (across, along, through the stair):
  `Σ_k = Q_k diag(σ²_r, σ²_t, σ²_n) Q_kᵀ`.
- **239 free parameters** instead of 479 for an unconstrained full-covariance GMM.

## Training (Generalized EM)

1. **E-step:** responsibilities via log-sum-exp.
2. **M-step (i):** mixture weights, closed form.
3. **M-step (ii):** angles, windowed 1-D search (±30°), so a component stays on its stair.
4. **M-step (iii):** variances, closed form with a floor `ε = 0.01 ft²`.

The log-likelihood is non-decreasing at every iteration. The code checks this numerically.
Initialisation places 24 components per staircase at equal spacing with aligned covariances.

## Run

```bash
pip install -r requirements.txt
python helix_gmm.py
```

This runs geometry checks, numerical self-tests (orthonormal frame, analytic gradient vs
finite differences) and three synthetic experiments.

Fit your own data (`X` is an `(N, 3)` array in feet):

```python
import numpy as np
from helix_gmm import fit

X = np.loadtxt("positions.csv", delimiter=",")
params, loglik_history, _ = fit(X, max_iter=300, tol=1e-7, verbose=True)
```

`params` is a dict with the fitted weights, angles and variances.

## Results on synthetic data (N = 1000, seed 0)

| Check | Result |
|---|---|
| Geometric start | converged in 56 iterations, log-likelihood -7017.16 |
| 20 random-angle starts | 116 iterations on average; none within 1.0 of the geometric start's log-likelihood |
| Monotone log-likelihood | holds every iteration |
| Means on helix | all angles stay in [0, 4π] |

## Limitations

- Geometry (radius, phase, turning direction) is fixed and assumed, not learned.
- Gaussians put some mass outside the physical stair edges (about 8% radially).
- Not validated on real visitor data.

## Authors

Team Ascend, University of Birmingham.
