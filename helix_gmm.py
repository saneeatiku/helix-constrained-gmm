#!/usr/bin/env python3
"""
helix_gmm.py -- Helix-constrained Gaussian Mixture Model trained by a
Generalized EM (ECM-type) algorithm.

Supporting code for the report
"A Helix-Constrained Gaussian Mixture Model for Visitor Positions on a
Double-Helix Observation Tower" (MFAIML CA, Variant D).

IMPORTANT
---------
* The scenario supplies NO numerical data (Figure 2 is only a plot).
  Everything in `run_experiments()` uses SYNTHETIC data generated from our
  own geometric assumptions.  Results are illustrations of the algorithm,
  NOT results for the City Park Authority's data set.
* To use real data, load an (N, 3) array of (x, y, z) in feet and call
  `fit(X)`.

Units: feet.  Dependencies: numpy, scipy.
Run:   python helix_gmm.py            (self-tests + synthetic experiments)
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

# ----------------------------------------------------------------------
# 1. Geometry (R, H, revolutions, width: from the scenario;
#    phases, handedness, centreline interpretation: our assumptions)
# ----------------------------------------------------------------------
R = 12.0                      # centreline radius (ASSUMED centreline reading)
H = 80.0                      # tower height            (scenario)
REVS = 2                      # revolutions per stair   (scenario)
W = 4.0                       # stair width             (scenario)
C = H / (2 * np.pi * REVS)    # rise per radian  = 80/(4*pi) = 6.3662 ft/rad
L = np.hypot(R, C)            # |dh/dtheta|      = 13.5841 ft
KAPPA = R / L**2              # helix curvature  = 0.06503 1/ft
THETA_MAX = 2 * np.pi * REVS  # 4*pi
K_PER = 24                    # components per staircase
K = 2 * K_PER                 # 48
PHASE = np.concatenate([np.zeros(K_PER), np.full(K_PER, np.pi)])  # A: 0, B: pi
EPS = 0.01                    # variance floor (ft^2)
DELTA = np.pi / 6             # trust-region half-width for theta update


def helix(theta, phase):
    """Centreline point h_s(theta); counter-clockwise ascending."""
    v = np.asarray(theta) + phase
    return np.stack([R * np.cos(v), R * np.sin(v), C * np.asarray(theta)
                     * np.ones_like(v)], axis=-1)


def frame(theta, phase):
    """Orthonormal local frame (e_r, tau, n), each of shape (..., 3)."""
    v = np.asarray(theta) + phase
    z0, o = np.zeros_like(v), np.ones_like(v)
    er = np.stack([np.cos(v), np.sin(v), z0], axis=-1)
    tau = np.stack([-R * np.sin(v), R * np.cos(v), C * o], axis=-1) / L
    n = np.stack([C * np.sin(v), -C * np.cos(v), R * o], axis=-1) / L
    return er, tau, n


# ----------------------------------------------------------------------
# 2. Parameters and initialisation
# ----------------------------------------------------------------------
def initial_params(sigma_t2=None, sigma_r2=W**2 / 12, sigma_n2=0.25):
    """Geometric initialisation (Section 7 of the report)."""
    ell = L * THETA_MAX / K_PER                    # arc length per component
    if sigma_t2 is None:
        sigma_t2 = ell**2 / 4                      # 12.65 ft^2
    j = np.tile(np.arange(1, K_PER + 1), 2)
    theta = (2 * j - 1) * np.pi / 12               # 15, 45, ..., 705 deg
    var = np.tile([sigma_r2, sigma_t2, sigma_n2], (K, 1))   # (r, t, n)
    return {"pi": np.full(K, 1.0 / K), "theta": theta, "var": var}


# ----------------------------------------------------------------------
# 3. E-step
# ----------------------------------------------------------------------
def local_coords(X, theta):
    """a[n,k,:] = (e_r, tau, n)^T (x_n - h_k(theta_k))   -> (N, K, 3)."""
    mu = helix(theta, PHASE)
    er, tau, n = frame(theta, PHASE)
    d = X[:, None, :] - mu[None, :, :]
    return np.stack([np.einsum("nki,ki->nk", d, u) for u in (er, tau, n)], -1)


def log_joint(X, p):
    """ln pi_k + ln N(x_n | mu_k, Sigma_k) -- no matrix inversion needed."""
    a = local_coords(X, p["theta"])
    logN = (-1.5 * np.log(2 * np.pi)
            - 0.5 * np.log(p["var"]).sum(-1)[None, :]
            - 0.5 * (a**2 / p["var"][None]).sum(-1))
    return np.log(np.maximum(p["pi"], 1e-300))[None, :] + logN


def e_step(X, p):
    lj = log_joint(X, p)
    lse = logsumexp(lj, axis=1, keepdims=True)
    return np.exp(lj - lse), float(lse.sum())     # gamma, log-likelihood


# ----------------------------------------------------------------------
# 4. M-step ingredients
# ----------------------------------------------------------------------
def suff_stats(X, gamma):
    """N_k, weighted mean xbar_k, weighted scatter C_k about xbar_k."""
    Nk = gamma.sum(0)
    safe = np.maximum(Nk, 1e-12)
    xbar = gamma.T @ X / safe[:, None]
    M2 = np.einsum("nk,ni,nj->kij", gamma, X, X) / safe[:, None, None]
    Ck = M2 - np.einsum("ki,kj->kij", xbar, xbar)
    return Nk, xbar, Ck


def Qk_theta(theta, k, Nk, xbar, Ck, var):
    """theta-dependent part of Q_k (vectorised over an array of theta)."""
    th = np.atleast_1d(theta)
    h = helix(th, PHASE[k])
    us = frame(th, PHASE[k])
    d = xbar[k] - h
    val = 0.0
    for j, u in enumerate(us):
        quad = np.einsum("mi,ij,mj->m", u, Ck[k], u) + np.einsum("mi,mi->m", u, d) ** 2
        val = val + quad / var[k, j]
    return -0.5 * Nk[k] * val


def grad_Qk_theta(theta, k, Nk, xbar, Ck, var):
    """Analytic dQ_k/dtheta (Eq. for the gradient in the report)."""
    th = np.atleast_1d(theta)
    h = helix(th, PHASE[k])
    er, tau, n = frame(th, PHASE[k])
    d = xbar[k] - h
    a_t = np.einsum("mi,mi->m", tau, d)
    Mrt = np.einsum("mi,ij,mj->m", er, Ck[k], tau) + np.einsum("mi,mi->m", er, d) * a_t
    Mrn = np.einsum("mi,ij,mj->m", er, Ck[k], n) + np.einsum("mi,mi->m", er, d) * np.einsum("mi,mi->m", n, d)
    vr, vt, vn = var[k]
    return Nk[k] * (L * a_t / vt
                    - (R / L) * Mrt * (1 / vr - 1 / vt)
                    + (C / L) * Mrn * (1 / vr - 1 / vn))


def update_theta(k, Nk, xbar, Ck, var, theta_old, n_grid=241):
    """Windowed 1-D maximisation of Q_k; theta_old is always a candidate,
    so Q_k(new) >= Q_k(old)  (generalized-EM condition)."""
    lo, hi = max(0.0, theta_old - DELTA), min(THETA_MAX, theta_old + DELTA)
    grid = np.append(np.linspace(lo, hi, n_grid), theta_old)
    q = Qk_theta(grid, k, Nk, xbar, Ck, var)
    i = int(np.argmax(q))
    best, qbest = grid[i], q[i]
    g_lo, g_hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    if g_hi > g_lo:
        res = minimize_scalar(lambda t: -Qk_theta(t, k, Nk, xbar, Ck, var)[0],
                              bounds=(g_lo, g_hi), method="bounded",
                              options={"xatol": 1e-10})
        if -res.fun > qbest:
            best, qbest = float(res.x), -res.fun
    return best


def update_var(k, Nk, xbar, Ck, theta_new):
    """Closed-form variances at the NEW theta, floored at EPS."""
    h = helix(theta_new, PHASE[k])
    d = xbar[k] - h
    S = Ck[k] + np.outer(d, d)                    # (1/N_k) sum gamma (x-h)(x-h)^T
    us = frame(theta_new, PHASE[k])
    return np.maximum([u @ S @ u for u in us], EPS)


def Q_full(X, p, gamma):
    return float((gamma * log_joint(X, p)).sum())


def m_step(X, gamma, p, check=False):
    """ECM M-step: (i) weights, (ii) theta (windowed), (iii) variances."""
    N = X.shape[0]
    Nk, xbar, Ck = suff_stats(X, gamma)
    q_trace = [Q_full(X, p, gamma)] if check else None
    new = {"pi": Nk / N, "theta": p["theta"].copy(), "var": p["var"].copy()}
    if check:
        q_trace.append(Q_full(X, new, gamma))
    for k in range(K):                            # block (ii)
        if Nk[k] > 1e-8:
            new["theta"][k] = update_theta(k, Nk, xbar, Ck, p["var"], p["theta"][k])
    if check:
        q_trace.append(Q_full(X, new, gamma))
    for k in range(K):                            # block (iii)
        if Nk[k] > 1e-8:
            new["var"][k] = update_var(k, Nk, xbar, Ck, new["theta"][k])
    if check:
        q_trace.append(Q_full(X, new, gamma))
    return new, q_trace


# ----------------------------------------------------------------------
# 5. Training loop
# ----------------------------------------------------------------------
def fit(X, params=None, max_iter=300, tol=1e-7, check=False, verbose=False):
    p = params if params is not None else initial_params()
    p = {k_: v.copy() for k_, v in p.items()}
    hist, qtraces = [], []
    for it in range(max_iter):
        gamma, ll = e_step(X, p)
        hist.append(ll)
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * (abs(hist[-2]) + 1):
            break
        p, qt = m_step(X, gamma, p, check=check)
        if check:
            qtraces.append(qt)
        if verbose:
            print(f"iter {it:3d}  logL = {ll:.4f}")
    gamma, ll = e_step(X, p)
    hist.append(ll)
    return p, np.array(hist), qtraces


# ----------------------------------------------------------------------
# 6. SYNTHETIC data (illustration only -- NOT the scenario's data)
# ----------------------------------------------------------------------
def synthetic_data(N=1000, seed=0, pA=0.5, noise_z=0.2):
    """Uniform on the helicoid ramp: theta ~ U[0, 4pi], radial offset
    ~ U[-W/2, W/2], vertical noise N(0, noise_z^2).  Our own assumptions."""
    rng = np.random.default_rng(seed)
    s = rng.random(N) >= pA                       # False -> A, True -> B
    theta = rng.uniform(0, THETA_MAX, N)
    rho = rng.uniform(-W / 2, W / 2, N)
    v = theta + np.where(s, np.pi, 0.0)
    return np.stack([(R + rho) * np.cos(v), (R + rho) * np.sin(v),
                     C * theta + rng.normal(0, noise_z, N)], axis=1)


# ----------------------------------------------------------------------
# 7. Self-tests (verification of the derivations)
# ----------------------------------------------------------------------
def self_tests():
    rng = np.random.default_rng(1)
    out = {}
    # (a) orthonormality, det = +1, n = e_r x tau
    th = rng.uniform(0, THETA_MAX, 1000)
    err_orth, err_det, err_cross = 0.0, 0.0, 0.0
    for ph in (0.0, np.pi):
        er, tau, n = frame(th, ph)
        Q = np.stack([er, tau, n], axis=-1)       # columns
        err_orth = max(err_orth, np.abs(np.einsum("mji,mjk->mik", Q, Q) - np.eye(3)).max())
        err_det = max(err_det, np.abs(np.linalg.det(Q) - 1).max())
        err_cross = max(err_cross, np.abs(np.cross(er, tau) - n).max())
    out["orthonormality max |Q^T Q - I|"] = err_orth
    out["max |det Q - 1|"] = err_det
    out["max |e_r x tau - n|"] = err_cross
    # (b) frame-derivative identities (finite differences)
    h = 1e-6
    t0 = rng.uniform(0.1, THETA_MAX - 0.1, 50)
    er1, ta1, n1 = frame(t0 + h, 0.0)
    er0, ta0, n0 = frame(t0 - h, 0.0)
    er, tau, n = frame(t0, 0.0)
    dh = (helix(t0 + h, 0.0) - helix(t0 - h, 0.0)) / (2 * h)
    out["h' = L tau"] = np.abs(dh - L * tau).max()
    out["e_r' = (R tau - C n)/L"] = np.abs((er1 - er0) / (2 * h) - (R * tau - C * n) / L).max()
    out["tau' = -(R/L) e_r"] = np.abs((ta1 - ta0) / (2 * h) + (R / L) * er).max()
    out["n' = (C/L) e_r"] = np.abs((n1 - n0) / (2 * h) - (C / L) * er).max()
    # (c) analytic gradient vs finite difference of Q_k(theta)
    X = synthetic_data(500, seed=3)
    p = initial_params()
    gamma, _ = e_step(X, p)
    Nk, xbar, Ck = suff_stats(X, gamma)
    p["var"] = np.abs(rng.normal(size=(K, 3))) + np.array([0.5, 2.0, 0.2])  # anisotropic test
    gerr = 0.0
    for k in (0, 7, 30, 47):
        for t in rng.uniform(0.2, THETA_MAX - 0.2, 5):
            fd = (Qk_theta(t + h, k, Nk, xbar, Ck, p["var"])[0]
                  - Qk_theta(t - h, k, Nk, xbar, Ck, p["var"])[0]) / (2 * h)
            an = grad_Qk_theta(t, k, Nk, xbar, Ck, p["var"])[0]
            gerr = max(gerr, abs(fd - an) / (1 + abs(an)))
    out["gradient max rel. error vs finite diff"] = gerr
    # (d) Q_k(theta) form == direct data-sum form (up to theta-independent const)
    k, t1, t2 = 5, 1.0, 1.3
    def direct(t):
        hh = helix(t, PHASE[k]); er_, ta_, n_ = frame(t, PHASE[k])
        d = X - hh
        a = np.stack([d @ er_, d @ ta_, d @ n_], -1)
        return -0.5 * (gamma[:, k][:, None] * a**2 / initial_params()["var"][k]).sum()
    p0 = initial_params()
    out["suff-stat Q_k vs direct (difference of differences)"] = abs(
        (Qk_theta(t1, k, Nk, xbar, Ck, p0["var"])[0] - Qk_theta(t2, k, Nk, xbar, Ck, p0["var"])[0])
        - (direct(t1) - direct(t2)))
    return out


def geometry_report():
    ell = L * THETA_MAX / K_PER
    # minimum 3-D distance between the two centreline helices
    d = np.linspace(0, np.pi, 200001)
    D = np.sqrt(2 * R**2 * (1 + np.cos(d)) + C**2 * d**2)
    # ripple of equally spaced Gaussians (spacing ell)
    def ripple(s):
        f = lambda t: sum(np.exp(-(t - j * ell) ** 2 / (2 * s**2)) for j in range(-12, 13))
        return f(ell / 2) / f(0.0)
    return {
        "c = H/(4 pi)": C, "L = sqrt(R^2+c^2)": L, "curvature kappa": KAPPA,
        "dtheta (deg)": np.degrees(THETA_MAX / K_PER),
        "dz per component": C * THETA_MAX / K_PER,
        "arc length ell per component": ell,
        "horizontal arc R dtheta": R * THETA_MAX / K_PER,
        "total arc length per staircase 4 pi L": THETA_MAX * L,
        "sigma_t^2 = ell^2/4": ell**2 / 4, "ell^2/12": ell**2 / 12,
        "sigma_t = ell/2": ell / 2, "sigma_r^2 = W^2/12": W**2 / 12,
        "sagitta kappa*ell^2/8": KAPPA * ell**2 / 8,
        "chord/centroid shift kappa*sigma_t^2/2": KAPPA * (ell**2 / 4) / 2,
        "deviation at 2 sigma_t": KAPPA * (2 * ell / 2) ** 2 / 2,
        "min distance between helices (ft)": D.min(),
        "delta at minimum (rad)": d[D.argmin()],
        "ripple sigma=ell/sqrt12": ripple(ell / np.sqrt(12)),
        "ripple sigma=ell/2": ripple(ell / 2),
        "ripple sigma=ell": ripple(ell),
        "twist sigma_n (c/L^2)*sqrt(E rho^2 E s^2)": (C / L**2) * np.sqrt((W**2 / 12) * (ell**2 / 4)),
        "free params constrained": K * 4 + (K - 1),
        "free params unconstrained full-cov": K * 9 + (K - 1),
        "example mu_1 (A)": helix(np.pi / 12, 0.0),
        "example mu_25 (B)": helix(np.pi / 12, np.pi),
    }


def handedness_check(X):
    """Optional diagnostic: initial log-likelihood for data vs y-mirrored data.
    A much larger value for the mirrored data would indicate a clockwise tower."""
    p = initial_params()
    return e_step(X, p)[1], e_step(X * np.array([1, -1, 1]), p)[1]


# ----------------------------------------------------------------------
# 8. Synthetic experiments (illustration only)
# ----------------------------------------------------------------------
def run_experiments():
    print("=== GEOMETRY / ARITHMETIC CHECKS ===")
    for k_, v in geometry_report().items():
        print(f"{k_:55s} {np.round(v, 5)}")
    print("\n=== SELF-TESTS (errors; should be ~1e-8 or smaller) ===")
    for k_, v in self_tests().items():
        print(f"{k_:55s} {v:.3e}")

    X = synthetic_data(1000, seed=0)
    print("\n=== E1: SYNTHETIC data, geometric init (N=1000, seed 0) ===")
    p, hist, qt = fit(X, check=True)
    diffs = np.diff(hist)
    q_ok = all(all(np.diff(q) >= -1e-8) for q in qt)
    print("iterations:", len(hist) - 1, " logL init:", round(hist[0], 3),
          " logL final:", round(hist[-1], 3))
    print("min per-iteration change in logL:", diffs.min(), "(>=0 => monotone)")
    print("block-wise Q non-decreasing (pi -> theta -> var) in all iterations:", q_ok)
    print("mean final variances (r, t, n):", np.round(p["var"].mean(0), 4))
    print("final pi: min %.4f max %.4f sum %.6f" % (p["pi"].min(), p["pi"].max(), p["pi"].sum()))
    print("sum of pi over staircase A / B: %.4f / %.4f" % (p["pi"][:24].sum(), p["pi"][24:].sum()))
    print("max |theta_final - theta_init| (deg): %.2f" %
          np.degrees(np.abs(p["theta"] - initial_params()["theta"]).max()))
    print("all theta in [0,4pi]:", bool((p["theta"] >= 0).all() and (p["theta"] <= THETA_MAX).all()))
    print("handedness diag (data, mirrored data) init logL:", np.round(handedness_check(X), 2))

    print("\n=== E2: sensitivity to initial sigma_n (same synthetic data) ===")
    print("sigma_n0   logL_init     iters   logL_final")
    for sn in (0.1, 0.25, 0.5, 1.0, 2.0):
        p0 = initial_params(sigma_n2=sn**2)
        pf, h, _ = fit(X, p0)
        print(f"{sn:7.2f}  {h[0]:12.2f}  {len(h)-1:6d}  {h[-1]:12.3f}")

    print("\n=== E3: geometric init vs random theta (same variances) ===")
    pg, hg, _ = fit(X)
    print("geometric : iters %d  logL_final %.3f" % (len(hg) - 1, hg[-1]))
    rng = np.random.default_rng(7)
    its, lls = [], []
    for r in range(20):
        p0 = initial_params()
        p0["theta"] = rng.uniform(0, THETA_MAX, K)
        pf, h, _ = fit(X, p0)
        its.append(len(h) - 1); lls.append(h[-1])
    print("random (20 runs): iters mean %.1f (min %d, max %d)" % (np.mean(its), min(its), max(its)))
    print("random (20 runs): logL_final mean %.3f, min %.3f, max %.3f" %
          (np.mean(lls), min(lls), max(lls)))
    print("runs reaching within 1.0 of geometric logL: %d/20" %
          sum(l >= hg[-1] - 1.0 for l in lls))


if __name__ == "__main__":
    run_experiments()
