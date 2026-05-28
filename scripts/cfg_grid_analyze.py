"""Two-method response-surface analysis of the CFG grid sweep (27 points).

Methods:
  1. 2D polynomial regression (degree 2) — parametric, interpretable.
  2. Gaussian process regression (RBF + WhiteKernel) — non-parametric, gives
     calibrated posterior uncertainty.

Outputs:
  - results/cfg_sweep_contour.png — side-by-side contour plots of both fits.
  - Console: optimum coords + 95% CI from each method, plateau extent.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# (ctx, prompt) → avg_len, with possible repeats
DATA = [
    # coarse
    (1.0, 3.0, 0.94), (1.0, 7.0, 0.44), (1.0, 11.0, 0.17),
    (3.0, 3.0, 0.55), (3.0, 7.0, 0.34), (3.0, 11.0, 0.28),
    (5.0, 3.0, 0.16), (5.0, 7.0, 0.20), (5.0, 11.0, 0.25),
    # fine1
    (1.0, 2.0, 0.93), (1.0, 3.0, 0.82), (1.0, 5.0, 0.61),
    (1.5, 2.0, 0.90), (1.5, 3.0, 0.89), (1.5, 5.0, 0.67),
    (2.0, 2.0, 0.74), (2.0, 3.0, 0.74), (2.0, 5.0, 0.75),
    # fine2
    (1.0, 1.0, 0.52), (1.0, 1.5, 0.72), (1.0, 2.0, 0.75),
    (1.25, 1.0, 0.47), (1.25, 1.5, 0.62), (1.25, 2.0, 0.89),
    (1.5, 1.0, 0.48), (1.5, 1.5, 0.63), (1.5, 2.0, 0.70),
    # fine3
    (0.5, 2.0, 0.75), (0.5, 2.5, 0.79), (0.5, 3.0, 0.84),
    (1.0, 2.0, 0.82), (1.0, 2.5, 0.89), (1.0, 3.0, 0.88),
    (1.5, 2.0, 0.70), (1.5, 2.5, 0.81), (1.5, 3.0, 0.92),
]

X = np.array([(c, p) for c, p, _ in DATA])    # (27, 2)
y = np.array([v for _, _, v in DATA])         # (27,)
NOISE_SIGMA = 0.13                            # from repeats

# Evaluation grid for visualization + argmax search
ctx_grid    = np.linspace(0.5, 5.5, 101)
prompt_grid = np.linspace(0.5, 12.0, 116)
CX, PX = np.meshgrid(ctx_grid, prompt_grid)
Xgrid  = np.column_stack([CX.ravel(), PX.ravel()])

# ----------------------------------------------------------------------
# (1) Polynomial regression (degree 2)
# ----------------------------------------------------------------------
print("=" * 60)
print("Polynomial regression (degree 2)")
print("=" * 60)

poly = PolynomialFeatures(degree=2, include_bias=False)
Xp = poly.fit_transform(X)
poly_lr = LinearRegression().fit(Xp, y)
yhat_train = poly_lr.predict(Xp)
rmse_train = np.sqrt(np.mean((y - yhat_train) ** 2))
print(f"Train RMSE: {rmse_train:.3f}  (noise σ ≈ {NOISE_SIGMA:.3f})")
print(f"Coefficients (poly degree-2):")
feat_names = poly.get_feature_names_out(["ctx", "prompt"])
for name, c in zip(feat_names, poly_lr.coef_):
    print(f"  {name:>20} : {c:+.4f}")
print(f"  {'intercept':>20} : {poly_lr.intercept_:+.4f}")

Xpg = poly.transform(Xgrid)
ZP = poly_lr.predict(Xpg).reshape(CX.shape)
imax = np.unravel_index(np.argmax(ZP), ZP.shape)
poly_opt = (ctx_grid[imax[1]], prompt_grid[imax[0]])
poly_max = ZP[imax]
print(f"\nPolynomial argmax: ctx={poly_opt[0]:.2f}, prompt={poly_opt[1]:.2f} → "
      f"{poly_max:.3f}")

# ----------------------------------------------------------------------
# (2) Gaussian process regression
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Gaussian process regression (RBF + White)")
print("=" * 60)

kernel = ConstantKernel(1.0, (1e-2, 10.0)) * RBF(length_scale=[2.0, 2.0],
                                                 length_scale_bounds=(0.5, 20.0)) \
       + WhiteKernel(noise_level=NOISE_SIGMA ** 2,
                     noise_level_bounds=(1e-3, 1.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                              n_restarts_optimizer=8, random_state=0).fit(X, y)
print(f"Fitted kernel: {gp.kernel_}")

mu, std = gp.predict(Xgrid, return_std=True)
ZG  = mu.reshape(CX.shape)
ZGs = std.reshape(CX.shape)
jmax = np.unravel_index(np.argmax(ZG), ZG.shape)
gp_opt = (ctx_grid[jmax[1]], prompt_grid[jmax[0]])
gp_max = ZG[jmax]
gp_max_std = ZGs[jmax]
print(f"GP argmax: ctx={gp_opt[0]:.2f}, prompt={gp_opt[1]:.2f} → "
      f"{gp_max:.3f} ± {gp_max_std:.3f}")

# Plateau: within 1σ_GP of the optimum mean
plateau_mask = ZG >= (gp_max - gp_max_std)
n_plateau = plateau_mask.sum()
ctx_in_plateau    = ctx_grid[plateau_mask.any(axis=0)]
prompt_in_plateau = prompt_grid[plateau_mask.any(axis=1)]
if len(ctx_in_plateau) > 0 and len(prompt_in_plateau) > 0:
    print(f"Plateau (≥ {gp_max - gp_max_std:.2f}): "
          f"ctx ∈ [{ctx_in_plateau.min():.2f}, {ctx_in_plateau.max():.2f}], "
          f"prompt ∈ [{prompt_in_plateau.min():.2f}, {prompt_in_plateau.max():.2f}]  "
          f"({n_plateau} grid cells)")

# ----------------------------------------------------------------------
# Cross-validation: leave-one-out RMSE for each method
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Leave-one-out CV (noise floor: σ ≈ 0.13)")
print("=" * 60)

def loo_rmse(fit_fn):
    errs = []
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool); mask[i] = False
        yhat = fit_fn(X[mask], y[mask], X[i:i+1])[0]
        errs.append((y[i] - yhat) ** 2)
    return float(np.sqrt(np.mean(errs)))

def poly_fit(Xtr, ytr, Xte):
    p = PolynomialFeatures(degree=2, include_bias=False)
    lr = LinearRegression().fit(p.fit_transform(Xtr), ytr)
    return lr.predict(p.transform(Xte))

def gp_fit(Xtr, ytr, Xte):
    k = ConstantKernel(1.0, (1e-2, 10.0)) * RBF(length_scale=[2.0, 2.0],
                                                 length_scale_bounds=(0.5, 20.0)) \
        + WhiteKernel(noise_level=NOISE_SIGMA ** 2,
                      noise_level_bounds=(1e-3, 1.0))
    g = GaussianProcessRegressor(kernel=k, alpha=0.0, normalize_y=True,
                                 n_restarts_optimizer=4, random_state=0).fit(Xtr, ytr)
    return g.predict(Xte)

print(f"Poly  LOO RMSE: {loo_rmse(poly_fit):.3f}")
print(f"GP    LOO RMSE: {loo_rmse(gp_fit):.3f}")

# ----------------------------------------------------------------------
# Side-by-side contour plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

for ax, Z, name, opt, omax in [
    (axes[0], ZP, "Polynomial (deg 2)", poly_opt, poly_max),
    (axes[1], ZG, "Gaussian process",   gp_opt, gp_max),
]:
    levels = np.linspace(0.0, 1.0, 21)
    cs  = ax.contourf(CX, PX, np.clip(Z, 0.0, 1.0), levels=levels, cmap="viridis")
    csl = ax.contour(CX, PX, np.clip(Z, 0.0, 1.0), levels=[0.5, 0.7, 0.85],
                     colors="white", linewidths=0.6, alpha=0.8)
    ax.clabel(csl, inline=True, fontsize=8, fmt="%.2f")
    # observed data points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="white", linewidths=0.6,
               cmap="viridis", vmin=0.0, vmax=1.0, s=60, zorder=3)
    # argmax
    ax.scatter(opt[0], opt[1], marker="*", s=260, c="red",
               edgecolors="white", linewidths=1.5, zorder=4,
               label=f"argmax → {omax:.2f}")
    ax.set_xlabel("context_cfg_scale")
    ax.set_ylabel("prompt_cfg_scale")
    ax.set_title(f"{name}\nopt: ctx={opt[0]:.2f}, prompt={opt[1]:.2f} → {omax:.2f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)

cbar = fig.colorbar(cs, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("avg seq length (100-seq teacher eval)")
fig.suptitle("CFG response surface — 27 data points, noise σ ≈ 0.13",
             fontsize=12, fontweight="bold")
plt.savefig("results/cfg_sweep_contour.png", dpi=180, bbox_inches="tight")
print(f"\nSaved results/cfg_sweep_contour.png")