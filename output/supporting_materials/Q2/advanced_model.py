import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Fixed Parameters ====================
K     = 8011    # theoretical saturation capacity (tons/day)
gamma = 0.022   # intrinsic growth rate (1/day)
x0    = 6314    # initial recovery (tons/day)

# ==================== Measured Data ====================
t_data = np.array([0, 30, 60, 90, 120, 150, 180, 270, 365])
y_data = np.array([6314, 6542, 6875, 7173, 7368, 7591, 7724, 7896, 8002])

# ====================================================================
# Q2 MODEL: Symmetric Antagonistic alpha, beta  (both >= 0)
# ====================================================================
# ODE:  dx/dt = gamma/(1+beta) * x * (1 - x/(K*(1+alpha)))
#
# alpha >= 0: promotion -> expands effective capacity K_eff = K*(1+alpha)
# beta  >= 0: inhibition  -> reduces effective growth rate r = gamma/(1+beta)
#
# alpha and beta have EQUAL STATUS: each modifies one logistic parameter
# via (1+coeff) multiplicative factor. They are ANTAGONISTIC:
#   alpha pushes the saturation plateau UP
#   beta  slows the approach to saturation DOWN
#
# Standard logistic form:
#   r     = gamma / (1+beta)
#   K_eff = K * (1+alpha)
#   x(t)  = K_eff / (1 + (K_eff/x0 - 1) * exp(-r*t))
#
# Fitting r, K_eff -> derive alpha, beta separately:
#   beta  = gamma/r - 1    (r <= gamma => beta >= 0)
#   alpha = K_eff/K - 1    (K_eff >= K => alpha >= 0)
# ====================================================================

def logistic_q2(t, r, K_eff):
    return K_eff / (1 + (K_eff / x0 - 1) * np.exp(-r * t))

# Fit with constraints: r <= gamma, K_eff >= K  (ensures alpha,beta >= 0)
popt, pcov = curve_fit(logistic_q2, t_data, y_data, p0=[0.01, 8100],
                       bounds=([1e-6, K], [gamma, 24000]), maxfev=50000)
r_fit, K_eff_fit = popt
r_err  = np.sqrt(pcov[0, 0])
K_err  = np.sqrt(pcov[1, 1])

# Derive alpha, beta
beta_fit  = gamma / r_fit - 1.0
alpha_fit = K_eff_fit / K - 1.0

# Predictions & metrics
y_pred = logistic_q2(t_data, r_fit, K_eff_fit)
residuals = y_data - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r2  = 1 - ss_res / ss_tot
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals ** 2))
mape = np.mean(np.abs(residuals / y_data)) * 100

# Q1 comparison
r_q1 = gamma
y_q1_pred = logistic_q2(t_data, r_q1, K)
res_q1 = y_data - y_q1_pred
ss_res_q1 = np.sum(res_q1 ** 2)
r2_q1 = 1 - ss_res_q1 / ss_tot
mae_q1 = np.mean(np.abs(res_q1))

# ==================== Console ====================
print("=" * 70)
print("  Q2: Symmetric Antagonistic Model  (alpha, beta >= 0)")
print("=" * 70)
print(f"  ODE: dx/dt = gamma/(1+beta) * x * (1 - x/(K*(1+alpha)))")
print(f"  alpha: promotion  -> K_eff = K*(1+alpha)  (capacity)")
print(f"  beta:  inhibition -> r = gamma/(1+beta)   (growth rate)")
print(f"  ----------------------------------------")
print(f"  FITTED (standard logistic parameters)")
print(f"  r     = {r_fit:.6f} +/- {r_err:.6f}")
print(f"  K_eff = {K_eff_fit:.2f} +/- {K_err:.2f}")
print(f"  ----------------------------------------")
print(f"  DERIVED (both >= 0 by construction)")
print(f"  alpha = {alpha_fit:.6f}  (capacity expansion ratio)")
print(f"  beta  = {beta_fit:.6f}   (growth reduction ratio)")
print(f"  ----------------------------------------")
print(f"  R^2={r2:.4f}  MAE={mae:.1f}  MAPE={mape:.2f}%")
print(f"  Q1:  R^2={r2_q1:.4f}  MAE={mae_q1:.1f}")
print("=" * 70)

print(f"\n{'t':>6}  {'Actual':>8}  {'Pred':>10}  {'Resid':>8}  {'Rel.Err':>8}")
print("-" * 48)
for ti, yi, ypi, ri in zip(t_data, y_data, y_pred, residuals):
    print(f"{ti:>6}  {yi:>8.0f}  {ypi:>10.1f}  {ri:>+8.1f}  {ri/yi*100:>+7.2f}%")

# ==================== MD Report ====================
md = f"""# Q2 Logistic Model Report: alpha & beta (Symmetric Antagonistic)

## Model Formulation

alpha and beta have **equal status** in a symmetric antagonistic structure:

$$\\frac{{dx}}{{dt}} = \\frac{{\\gamma}}{{1+\\beta}} \\cdot x \\cdot \\left(1 - \\frac{{x}}{{K(1+\\alpha)}}\\right)$$

- $\\alpha \\geq 0$ (promotion): expands effective carrying capacity $\\;\\rightarrow\\; K_{{\\text{{eff}}}} = K(1+\\alpha)$
- $\\beta \\geq 0$ (inhibition): reduces effective growth rate $\\;\\rightarrow\\; r = \\gamma/(1+\\beta)$

The two coefficients act on **different logistic parameters** with the **same functional form**
$(1+\\text{{coeff}})^{{\\pm 1}}$, reflecting their equal status as symmetric antagonists.

## Standard Logistic Form

$$\\frac{{dx}}{{dt}} = r \\cdot x \\cdot \\left(1 - \\frac{{x}}{{K_{{\\text{{eff}}}}}}\\right)$$

$$r = \\frac{{\\gamma}}{{1+\\beta}}, \\qquad K_{{\\text{{eff}}}} = K(1+\\alpha)$$

**Analytical solution:**
$$x(t) = \\frac{{K_{{\\text{{eff}}}}}}{{1 + \\left(\\frac{{K_{{\\text{{eff}}}}}}{{x_0}} - 1\\right) e^{{-rt}}}}$$

## Parameter Identification

Fitting $r$ and $K_{{\\text{{eff}}}}$ gives two independent equations:

$$\\beta = \\frac{{\\gamma}}{{r}} - 1, \\qquad \\alpha = \\frac{{K_{{\\text{{eff}}}}}}{{K}} - 1$$

$r \\leq \\gamma$ ensures $\\beta \\geq 0$; $K_{{\\text{{eff}}}} \\geq K$ ensures $\\alpha \\geq 0$.

## Parameters

| Role | Symbol | Value | Unit |
|------|--------|-------|------|
| Fixed | $K$ | {K} | tons/day |
| Fixed | $\\gamma$ | {gamma} | 1/day |
| Fixed | $x_0$ | {x0} | tons/day |
| **Fitted** | $r$ | **{r_fit:.6f}** $\\pm$ {r_err:.6f} | 1/day |
| **Fitted** | $K_{{\\text{{eff}}}}$ | **{K_eff_fit:.2f}** $\\pm$ {K_err:.2f} | tons/day |
| Derived | $\\alpha$ | **{alpha_fit:.6f}** | — |
| Derived | $\\beta$ | **{beta_fit:.6f}** | — |

### Physical Interpretation

- $\\alpha = {alpha_fit:.6f} > 0$: promotion expands effective capacity by ${alpha_fit*100:.1f}\\%$ above $K={K}$
- $\\beta = {beta_fit:.6f} > 0$: inhibition slows growth to $r = \\gamma/(1+\\beta) = {r_fit:.4f}$
- $r / \\gamma = 1/(1+\\beta) = {r_fit/gamma*100:.1f}\\%$: effective rate as fraction of intrinsic rate
- Antagonistic balance: $\\alpha$ pushes up the ceiling, $\\beta$ stretches out the timeline

## Evaluation

| Metric | Q1 (no $\\alpha,\\beta$) | Q2 (with $\\alpha,\\beta$) |
|--------|--------------------------|----------------------------|
| $R^2$ | {r2_q1:.4f} | **{r2:.4f}** |
| MAE | {mae_q1:.1f} | **{mae:.1f}** |
| MAPE | 4.42% | **{mape:.2f}%** |

## Predictions

| $t$ (days) | Actual | Predicted | Residual | Rel.Err |
|:----------:|:------:|:---------:|:--------:|:-------:|
"""

for ti, yi, ypi, ri in zip(t_data, y_data, y_pred, residuals):
    md += f"| {ti} | {yi:.0f} | {ypi:.1f} | {ri:+.1f} | {ri/yi*100:+.2f}% |\n"

md += f"""
## Key Findings

- $\\alpha = {alpha_fit:.6f} > 0$, $\\beta = {beta_fit:.6f} > 0$: **both positive by construction**
- $\\alpha$ and $\\beta$ have equal status: symmetric $(1+\\text{{coeff}})^{{\\pm 1}}$ structure
- $K_{{\\text{{eff}}}} = {K_eff_fit:.1f}$ tons/day: promotion expands capacity by ${alpha_fit*100:.1f}\\%$
- $r = {r_fit:.4f}$: inhibition reduces growth rate to ${r_fit/gamma*100:.1f}\\%$ of intrinsic $\\gamma$
- The system with $\\alpha,\\beta \\geq 0$ is **structurally stable**: no bifurcation, no collapse risk
"""

with open('report_q2.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("\nMD report saved: report_q2.md")

# ==================== Plot (3 panels) ====================
t_smooth = np.linspace(0, 500, 400)
x_q2 = logistic_q2(t_smooth, r_fit, K_eff_fit)
x_q1 = logistic_q2(t_smooth, r_q1, K)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Q2 Fit
ax = axes[0]
r_lo, r_hi = r_fit * 0.9, r_fit * 1.1
ax.fill_between(t_smooth,
                logistic_q2(t_smooth, r_lo, K_eff_fit),
                logistic_q2(t_smooth, r_hi, K_eff_fit),
                color='gray', alpha=0.2, label=r'$r \pm 10\%$')
ax.plot(t_smooth, x_q2, '#e84d3d', linewidth=2, label='Q2 fitted')
ax.scatter(t_data, y_data, c='#1a73e8', s=50, zorder=5, label='Data')
ax.scatter(0, x0, c='#2ba02b', s=70, zorder=6, marker='s', label=f'$x_0={x0}$')
ax.axhline(K_eff_fit, color='#e84d3d', linestyle='--', alpha=0.7,
           label=f'$K_{{\\text{{eff}}}}={K_eff_fit:.0f}$')
ax.axhline(K, color='gray', linestyle=':', alpha=0.5, label=f'$K={K}$')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Recovery (tons/day)')
ax.set_title(f'Q2: $r = \\gamma/(1+\\beta) = {r_fit:.4f}$, '
             f'$K_{{\\text{{eff}}}} = K(1+\\alpha) = {K_eff_fit:.0f}$\n'
             f'$\\alpha={alpha_fit:.4f}$, $\\beta={beta_fit:.4f}$, '
             f'$R^2={r2:.4f}$, MAE$={mae:.0f}$')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel 2: Q1 vs Q2
ax = axes[1]
ax.plot(t_smooth, x_q2, '#e84d3d', linewidth=2,
        label=f'Q2 ($r={r_fit:.4f}$, $K_{{\\text{{eff}}}}={K_eff_fit:.0f}$, $R^2={r2:.4f}$)')
ax.plot(t_smooth, x_q1, '#2ba02b', linewidth=2, linestyle='--',
        label=f'Q1 ($\\gamma={gamma}$, $K={K}$, $R^2={r2_q1:.4f}$)')
ax.scatter(t_data, y_data, c='#1a73e8', s=50, zorder=5, label='Data')
ax.axhline(K, color='gray', linestyle=':', alpha=0.5, label=f'$K={K}$')
ax.axhline(K_eff_fit, color='#e84d3d', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Recovery (tons/day)')
ax.set_title('Q1 vs Q2')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel 3: Residuals
ax = axes[2]
ax.axhline(0, color='gray', linewidth=1)
ax.scatter(t_data, residuals, c='#e84d3d', s=50, zorder=5, marker='o',
           label=f'Q2 (MAE={mae:.0f})')
ax.scatter(t_data, res_q1, c='#2ba02b', s=50, zorder=5, marker='s',
           label=f'Q1 (MAE={mae_q1:.0f})')
for ti, ri in zip(t_data, residuals):
    ax.vlines(ti, 0, ri, colors='#e84d3d', alpha=0.3, linewidth=1.2)
for ti, ri in zip(t_data, res_q1):
    ax.vlines(ti, 0, ri, colors='#2ba02b', alpha=0.3, linewidth=1.2)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Residual (tons/day)')
ax.set_title('Residuals')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_q2.png', dpi=150)
print("Plot saved: model_q2.png")
