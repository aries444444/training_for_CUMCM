import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Fixed Parameters (from 基础固定参数.md) ====================
K     = 8011    # saturation capacity (tons/day)
gamma = 0.022   # intrinsic growth rate
x0    = 6314    # initial recovery amount (tons/day)

# ==================== Measured Data ====================
t_data = np.array([0, 30, 60, 90, 120, 150, 180, 270, 365])
y_data = np.array([6314, 6542, 6875, 7173, 7368, 7591, 7724, 7896, 8002])

# ==================== Logistic Model (ZERO free parameters) ====================
# x(t) = K / (1 + (K/x0 - 1) * exp(-gamma * t))
A = K / x0 - 1   # = 0.26877


def logistic(t):
    return K / (1 + A * np.exp(-gamma * t))


# ==================== Predictions & Metrics ====================
y_pred = logistic(t_data)
residuals = y_data - y_pred

ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r2    = 1 - ss_res / ss_tot
mae   = np.mean(np.abs(residuals))
rmse  = np.sqrt(np.mean(residuals ** 2))
mape  = np.mean(np.abs(residuals / y_data)) * 100

# ==================== Console Output ====================
print("=" * 60)
print("  Logistic Model: x(t) = K/(1 + (K/x0-1)*e^(-gamma*t))")
print("=" * 60)
print(f"  Fixed:  K     = {K}      tons/day")
print(f"  Fixed:  gamma = {gamma}       1/day")
print(f"  Fixed:  x0    = {x0}      tons/day")
print(f"  Derived A = K/x0-1 = {A:.5f}")
print(f"  --------------------------------------------------")
print(f"  R^2   = {r2:.4f}")
print(f"  MAE   = {mae:.1f}   tons/day")
print(f"  RMSE  = {rmse:.1f}  tons/day")
print(f"  MAPE  = {mape:.2f}%")
print("=" * 60)
print(f"\n{'t(天)':>6}  {'实际 y':>8}  {'预测 x(t)':>10}  {'残差':>8}  {'误差%':>8}")
print("-" * 48)
for ti, yi, ypi, ri in zip(t_data, y_data, y_pred, residuals):
    print(f"{ti:>6}  {yi:>8.0f}  {ypi:>10.1f}  {ri:>+8.1f}  {ri/yi*100:>+7.2f}%")

# ==================== MD Report ====================
md = f"""# Logistic Growth Model Report

## Parameters (All Fixed)

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Saturation capacity | $K$ | {K} | tons/day | Given |
| Intrinsic growth rate | $\\gamma$ | {gamma} | 1/day | Given |
| Initial recovery | $x_0$ | {x0} | tons/day | Given |
| Derived constant | $A = K/x_0 - 1$ | {A:.5f} | — | Computed |

**Model equation:**

$$x(t) = \\frac{{K}}{{1 + \\left(\\frac{{K}}{{x_0}} - 1\\right) e^{{-\\gamma t}}}} = \\frac{{{K}}}{{1 + {A:.5f} \\cdot e^{{-{gamma} \\cdot t}}}}$$

**Governing ODE:**

$$\\frac{{dx}}{{dt}} = \\gamma \\cdot x \\cdot \\left(1 - \\frac{{x}}{{K}}\\right), \\quad x(0) = x_0$$

## Model Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| $R^2$ | {r2:.4f} | {('Good' if r2 > 0.8 else 'Poor' if r2 < 0.6 else 'Moderate')} |
| MAE | {mae:.1f} tons/day | Avg absolute deviation |
| RMSE | {rmse:.1f} tons/day | RMS deviation |
| MAPE | {mape:.2f}% | Avg relative error |

## Predictions vs Actual

| $t$ (days) | Actual $y$ | Predicted $x(t)$ | Residual | Rel. Error |
|:----------:|:----------:|:----------------:|:--------:|:----------:|
"""

for ti, yi, ypi, ri in zip(t_data, y_data, y_pred, residuals):
    md += f"| {ti:>4} | {yi:>8.0f} | {ypi:>13.1f} | {ri:>+8.1f} | {ri/yi*100:>+7.2f}% |\n"

md += f"""
## Forecast

| $t$ (days) | $x(t)$ (tons/day) | % of $K$ |
|:----------:|:-----------------:|:--------:|
"""

for ti in [400, 450, 500, 600, 730]:
    xi = logistic(ti)
    md += f"| {ti:>4} | {xi:.1f} | {xi/K*100:.2f}% |\n"

md += f"""
## Residual Analysis

| Statistic | Value |
|-----------|-------|
| Max positive residual | {residuals.max():.1f} at t = {t_data[np.argmax(residuals)]} days |
| Max negative residual | {residuals.min():.1f} at t = {t_data[np.argmin(residuals)]} days |
| Mean residual | {np.mean(residuals):.1f} |

**Observation:** The model systematically **overestimates** early-stage recovery (t = 30–150 days),
with errors reaching -{abs(residuals.min()):.0f} tons/day. This indicates that the theoretical
$\\gamma = {gamma}$ is larger than the empirically observed effective growth rate.
Introducing modulation parameters ($\\alpha$ for promotion, $\\beta$ for inhibition)
could resolve this discrepancy.

## Stage-wise Behavior

| Phase | $t$ (days) | Characteristic |
|-------|------------|----------------|
| Initiation | 0–60 | System optimization, promotion rollout; slow growth |
| Rapid growth | 60–180 | Community engagement peaks; recovery volume accelerates |
| Saturation | 180–365+ | Growth decelerates; asymptotically approaches $K = {K}$ |
"""

with open('report.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("\nMD report saved: report.md")
print("Generating plots...")

# ==================== Plotting ====================
t_smooth = np.linspace(0, 500, 400)
x_smooth = logistic(t_smooth)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ===== Panel 1: Model Curve + Data =====
ax = axes[0]
# gamma +/- 10% sensitivity band
gam_lo, gam_hi = gamma * 0.9, gamma * 1.1
x_lo = K / (1 + A * np.exp(-gam_lo * t_smooth))
x_hi = K / (1 + A * np.exp(-gam_hi * t_smooth))
ax.fill_between(t_smooth, x_lo, x_hi, color='gray', alpha=0.2,
                label=r'$\gamma \pm 10\%$')
ax.plot(t_smooth, x_smooth, '#e84d3d', linewidth=2, label='x(t)')
ax.scatter(t_data, y_data, c='#1a73e8', s=50, zorder=5, label='Data')
ax.scatter(0, x0, c='#2ba02b', s=80, zorder=6, marker='s',
           label=f'$x_0$ = {x0} (fixed)')
ax.axhline(K, color='gray', linestyle=':', alpha=0.5, label=f'K = {K}')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Recovery (tons/day)')
ax.set_title(f'Logistic Model: Zero Free Parameters\n'
             f'$R^2={r2:.4f}$,  MAE={mae:.0f}$,  MAPE={mape:.1f}%')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# ===== Panel 2: Residuals (stem + MAE band) =====
ax = axes[1]
ax.axhline(0, color='gray', linewidth=1)
colors = ['#e84d3d' if r < 0 else '#2ba02b' for r in residuals]
ax.stem(t_data, residuals, linefmt='grey', markerfmt='o', basefmt=' ')
for ti, ri, ci in zip(t_data, residuals, colors):
    ax.scatter(ti, ri, c=ci, s=60, zorder=5)
    ax.vlines(ti, 0, ri, colors=ci, alpha=0.4, linewidth=1.5)
ax.axhline( mae, color='red', linestyle='--', alpha=0.6, label=f'MAE = {mae:.0f}')
ax.axhline(-mae, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Residual (tons/day)')
ax.set_title(f'Residuals  (RMSE={rmse:.0f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ===== Panel 3: Three-Phase Analysis =====
ax = axes[2]
ax.plot(t_smooth, x_smooth, 'b-', linewidth=2, label='x(t)')
for ti, yi, ri in zip(t_data, y_data, residuals):
    c = '#2ba02b' if ri >= 0 else '#e84d3d'
    ax.scatter(ti, yi, c=c, s=60, zorder=5)
ax.axhline(K, color='gray', linestyle=':', alpha=0.5, label=f'K = {K}')
# Phase dividers
ax.axvline(60,  color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax.axvline(180, color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax.text(25,  K+55, 'Phase I\nInitiation',   ha='center', fontsize=9, color='gray')
ax.text(120, K+55, 'Phase II\nRapid Growth', ha='center', fontsize=9, color='gray')
ax.text(320, K+55, 'Phase III\nSaturation',  ha='center', fontsize=9, color='gray')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Recovery (tons/day)')
ax.set_title('Three-Phase Analysis')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_fit.png', dpi=150)
print("Plot saved: logistic_fit.png")
