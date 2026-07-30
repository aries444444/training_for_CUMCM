import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Fixed Parameters ====================
K     = 8011
gamma = 0.022
x0    = 6314

# ==================== Measured Data ====================
t_data = np.array([0, 30, 60, 90, 120, 150, 180, 270, 365])
y_data = np.array([6314, 6542, 6875, 7173, 7368, 7591, 7724, 7896, 8002])

# ====================================================================
# Q3 STABILITY ANALYSIS
# Model: dx/dt = gamma/(1+beta) * x * (1 - x/(K*(1+alpha)))
#   alpha >= 0: promotion -> K_eff = K*(1+alpha)  (capacity expansion)
#   beta  >= 0: inhibition  -> r = gamma/(1+beta)   (growth reduction)
#
# Key fact: alpha, beta >= 0 BY DEFINITION.
#   => r = gamma/(1+beta) > 0 ALWAYS
#   => K_eff = K*(1+alpha) >= K
#   => The system is STRUCTURALLY STABLE in its entire design space.
#   => NO bifurcation exists for any valid (alpha, beta).
#
# Q3 focus:
#   1. Equilibrium & eigenvalue analysis
#   2. Structural stability proof
#   3. Practical safety boundaries (performance-based, not bifurcation-based)
#   4. Safe parameter ranges from operational constraints (r_min, K_eff_min)
# ====================================================================

def logistic_q3(t, r, K_eff):
    return K_eff / (1 + (K_eff / x0 - 1) * np.exp(-r * t))

# Fit
popt, pcov = curve_fit(logistic_q3, t_data, y_data, p0=[0.01, 8100],
                       bounds=([1e-6, K], [gamma, 24000]), maxfev=50000)
r_fit, K_eff_fit = popt
r_err  = np.sqrt(pcov[0, 0])
K_err  = np.sqrt(pcov[1, 1])

beta_fit  = gamma / r_fit - 1.0
alpha_fit = K_eff_fit / K - 1.0

y_pred = logistic_q3(t_data, r_fit, K_eff_fit)
residuals = y_data - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r2  = 1 - ss_res / ss_tot
mae = np.mean(np.abs(residuals))
mape = np.mean(np.abs(residuals / y_data)) * 100

# ==================== Console ====================
print("=" * 70)
print("  Q3: Stability Analysis (alpha, beta >= 0 by definition)")
print("=" * 70)
print(f"  Model: dx/dt = gamma/(1+beta)*x*(1-x/(K*(1+alpha)))")
print(f"  Fitted: r={r_fit:.6f}, K_eff={K_eff_fit:.1f}")
print(f"          alpha={alpha_fit:.6f}, beta={beta_fit:.6f}")
print(f"  ----------------------------------------")
print(f"  EQUILIBRIA")
print(f"  x1* = 0:           eigenvalue = +{r_fit:.6f}  -> UNSTABLE")
print(f"  x2* = {K_eff_fit:.1f}:      eigenvalue = -{r_fit:.6f}  -> STABLE")
print(f"  ----------------------------------------")
print(f"  STRUCTURAL STABILITY")
print(f"  alpha,beta >= 0 by definition => r = gamma/(1+beta) > 0")
print(f"  => No bifurcation in the entire valid parameter space.")
print(f"  => System ALWAYS converges to K_eff from any x(0) > 0.")
print(f"  ----------------------------------------")
print(f"  PRACTICAL SAFETY (performance-based)")
print(f"  Operational constraint: r >= r_min")
print(f"    r_min = 0.005 -> beta_max = {gamma/0.005 - 1:.2f}")
print(f"    Current beta = {beta_fit:.2f} (margin = {gamma/0.005 - 1 - beta_fit:.2f})")
print(f"  Operational constraint: K_eff >= K_min")
print(f"    K_min = K -> alpha >= 0 (always satisfied)")
print("=" * 70)

# ====================================================================
# MD REPORT
# ====================================================================
r_min = 0.005
beta_max = gamma / r_min - 1

md = f"""# Q3 Stability Analysis: Symmetric Antagonistic Model

## Model (from Q2)

$$\\frac{{dx}}{{dt}} = \\frac{{\\gamma}}{{1+\\beta}} \\cdot x \\cdot \\left(1 - \\frac{{x}}{{K(1+\\alpha)}}\\right)
= r \\cdot x \\cdot \\left(1 - \\frac{{x}}{{K_{{\\text{{eff}}}}}}\\right)$$

$$r = \\frac{{\\gamma}}{{1+\\beta}}, \\qquad K_{{\\text{{eff}}}} = K(1+\\alpha)$$

**By definition, $\\alpha \\geq 0$ (promotion) and $\\beta \\geq 0$ (inhibition).**

## Equilibrium Analysis

Setting $dx/dt = 0$:

| Equilibrium | Value | Eigenvalue | Stability |
|-------------|-------|-----------|-----------|
| $x_1^*$ (zero) | $0$ | $\\lambda_1 = r = \\gamma/(1+\\beta)$ | **Unstable** ($r > 0$ always) |
| $x_2^*$ (sustainable) | $K_{{\\text{{eff}}}} = K(1+\\alpha)$ | $\\lambda_2 = -r$ | **Asymptotically Stable** |

**Proof**: $\\beta \\geq 0 \\Rightarrow 1+\\beta \\geq 1 \\Rightarrow r = \\gamma/(1+\\beta) \\in (0, \\gamma]$.
Therefore $r > 0$ for all valid $\\beta$, and $x_2^*$ is always stable.

## Structural Stability

Because $\\alpha, \\beta \\geq 0$ **by definition** (they are promotional and inhibitory coefficients),
the entire valid parameter space satisfies $r > 0$. There is **no bifurcation anywhere**
in the design region. The recycling system is **structurally stable**:

> Any positive initial recovery volume $x(0) > 0$ will converge to
> $K_{{\\text{{eff}}}}$ as $t \\to \\infty$, for **all** $\\alpha \\geq 0$, $\\beta \\geq 0$.

This is a strong result: "沪尚回收" will not experience mathematical collapse
(断崖下跌) as long as the promotion and inhibition coefficients remain non-negative
— which they do by construction. The operational risk is **performance degradation**,
not structural instability.

## Practical Safety Boundaries

While the system is mathematically stable everywhere, operational viability
requires **adequate performance**:

### Growth Rate Constraint

$$r = \\frac{{\\gamma}}{{1+\\beta}} \\geq r_{{\\min}} \\quad \\Rightarrow \\quad
\\beta \\leq \\frac{{\\gamma}}{{r_{{\\min}}}} - 1$$

With $r_{{\\min}} = {r_min}$ (minimum acceptable growth rate):
$$\\beta_{{\\max}} = \\frac{{{gamma}}}{{{r_min}}} - 1 = {beta_max:.2f}$$

### Capacity Constraint

$$K_{{\\text{{eff}}}} = K(1+\\alpha) \\geq K_{{\\min}} \\quad \\Rightarrow \\quad
\\alpha \\geq \\frac{{K_{{\\min}}}}{{K}} - 1$$

With $K_{{\\min}} = K$ (baseline capacity): $\\alpha \\geq 0$ — always satisfied.

### Safe Operating Ranges

| Parameter | Lower Bound | Upper Bound | Current | Status |
|-----------|-------------|-------------|---------|--------|
| $\\alpha$ | $0$ (definition) | — | ${alpha_fit:.4f}$ | Safe (capacity at ${K_eff_fit/K*100:.1f}%$ of $K$) |
| $\\beta$ | $0$ (definition) | ${beta_max:.2f}$ (from $r_{{\\min}}$) | ${beta_fit:.2f}$ | Safe (r at ${r_fit/gamma*100:.1f}%$ of $\\gamma$) |

### Safety Margins

| Metric | Value |
|--------|-------|
| $\\beta$ margin to $\\beta_{{\\max}}$ | ${beta_max - beta_fit:.2f}$ |
| $r$ margin above $r_{{\\min}}$ | ${r_fit - r_min:.4f}$ |
| $r/\\gamma$ (effective rate fraction) | ${r_fit/gamma*100:.1f}\\%$ |

## Decoupled Regulation

A key advantage of this model: $\\alpha$ and $\\beta$ are **fully decoupled**.

- **$\\alpha$ only affects $K_{{\\text{{eff}}}}$**: adjusting promotion changes
  the saturation ceiling without affecting convergence speed
- **$\\beta$ only affects $r$**: controlling inhibition changes the convergence
  speed without affecting the saturation ceiling

This enables **independent tuning**: policymakers can raise $\\alpha$ to expand
total capacity while separately managing $\\beta$ to maintain adequate growth speed.

## Key Findings

1. **Structurally stable**: $r = \\gamma/(1+\\beta) > 0$ for all $\\alpha,\\beta \\geq 0$ — no bifurcation exists.
2. **Collapse impossible by design**: as long as promotional and inhibitory effects remain positive by definition, $x(t) \\to K_{{\\text{{eff}}}} > 0$ always.
3. **Risk is performance, not stability**: the danger lies in $\\beta$ growing large enough to make $r$ impractically small, or $\\alpha$ shrinking to near zero.
4. **Decoupled control**: $\\alpha \\to K_{{\\text{{eff}}}}$, $\\beta \\to r$ — independently adjustable.
5. **Safety boundary**: $\\beta \\leq \\gamma/r_{{\\min}} - 1 = {beta_max:.2f}$ for operational viability with $r_{{\\min}} = {r_min}$.
"""

with open('report_q3.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("\nMD report saved: report_q3.md")

# ====================================================================
# FIGURE 1: 5-panel stability analysis
# ====================================================================
t_smooth = np.linspace(0, 500, 400)
x_q3 = logistic_q3(t_smooth, r_fit, K_eff_fit)

fig = plt.figure(figsize=(16, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35)

# Panel 1: Model Fit
ax = fig.add_subplot(gs[0, 0])
ax.fill_between(t_smooth,
                logistic_q3(t_smooth, r_fit*0.9, K_eff_fit),
                logistic_q3(t_smooth, r_fit*1.1, K_eff_fit),
                color='gray', alpha=0.2, label=r'$r \pm 10\%$')
ax.plot(t_smooth, x_q3, '#e84d3d', linewidth=2, label='Model')
ax.scatter(t_data, y_data, c='#1a73e8', s=50, zorder=5, label='Data')
ax.axhline(K_eff_fit, color='#e84d3d', linestyle='--', alpha=0.7,
           label=f'$K_{{eff}}=K(1+\\alpha)={K_eff_fit:.0f}$')
ax.axhline(K, color='gray', linestyle=':', alpha=0.5, label=f'$K={K}$')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Recovery (tons/day)')
ax.set_title(f'Model Fit: $r={r_fit:.4f}$, $K_{{eff}}={K_eff_fit:.0f}$\n'
             f'$\\alpha={alpha_fit:.4f}$, $\\beta={beta_fit:.2f}$, '
             f'$R^2={r2:.4f}$, MAE$={mae:.1f}$')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel 2: Phase Portrait (alpha, beta >= 0 region)
ax = fig.add_subplot(gs[0, 1])
x_range = np.linspace(-200, max(K_eff_fit, K) * 1.2, 500)
dxdt = r_fit * x_range * (1 - x_range / K_eff_fit)
ax.plot(x_range, dxdt, '#1a73e8', linewidth=2)
ax.axhline(0, color='gray', linewidth=1)
ax.axvline(0, color='gray', linewidth=1)
ax.scatter(0, 0, c='#e84d3d', s=120, zorder=6, marker='o', edgecolors='black',
           label='$x_1^*=0$ (unstable)')
ax.scatter(K_eff_fit, 0, c='green', s=120, zorder=6, marker='s', edgecolors='black',
           label=f'$x_2^*={K_eff_fit:.0f}$ (stable)')
for xp in np.linspace(300, K_eff_fit * 0.7, 5):
    ax.annotate('', xy=(xp + 200, 0), xytext=(xp, 0),
                arrowprops=dict(arrowstyle='->', color='#e84d3d', lw=2.5))
for xp in np.linspace(K_eff_fit + 500, K_eff_fit * 1.3, 4):
    ax.annotate('', xy=(xp - 200, 0), xytext=(xp, 0),
                arrowprops=dict(arrowstyle='->', color='#e84d3d', lw=2.5))
ax.set_xlabel('x (tons/day)')
ax.set_ylabel('dx/dt (tons/day^2)')
ax.set_title(f'Phase Portrait: $r={r_fit:.4f} > 0$ always\n'
             f'$\\alpha,\\beta \\geq 0$ by definition $\\Rightarrow$ structurally stable')
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

# Panel 3: K_eff vs alpha (valid range: alpha >= 0)
ax = fig.add_subplot(gs[1, 0])
alpha_vals = np.linspace(0, alpha_fit + 0.3, 200)
K_eff_vals = K * (1 + alpha_vals)
ax.plot(alpha_vals, K_eff_vals, '#1a73e8', linewidth=2.5, label='$K_{eff} = K(1+\\alpha)$')
ax.axhline(K, color='gray', linestyle=':', alpha=0.5, label=f'$K={K}$')
ax.scatter(alpha_fit, K_eff_fit, c='#e84d3d', s=120, zorder=10, marker='*',
           edgecolors='black', label=f'Current $\\alpha={alpha_fit:.4f}$')
# Shade: alpha >= 0 is the entire valid region
ax.axvspan(0, alpha_vals[-1], alpha=0.05, color='green')
ax.set_xlabel(r'$\alpha$ (>= 0 by definition)')
ax.set_ylabel(r'$K_{eff}$ (tons/day)')
ax.set_title(f'Capacity Expansion: $K_{{eff}} = K(1+\\alpha)$\n'
             f'$\\alpha \\geq 0$ always $\\Rightarrow$ $K_{{eff}} \\geq K$, structurally safe')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel 4: r vs beta (valid range: beta >= 0)
ax = fig.add_subplot(gs[1, 1])
beta_vals = np.linspace(0, max(beta_fit * 2.2, beta_max * 1.2), 200)
r_vals = gamma / (1 + beta_vals)
ax.plot(beta_vals, r_vals, '#e84d3d', linewidth=2.5, label=r'$r = \gamma/(1+\beta)$')
ax.axhline(gamma, color='gray', linestyle=':', alpha=0.5, label=f'$\\gamma={gamma}$')
ax.axhline(r_min, color='orange', linestyle='--', alpha=0.7,
           label=f'$r_{{min}}={r_min}$ (practical)')
ax.scatter(beta_fit, r_fit, c='#1a73e8', s=120, zorder=10, marker='*',
           edgecolors='black', label=f'Current $\\beta={beta_fit:.2f}$')
# Shade safe operating region
idx_safe = r_vals >= r_min
ax.fill_between(beta_vals, r_vals, alpha=0.08, color='green',
                where=idx_safe)
ax.fill_between(beta_vals, r_vals, alpha=0.08, color='orange',
                where=~idx_safe)
ax.set_xlabel(r'$\beta$ (>= 0 by definition)')
ax.set_ylabel('r (effective growth rate)')
ax.set_title(f'Growth Rate: $r = \\gamma/(1+\\beta)$\n'
             f'$r > 0$ always; $r \\geq {r_min}$ for $\\beta \\leq {beta_max:.2f}$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 5: Safe operating region in (alpha, beta) space
ax = fig.add_subplot(gs[2, :])
alpha_grid = np.linspace(0, alpha_fit + 0.5, 200)
beta_grid = np.linspace(0, max(beta_fit * 2, beta_max * 1.2), 200)
AA, BB = np.meshgrid(alpha_grid, beta_grid)
R_grid = gamma / (1 + BB)

# Practical safety: r >= r_min
safe_mask = R_grid >= r_min

ax.contourf(AA, BB, safe_mask.astype(float), levels=[0, 0.5, 1],
            colors=['#ffddcc', '#ccffcc'], alpha=0.5)
ax.contour(AA, BB, R_grid, levels=[r_min], colors=['orange'], linewidths=2,
           linestyles='--')
current_point = ax.scatter(alpha_fit, beta_fit, c='#1a73e8', s=200, zorder=10, marker='*',
           edgecolors='black', linewidth=1.5,
           label=f'Current ($\\alpha={alpha_fit:.4f}$, $\\beta={beta_fit:.2f}$)')

safe_patch = Patch(color='#ccffcc', alpha=0.5, label=f'$r \\geq {r_min}$ (operational)')
degraded_patch = Patch(color='#ffddcc', alpha=0.5, label=f'$r < {r_min}$ (degraded)')
threshold_line, = ax.plot([], [], color='orange', linewidth=2, linestyle='--',
                          label=f'$\\beta = {beta_max:.2f}$ ($r={r_min}$)')
ax.legend(handles=[safe_patch, degraded_patch, threshold_line, current_point],
          fontsize=7, loc='upper right')
ax.set_xlabel(r'$\alpha$ (>= 0)')
ax.set_ylabel(r'$\beta$ (>= 0)')
ax.set_title(f'Safe Operating Region\n'
             f'Practical boundary: $r = \\gamma/(1+\\beta) \\geq {r_min}$')
ax.grid(True, alpha=0.3)

plt.savefig('stability_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved: stability_analysis.png")

# ====================================================================
# FIGURE 2: Supplementary analysis
# ====================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: Convergence trajectories
ax = axes2[0]
def simulate(r_val, K_eff_val, x0_val, t_arr):
    x = np.zeros(len(t_arr))
    x[0] = x0_val
    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]
        dx = r_val * x[i - 1] * (1 - x[i - 1] / K_eff_val)
        x[i] = x[i - 1] + dx * dt
    return x

t_traj = np.linspace(0, 500, 400)
initial_conditions = [1000, 4000, 6314, 9000, 14000]
colors_traj = plt.cm.viridis(np.linspace(0.1, 0.9, len(initial_conditions)))
for ic, col in zip(initial_conditions, colors_traj):
    traj = simulate(r_fit, K_eff_fit, ic, t_traj)
    ax.plot(t_traj, traj, color=col, linewidth=1.5, alpha=0.8, label=f'$x_0={ic}$')
ax.axhline(K_eff_fit, color='green', linestyle='--', linewidth=1.5,
           label=f'$K_{{eff}}={K_eff_fit:.0f}$')
ax.axhline(0, color='red', linestyle=':', linewidth=1.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel('x(t) (tons/day)')
ax.set_title(f'All $x_0 > 0 \\to K_{{eff}}$ (structurally stable)\n'
             f'$\\alpha={alpha_fit:.4f}$, $\\beta={beta_fit:.2f}$, $r={r_fit:.4f}$')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel B: Decoupled effects
ax = axes2[1]
alpha_show = np.linspace(0, alpha_fit + 0.2, 100)
beta_show  = np.linspace(0, beta_fit + 1, 100)
ax2b = ax.twinx()
l1, = ax.plot(alpha_show, K*(1+alpha_show)/K, '#1a73e8', linewidth=2,
              label=r'$K_{eff}/K = 1+\alpha$')
l2, = ax2b.plot(beta_show, gamma/(1+beta_show)/gamma, '#e84d3d', linewidth=2,
                label=r'$r/\gamma = 1/(1+\beta)$')
ax.axhline(1, color='gray', linestyle=':', alpha=0.5)
ax.axvline(alpha_fit, color='#1a73e8', linestyle='--', alpha=0.4)
ax2b.axvline(beta_fit, color='#e84d3d', linestyle='--', alpha=0.4)
ax.scatter(alpha_fit, 1+alpha_fit, c='#1a73e8', s=80, zorder=10, marker='*', edgecolors='black')
ax2b.scatter(beta_fit, 1/(1+beta_fit), c='#e84d3d', s=80, zorder=10, marker='*', edgecolors='black')
ax.set_xlabel(r'$\alpha$ (blue) / $\beta$ (red)')
ax.set_ylabel(r'$K_{eff} / K$', color='#1a73e8')
ax2b.set_ylabel(r'$r / \gamma$', color='#e84d3d')
ax.set_title('Decoupled Control\n$\\alpha \\to K_{eff}$ only, $\\beta \\to r$ only')
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

# Panel C: Practical safety boundary (r vs beta with threshold)
ax = axes2[2]
beta_full = np.linspace(0, beta_fit * 2.5, 300)
r_full = gamma / (1 + beta_full)
ax.plot(beta_full, r_full, '#e84d3d', linewidth=2.5)
ax.axhline(r_min, color='orange', linestyle='--', linewidth=1.5,
           label=f'$r_{{min}}={r_min}$')
ax.axvline(beta_max, color='orange', linestyle=':', linewidth=1)
ax.axhline(gamma, color='gray', linestyle=':', alpha=0.5, label=f'$\\gamma={gamma}$')
ax.scatter(beta_fit, r_fit, c='#1a73e8', s=120, zorder=10, marker='*',
           edgecolors='black', label=f'Current')
ax.fill_between(beta_full[beta_full <= beta_max], r_full[beta_full <= beta_max],
                alpha=0.08, color='green')
ax.fill_between(beta_full[beta_full >= beta_max], r_full[beta_full >= beta_max],
                alpha=0.08, color='orange')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('r (effective growth rate)')
ax.set_title(f'Practical Safety Boundary\n'
             f'Operational: $\\beta \\leq {beta_max:.2f}$ ($r \\geq {r_min}$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('collapse_scenario.png', dpi=150, bbox_inches='tight')
print("Plot saved: collapse_scenario.png")

print("\nDone! All Q3 analysis complete.")
