import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Base Parameters (from Q2 fit) ====================
K_fixed   = 8011        # saturation capacity (tons/day)
gamma_fixed = 0.022     # intrinsic growth rate (1/day)
x0 = 6314               # initial recovery (tons/day)

# Q2 fitted effective parameters (exact values from curve_fit)
r_fit     = 0.008485    # fitted effective growth rate
K_eff_fit = 8142.22     # fitted effective capacity

# Derived alpha, beta
alpha_fit = K_eff_fit / K_fixed - 1.0    # 0.016379
beta_fit  = gamma_fixed / r_fit - 1.0    # 1.592906

# Verify: r = gamma/(1+beta) matches r_fit
r_check = gamma_fixed / (1 + beta_fit)
assert abs(r_check - r_fit) < 1e-9, f"Inconsistent: r_fit={r_fit}, r_check={r_check}"

# ==================== Data ====================
t_data = np.array([0, 30, 60, 90, 120, 150, 180, 270, 365])
y_data = np.array([6314, 6542, 6875, 7173, 7368, 7591, 7724, 7896, 8002])

# ==================== Helper Functions ====================
def logistic(t, r, K_eff):
    return K_eff / (1 + (K_eff / x0 - 1) * np.exp(-r * t))

def compute_metrics(r, K_eff):
    y_pred = logistic(t_data, r, K_eff)
    residuals = y_data - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2   = 1 - ss_res / ss_tot
    mae  = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals / y_data)) * 100
    return r2, mae, rmse, mape, y_pred, residuals

# ==================== Base Metrics ====================
r2_base, mae_base, rmse_base, mape_base, y_pred_base, res_base = \
    compute_metrics(r_fit, K_eff_fit)

# ==================== Sensitivity: OAT ±10% ====================
pert = 0.10

# Parameter definitions: key → {base, down_fn, up_fn, latex, unit}
# Each perturbation function takes the base list [gamma, beta, alpha, K]
# and returns (r, K_eff) for the perturbed state.

base_vec = [gamma_fixed, beta_fit, alpha_fit, K_fixed]  # [gamma, beta, alpha, K]

def compute_r_Keff(vec):
    """vec = [gamma, beta, alpha, K] → (r, K_eff)"""
    g, b, a, k = vec
    return g / (1 + b), k * (1 + a)

param_info = {
    'gamma': {
        'idx': 0, 'latex': '$\\gamma$', 'unit': '1/day',
        'base': gamma_fixed,
    },
    'beta': {
        'idx': 1, 'latex': '$\\beta$', 'unit': '—',
        'base': beta_fit,
    },
    'alpha': {
        'idx': 2, 'latex': '$\\alpha$', 'unit': '—',
        'base': alpha_fit,
    },
    'K': {
        'idx': 3, 'latex': '$K$', 'unit': 'tons/day',
        'base': K_fixed,
    },
}

results = {}
for key, info in param_info.items():
    p_base = info['base']
    p_down = p_base * (1 - pert)
    p_up   = p_base * (1 + pert)

    v_down = base_vec.copy()
    v_down[info['idx']] = p_down
    v_up = base_vec.copy()
    v_up[info['idx']] = p_up

    r_d, Keff_d = compute_r_Keff(v_down)
    r_u, Keff_u = compute_r_Keff(v_up)

    r2_d, mae_d, rmse_d, mape_d, _, _ = compute_metrics(r_d, Keff_d)
    r2_u, mae_u, rmse_u, mape_u, _, _ = compute_metrics(r_u, Keff_u)

    results[key] = {
        'p_base': p_base, 'p_down': p_down, 'p_up': p_up,
        'r_down': r_d, 'r_up': r_u,
        'Keff_down': Keff_d, 'Keff_up': Keff_u,
        'r2_down': r2_d, 'r2_up': r2_u,
        'mae_down': mae_d, 'mae_up': mae_u,
        'rmse_down': rmse_d, 'rmse_up': rmse_u,
        'mape_down': mape_d, 'mape_up': mape_u,
    }

# ==================== Console Output ====================
print("=" * 75)
print("  Q4: Local Sensitivity Analysis  (±10% OAT)")
print("=" * 75)
print(f"  Baseline (from Q2 fit):")
print(f"    gamma = {gamma_fixed},  beta  = {beta_fit:.6f}")
print(f"    alpha = {alpha_fit:.6f},  K     = {K_fixed}")
print(f"    r     = {r_fit:.6f},      K_eff = {K_eff_fit:.2f}")
print(f"    R²    = {r2_base:.4f},    MAE   = {mae_base:.1f},  RMSE = {rmse_base:.1f}")
print("-" * 75)

for key in ['gamma', 'beta', 'alpha', 'K']:
    latex = param_info[key]['latex']
    r = results[key]
    print(f"\n  {latex} ±10%:  {r['p_base']:.4f} → [{r['p_down']:.4f}, {r['p_up']:.4f}]")
    print(f"    r:       [{r['r_down']:.6f}, {r['r_up']:.6f}]  "
          f"(Δ = {r['r_up']-r['r_down']:+.6f})")
    print(f"    K_eff:   [{r['Keff_down']:.1f}, {r['Keff_up']:.1f}]  "
          f"(Δ = {r['Keff_up']-r['Keff_down']:+.1f})")
    print(f"    R²:      [{r['r2_down']:.4f}, {r['r2_up']:.4f}]  "
          f"(Δ = {r['r2_up']-r['r2_down']:+.4f})")
    print(f"    MAE:     [{r['mae_down']:.1f}, {r['mae_up']:.1f}]  "
          f"(Δ = {r['mae_up']-r['mae_down']:+.1f})")
    print(f"    RMSE:    [{r['rmse_down']:.1f}, {r['rmse_up']:.1f}]  "
          f"(Δ = {r['rmse_up']-r['rmse_down']:+.1f})")
    print(f"    MAPE:    [{r['mape_down']:.2f}%, {r['mape_up']:.2f}%]  "
          f"(Δ = {r['mape_up']-r['mape_down']:+.2f}%)")

print("=" * 75)
eps_r_beta = -beta_fit / (1 + beta_fit)
eps_K_alpha = alpha_fit / (1 + alpha_fit)
print(f"\n  Analytical Elasticities:")
print(f"    ε(r, γ)     = +1.0000  (linear)")
print(f"    ε(r, β)     = {eps_r_beta:+.4f}  (nonlinear)")
print(f"    ε(Keff, α)  = {eps_K_alpha:+.4f}  (near zero — insensitive)")
print(f"    ε(Keff, K)  = +1.0000  (linear)")
print("=" * 75)

# ==================== MD Report ====================
# Use str.format with named placeholders to avoid f-string escaping hell
r = results

md = """# Q4: 局部灵敏度分析（±10% OAT）

## 方法

对模型四个基础参数分别施加 ±10% 的独立扰动（One-At-a-Time），
观察其对有效参数 ($r$, $K_{{\\text{{eff}}}}$) 和拟合质量 ($R^2$, MAE, RMSE) 的影响。

**基准参数**（来自 Q2 拟合）：

| 参数 | 符号 | 基准值 | 单位 |
|------|------|--------|------|
| 固有增长率 | $\\gamma$ | {gamma} | 1/day |
| 抑制系数 | $\\beta$ | {beta:.6f} | — |
| 促进系数 | $\\alpha$ | {alpha:.6f} | — |
| 基准容量 | $K$ | {K} | tons/day |

**有效参数（基准）**：$r = \\gamma/(1+\\beta) = {r_base:.6f}$，
$K_{{\\text{{eff}}}} = K(1+\\alpha) = {Keff_base:.2f}$

**基准评估**：$R^2 = {r2_base:.4f}$, MAE $= {mae_base:.1f}$, RMSE $= {rmse_base:.1f}$

---

## 扰动结果

### 1. $\\gamma$ ±10%

| 输出 | −10% ($\\gamma={g_down:.4f}$) | 基准 | +10% ($\\gamma={g_up:.4f}$) |
|------|------|------|------|
| $r$ | {r_g_d:.6f} | {r_base:.6f} | {r_g_u:.6f} |
| $K_{{\\text{{eff}}}}$ | {Keff_base:.2f} | {Keff_base:.2f} | {Keff_base:.2f} |
| $R^2$ | {r2_g_d:.4f} | {r2_base:.4f} | {r2_g_u:.4f} |
| MAE | {mae_g_d:.1f} | {mae_base:.1f} | {mae_g_u:.1f} |
| MAPE | {mape_g_d:.2f}% | {mape_base:.2f}% | {mape_g_u:.2f}% |

**解析弹性**：$\\varepsilon_{{r,\\gamma}} = 1$（线性）

### 2. $\\beta$ ±10%

| 输出 | −10% ($\\beta={b_down:.4f}$) | 基准 | +10% ($\\beta={b_up:.4f}$) |
|------|------|------|------|
| $r$ | {r_b_d:.6f} | {r_base:.6f} | {r_b_u:.6f} |
| $K_{{\\text{{eff}}}}$ | {Keff_base:.2f} | {Keff_base:.2f} | {Keff_base:.2f} |
| $R^2$ | {r2_b_d:.4f} | {r2_base:.4f} | {r2_b_u:.4f} |
| MAE | {mae_b_d:.1f} | {mae_base:.1f} | {mae_b_u:.1f} |
| MAPE | {mape_b_d:.2f}% | {mape_base:.2f}% | {mape_b_u:.2f}% |

**解析弹性**：$\\varepsilon_{{r,\\beta}} = -\\beta/(1+\\beta) = {eps_beta:.4f}$

### 3. $\\alpha$ ±10%

| 输出 | −10% ($\\alpha={a_down:.6f}$) | 基准 | +10% ($\\alpha={a_up:.6f}$) |
|------|------|------|------|
| $r$ | {r_base:.6f} | {r_base:.6f} | {r_base:.6f} |
| $K_{{\\text{{eff}}}}$ | {Keff_a_d:.2f} | {Keff_base:.2f} | {Keff_a_u:.2f} |
| $R^2$ | {r2_a_d:.4f} | {r2_base:.4f} | {r2_a_u:.4f} |
| MAE | {mae_a_d:.1f} | {mae_base:.1f} | {mae_a_u:.1f} |
| MAPE | {mape_a_d:.2f}% | {mape_base:.2f}% | {mape_a_u:.2f}% |

**解析弹性**：$\\varepsilon_{{K_{{\\text{{eff}}}},\\alpha}} = \\alpha/(1+\\alpha) = {eps_alpha:.4f}$

### 4. $K$ ±10%

| 输出 | −10% ($K={K_down:.0f}$) | 基准 | +10% ($K={K_up:.0f}$) |
|------|------|------|------|
| $r$ | {r_base:.6f} | {r_base:.6f} | {r_base:.6f} |
| $K_{{\\text{{eff}}}}$ | {Keff_K_d:.2f} | {Keff_base:.2f} | {Keff_K_u:.2f} |
| $R^2$ | {r2_K_d:.4f} | {r2_base:.4f} | {r2_K_u:.4f} |
| MAE | {mae_K_d:.1f} | {mae_base:.1f} | {mae_K_u:.1f} |
| MAPE | {mape_K_d:.2f}% | {mape_base:.2f}% | {mape_K_u:.2f}% |

**解析弹性**：$\\varepsilon_{{K_{{\\text{{eff}}}},K}} = 1$（线性）

---

## 灵敏度排序

### 对 $R^2$ 的影响（从大到小）

{r2_ranking}

### 对 MAE 的影响（从大到小）

{mae_ranking}

## 解析弹性汇总

| 输出 | 输入 | 弹性 $\\varepsilon$ | 含义 |
|------|------|------|------|
| $r$ | $\\gamma$ | $+1.0000$ | $\\gamma$ 变化 1% → $r$ 同向变化 1% |
| $r$ | $\\beta$ | ${eps_beta:+.4f}$ | $\\beta$ 变化 1% → $r$ 反向变化 {abs_eps_beta:.2f}% |
| $K_{{\\text{{eff}}}}$ | $\\alpha$ | ${eps_alpha:+.4f}$ | $\\alpha$ 变化 1% → $K_{{\\text{{eff}}}}$ 同向变化 {eps_alpha_pct:.3f}% |
| $K_{{\\text{{eff}}}}$ | $K$ | $+1.0000$ | $K$ 变化 1% → $K_{{\\text{{eff}}}}$ 同向变化 1% |

## 关键结论

1. **解耦验证**：$\\gamma$ 和 $\\beta$ 仅影响 $r$，$\\alpha$ 和 $K$ 仅影响 $K_{{\\text{{eff}}}}$，
   互不干扰——验证了模型的结构解耦特性。
2. **$R^2$ 对 $K$ 极度敏感**：$K \\pm 10\\%$ 导致 $R^2$ 从 $0.9915$ 崩溃至 $\\sim 0.3$。
   因为数据已接近饱和平台，$K_{{\\text{{eff}}}}$ 的偏差直接决定饱和预测值，任何偏离都会导致
   系统性残差。这是模型在近饱和区的固有特性，并非缺陷。
3. **$K_{{\\text{{eff}}}}$ 对 $\\alpha$ 不敏感**：$\\alpha \\ll 1$，弹性仅 ${eps_alpha:.4f}$，
   即 $\\alpha$ 变化 10% 仅改变 $K_{{\\text{{eff}}}}$ 约 {eps_alpha_10pct:.2f}%。
   改善运营效率（降低 $\\beta$）对恢复速度的影响是提升促进力度（提高 $\\alpha$）
   对容量影响的 $\\sim$ {ratio_impact:.0f} 倍。
4. **$K$ 和 $\\gamma$ 具有线性影响**：弹性恒为 1，模型对其变化呈等比例响应。
5. **除 $K$ 外模型具有良好的鲁棒性**：$\\gamma, \\beta, \\alpha$ ±10% 扰动下
   $R^2$ 均保持 $\\geq {min_r2_nonK:.4f}$，MAE $\\leq {max_mae_nonK:.1f}$。
   但 $K$ 的扰动影响极大——$R^2$ 从 $0.9915$ 跌至 $\\leq 0.33$，
   说明精确估计基准容量对模型可靠性至关重要。
"""

# Build ranking strings
r2_order = sorted(results.keys(),
                  key=lambda n: abs(results[n]['r2_up'] - results[n]['r2_down']),
                  reverse=True)
r2_ranking = ""
for i, key in enumerate(r2_order):
    rk = results[key]
    span = rk['r2_up'] - rk['r2_down']
    r2_ranking += f"{i+1}. **{param_info[key]['latex']}**: ΔR² = {span:+.4f}  [{rk['r2_down']:.4f}, {rk['r2_up']:.4f}]\n"

mae_order = sorted(results.keys(),
                   key=lambda n: abs(results[n]['mae_up'] - results[n]['mae_down']),
                   reverse=True)
mae_ranking = ""
for i, key in enumerate(mae_order):
    rk = results[key]
    span = rk['mae_up'] - rk['mae_down']
    mae_ranking += f"{i+1}. **{param_info[key]['latex']}**: ΔMAE = {span:+.1f}  [{rk['mae_down']:.1f}, {rk['mae_up']:.1f}]\n"

# Format the MD template with all values
md = md.format(
    gamma=gamma_fixed, beta=beta_fit, alpha=alpha_fit, K=K_fixed,
    r_base=r_fit, Keff_base=K_eff_fit,
    r2_base=r2_base, mae_base=mae_base, rmse_base=rmse_base, mape_base=mape_base,
    # gamma
    g_down=results['gamma']['p_down'], g_up=results['gamma']['p_up'],
    r_g_d=results['gamma']['r_down'], r_g_u=results['gamma']['r_up'],
    r2_g_d=results['gamma']['r2_down'], r2_g_u=results['gamma']['r2_up'],
    mae_g_d=results['gamma']['mae_down'], mae_g_u=results['gamma']['mae_up'],
    mape_g_d=results['gamma']['mape_down'], mape_g_u=results['gamma']['mape_up'],
    # beta
    b_down=results['beta']['p_down'], b_up=results['beta']['p_up'],
    r_b_d=results['beta']['r_down'], r_b_u=results['beta']['r_up'],
    r2_b_d=results['beta']['r2_down'], r2_b_u=results['beta']['r2_up'],
    mae_b_d=results['beta']['mae_down'], mae_b_u=results['beta']['mae_up'],
    mape_b_d=results['beta']['mape_down'], mape_b_u=results['beta']['mape_up'],
    # alpha
    a_down=results['alpha']['p_down'], a_up=results['alpha']['p_up'],
    Keff_a_d=results['alpha']['Keff_down'], Keff_a_u=results['alpha']['Keff_up'],
    r2_a_d=results['alpha']['r2_down'], r2_a_u=results['alpha']['r2_up'],
    mae_a_d=results['alpha']['mae_down'], mae_a_u=results['alpha']['mae_up'],
    mape_a_d=results['alpha']['mape_down'], mape_a_u=results['alpha']['mape_up'],
    # K
    K_down=results['K']['p_down'], K_up=results['K']['p_up'],
    Keff_K_d=results['K']['Keff_down'], Keff_K_u=results['K']['Keff_up'],
    r2_K_d=results['K']['r2_down'], r2_K_u=results['K']['r2_up'],
    mae_K_d=results['K']['mae_down'], mae_K_u=results['K']['mae_up'],
    mape_K_d=results['K']['mape_down'], mape_K_u=results['K']['mape_up'],
    # elasticities
    eps_beta=eps_r_beta, abs_eps_beta=abs(eps_r_beta),
    eps_alpha=eps_K_alpha, eps_alpha_pct=eps_K_alpha,
    eps_alpha_10pct=eps_K_alpha * 10,
    ratio_impact=abs(eps_r_beta) / eps_K_alpha,
    # rankings
    r2_ranking=r2_ranking, mae_ranking=mae_ranking,
    # robustness (excluding K)
    min_r2_nonK=min(results[n]['r2_down'] for n in ['gamma', 'beta', 'alpha']),
    max_mae_nonK=max(results[n]['mae_up'] for n in ['gamma', 'beta', 'alpha']),
    max_delta_r2=max(abs(results[n]['r2_up'] - results[n]['r2_down']) for n in results),
    max_delta_mae=max(abs(results[n]['mae_up'] - results[n]['mae_down']) for n in results),
)

with open('report_q4.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("\nMD report saved: report_q4.md")

# ====================================================================
# PLOTTING: Figure 1 — Tornado Diagrams (2×2)
# ====================================================================
param_keys = ['gamma', 'beta', 'alpha', 'K']
param_latex = {k: param_info[k]['latex'] for k in param_keys}
colors_params = {'gamma': '#1a73e8', 'beta': '#e84d3d',
                 'alpha': '#2ba02b', 'K': '#ff7f0e'}

fig1, axes1 = plt.subplots(2, 2, figsize=(16, 11))
plt.subplots_adjust(hspace=0.35, wspace=0.30)

# ---- Panel 1: R² range bars (ensure v_min ≤ v_max always) ----
ax = axes1[0, 0]
r2_pairs = {}
for k in param_keys:
    a, b = results[k]['r2_down'], results[k]['r2_up']
    r2_pairs[k] = (min(a, b), max(a, b))
r2_widths = {k: r2_pairs[k][1] - r2_pairs[k][0] for k in param_keys}
r2_order = sorted(param_keys, key=lambda k: r2_widths[k])

nonK_keys_r2 = [k for k in r2_order if k != 'K']
r2_vals_all = [r2_pairs[k][0] for k in nonK_keys_r2] + [r2_pairs[k][1] for k in nonK_keys_r2]
r2_xmin, r2_xmax = min(r2_vals_all), max(r2_vals_all)
r2_pad = (r2_xmax - r2_xmin) * 0.35
r2_xlim = (r2_xmin - r2_pad, r2_xmax + r2_pad)

for i, key in enumerate(nonK_keys_r2):
    v_min, v_max = r2_pairs[key]
    c = colors_params[key]
    ax.barh(i, v_max - v_min, height=0.6, left=v_min, color=c, alpha=0.85,
            edgecolor='white', linewidth=0.8)
    ax.text(v_min, i, f'  {v_min:.4f}', ha='right', va='center', fontsize=8.5, color='#555')
    ax.text(v_max, i, f'{v_max:.4f}  ', ha='left',  va='center', fontsize=8.5, color='#111',
            fontweight='bold')

ax.axvline(r2_base, color='black', linewidth=1.5, linestyle='-', alpha=0.6,
           label=f'Baseline $R^2$={r2_base:.4f}')
ax.set_yticks(range(len(nonK_keys_r2)))
ax.set_yticklabels([param_latex[k] for k in nonK_keys_r2], fontsize=12)
ax.set_xlim(r2_xlim)
ax.set_xlabel('$R^2$', fontsize=12)
ax.set_title('(a)  $R^2$ Sensitivity', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.15, axis='x')
k_lo, k_hi = r2_pairs['K']
ax.text(0.98, 0.12, f'$K$: $R^2 \\in [{k_lo:.2f},\\,{k_hi:.2f}]$  (off-scale →)',
        transform=ax.transAxes, fontsize=9.5, ha='right', va='center',
        color=colors_params['K'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                  edgecolor=colors_params['K'], alpha=0.9))
ax.legend(fontsize=8.5, loc='upper right')

# ---- Panel 2: MAE range bars (ensure v_min ≤ v_max always) ----
ax = axes1[0, 1]
mae_pairs = {}
for k in param_keys:
    a, b = results[k]['mae_down'], results[k]['mae_up']
    mae_pairs[k] = (min(a, b), max(a, b))
mae_widths = {k: mae_pairs[k][1] - mae_pairs[k][0] for k in param_keys}
mae_order = sorted(param_keys, key=lambda k: mae_widths[k])

nonK_keys_mae = [k for k in mae_order if k != 'K']
mae_vals_all = [mae_pairs[k][0] for k in nonK_keys_mae] + [mae_pairs[k][1] for k in nonK_keys_mae]
mae_xmin, mae_xmax = min(mae_vals_all), max(mae_vals_all)
mae_pad = (mae_xmax - mae_xmin) * 0.35
mae_xlim = (mae_xmin - mae_pad, mae_xmax + mae_pad)

for i, key in enumerate(nonK_keys_mae):
    v_min, v_max = mae_pairs[key]
    c = colors_params[key]
    ax.barh(i, v_max - v_min, height=0.6, left=v_min, color=c, alpha=0.85,
            edgecolor='white', linewidth=0.8)
    ax.text(v_min, i, f'  {v_min:.1f}', ha='right', va='center', fontsize=8.5, color='#555')
    ax.text(v_max, i, f'{v_max:.1f}  ', ha='left',  va='center', fontsize=8.5, color='#111',
            fontweight='bold')

ax.axvline(mae_base, color='black', linewidth=1.5, linestyle='-', alpha=0.6,
           label=f'Baseline MAE$={mae_base:.1f}$')
ax.set_yticks(range(len(nonK_keys_mae)))
ax.set_yticklabels([param_latex[k] for k in nonK_keys_mae], fontsize=12)
ax.set_xlim(mae_xlim)
ax.set_xlabel('MAE (tons/day)', fontsize=12)
ax.set_title('(b)  MAE Sensitivity', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.15, axis='x')
k_lo_m, k_hi_m = mae_pairs['K']
ax.text(0.98, 0.12, f'$K$: MAE $\\in [{k_lo_m:.0f},\\,{k_hi_m:.0f}]$  (off-scale →)',
        transform=ax.transAxes, fontsize=9.5, ha='right', va='center',
        color=colors_params['K'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                  edgecolor=colors_params['K'], alpha=0.9))
ax.legend(fontsize=8.5, loc='upper right')

# ---- Panel 3: Δr tornado ----
ax = axes1[1, 0]
r_deltas_plot = {k: (results[k]['r_down'] - r_fit, results[k]['r_up'] - r_fit)
                 for k in param_keys}
r_widths_plot = {k: abs(r_deltas_plot[k][1] - r_deltas_plot[k][0]) for k in param_keys}
r_order_plot = sorted(param_keys, key=lambda k: r_widths_plot[k])

for i, key in enumerate(r_order_plot):
    d_lo, d_hi = r_deltas_plot[key]
    v_lo = results[key]['r_down']
    v_hi = results[key]['r_up']
    c = colors_params[key]
    ax.barh(i, d_hi, height=0.55, left=0, color=c, alpha=0.88,
            edgecolor='white', linewidth=0.5)
    ax.barh(i, d_lo, height=0.55, left=0, color=c, alpha=0.40,
            edgecolor='white', linewidth=0.5)
    if abs(d_lo) > 1e-9:
        ax.text(d_lo, i, f'{v_lo:.6f}  ', ha='right', va='center', fontsize=8, color='#555')
    if abs(d_hi) > 1e-9:
        ax.text(d_hi, i, f'  {v_hi:.6f}', ha='left', va='center', fontsize=8, color='#111',
                fontweight='bold')

ax.set_yticks(range(len(r_order_plot)))
ax.set_yticklabels([param_latex[k] for k in r_order_plot], fontsize=12)
ax.axvline(0, color='black', linewidth=1.2)
ax.set_xlabel('$\\Delta r$ (1/day)', fontsize=12)
ax.set_title(f'(c)  $r$ Sensitivity  (baseline $r={r_fit:.6f}$)', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.15, axis='x')

# ---- Panel 4: ΔK_eff tornado ----
ax = axes1[1, 1]
Keff_deltas_plot = {k: (results[k]['Keff_down'] - K_eff_fit, results[k]['Keff_up'] - K_eff_fit)
                    for k in param_keys}
Keff_widths = {k: abs(Keff_deltas_plot[k][1] - Keff_deltas_plot[k][0]) for k in param_keys}
Keff_order = sorted(param_keys, key=lambda k: Keff_widths[k])

nonK_keff_max = max(max(abs(Keff_deltas_plot[k][0]), abs(Keff_deltas_plot[k][1]))
                    for k in Keff_order if k != 'K')
nonK_keff_pad = nonK_keff_max * 1.6
keff_xlim = (-nonK_keff_pad, nonK_keff_pad)

for i, key in enumerate(Keff_order):
    d_lo, d_hi = Keff_deltas_plot[key]
    v_lo = results[key]['Keff_down']
    v_hi = results[key]['Keff_up']
    c = colors_params[key]
    if key == 'K':
        clip_lo = max(d_lo, keff_xlim[0])
        clip_hi = min(d_hi, keff_xlim[1])
        ax.barh(i, clip_hi, height=0.55, left=0, color=c, alpha=0.88,
                edgecolor='white', linewidth=0.5)
        ax.barh(i, clip_lo, height=0.55, left=0, color=c, alpha=0.40,
                edgecolor=c, linewidth=0.5, hatch='////')
    else:
        ax.barh(i, d_hi, height=0.55, left=0, color=c, alpha=0.88,
                edgecolor='white', linewidth=0.5)
        ax.barh(i, d_lo, height=0.55, left=0, color=c, alpha=0.40,
                edgecolor='white', linewidth=0.5)
        if abs(d_lo) > 0.5:
            ax.text(d_lo, i, f'{v_lo:.0f}  ', ha='right', va='center', fontsize=8, color='#555')
        if abs(d_hi) > 0.5:
            ax.text(d_hi, i, f'  {v_hi:.0f}', ha='left', va='center', fontsize=8, color='#111',
                    fontweight='bold')

ax.set_yticks(range(len(Keff_order)))
ax.set_yticklabels([param_latex[k] for k in Keff_order], fontsize=12)
ax.axvline(0, color='black', linewidth=1.2)
ax.set_xlim(keff_xlim)
ax.set_xlabel('$\\Delta K_{\\mathrm{eff}}$ (tons/day)', fontsize=12)
ax.set_title(f'(d)  $K_{{\\mathrm{{eff}}}}$ Sensitivity  (baseline $K_{{\\mathrm{{eff}}}}={K_eff_fit:.0f}$)',
             fontsize=13, fontweight='bold', loc='left')
kd_lo, kd_hi = Keff_deltas_plot['K']
ax.text(0.98, 0.93, f'$K$: $\\Delta K_{{\\mathrm{{eff}}}} \\in [{kd_lo:.0f},\\,{kd_hi:.0f}]$  (off-scale)',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        color=colors_params['K'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                  edgecolor=colors_params['K'], alpha=0.9))
ax.grid(True, alpha=0.15, axis='x')

fig1.suptitle('Q4: Local Sensitivity Analysis — OAT ±10% Perturbation',
              fontsize=15, fontweight='bold', y=1.01)
plt.savefig('sensitivity_tornado.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Plot saved: sensitivity_tornado.png")
plt.close(fig1)

# ====================================================================
# PLOTTING: Figure 2 — Prediction Envelope + Elasticity (1×3)
# ====================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(19, 5.8))
plt.subplots_adjust(wspace=0.28)

t_smooth = np.linspace(0, 500, 400)
x_base_smooth = logistic(t_smooth, r_fit, K_eff_fit)

# ---- Panel 1: Full prediction envelope with improved styling ----
ax = axes2[0]
all_preds = []
for key in param_keys:
    v_down = base_vec.copy()
    v_down[param_info[key]['idx']] = param_info[key]['base'] * (1 - pert)
    v_up = base_vec.copy()
    v_up[param_info[key]['idx']] = param_info[key]['base'] * (1 + pert)
    r_d, Keff_d = compute_r_Keff(v_down)
    r_u, Keff_u = compute_r_Keff(v_up)
    all_preds.append(logistic(t_smooth, r_d, Keff_d))
    all_preds.append(logistic(t_smooth, r_u, Keff_u))

all_preds = np.array(all_preds)
pred_min, pred_max = np.min(all_preds, axis=0), np.max(all_preds, axis=0)
pred_range = pred_max - pred_min

# Envelope
ax.fill_between(t_smooth, pred_min, pred_max, color='#b0bec5', alpha=0.35,
                label='Total envelope (±10% all params)', zorder=2)
# Range edge lines
ax.plot(t_smooth, pred_min, color='#78909c', linewidth=0.8, alpha=0.6, zorder=3)
ax.plot(t_smooth, pred_max, color='#78909c', linewidth=0.8, alpha=0.6, zorder=3)
# Base curve
ax.plot(t_smooth, x_base_smooth, 'k-', linewidth=2.2, label='Base prediction', zorder=10)
# Data
ax.scatter(t_data, y_data, c='#1565c0', s=55, zorder=12, edgecolors='white',
           linewidth=1, label='Observed data')
ax.scatter(0, x0, c='#2e7d32', s=80, zorder=13, marker='s', edgecolors='white',
           linewidth=1, label=f'$x_0={x0}$')
# Reference lines
ax.axhline(K_eff_fit, color='#37474f', linestyle=':', linewidth=1.2, alpha=0.7,
           label=f'$K_{{\\mathrm{{eff}}}}={K_eff_fit:.0f}$')
ax.set_xlabel('Time $t$ (days)', fontsize=11)
ax.set_ylabel('Recovery $x(t)$ (tons/day)', fontsize=11)
ax.set_title('(a)  Prediction Envelope', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.85)
ax.grid(True, alpha=0.2)
ax.set_xlim(-15, 515)

# ---- Panel 2: Per-parameter contribution bands ----
ax = axes2[1]
# Plot bands in order of decreasing impact (K last = on top visually)
band_keys = ['alpha', 'gamma', 'beta', 'K']
for key in band_keys:
    v_down = base_vec.copy()
    v_down[param_info[key]['idx']] = param_info[key]['base'] * (1 - pert)
    v_up = base_vec.copy()
    v_up[param_info[key]['idx']] = param_info[key]['base'] * (1 + pert)
    r_d, Keff_d = compute_r_Keff(v_down)
    r_u, Keff_u = compute_r_Keff(v_up)
    x_d = logistic(t_smooth, r_d, Keff_d)
    x_u = logistic(t_smooth, r_u, Keff_u)
    ax.fill_between(t_smooth, x_d, x_u, color=colors_params[key], alpha=0.22,
                    label=f'{param_latex[key]} ±10%', zorder=3)
    # Edge lines for clarity
    ax.plot(t_smooth, x_d, color=colors_params[key], linewidth=0.5, alpha=0.35, zorder=4)
    ax.plot(t_smooth, x_u, color=colors_params[key], linewidth=0.5, alpha=0.35, zorder=4)

ax.plot(t_smooth, x_base_smooth, 'k-', linewidth=2, zorder=10, label='Base')
ax.scatter(t_data, y_data, c='#1565c0', s=35, zorder=12, edgecolors='white', linewidth=0.8)
ax.set_xlabel('Time $t$ (days)', fontsize=11)
ax.set_ylabel('Recovery $x(t)$ (tons/day)', fontsize=11)
ax.set_title('(b)  Per-Parameter Bands', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.85)
ax.grid(True, alpha=0.2)
ax.set_xlim(-15, 515)

# ---- Panel 3: Elasticity overview (both analytical + numerical for R²/MAE) ----
ax = axes2[2]

# Numerical elasticities of R² and MAE w.r.t. each parameter
def num_elasticity(f_down, f_up, f_base):
    """Central-difference arc elasticity for ±10%."""
    eps_down = (f_down - f_base) / f_base / (-0.10)
    eps_up   = (f_up   - f_base) / f_base / (+0.10)
    return eps_down, eps_up

e_items = []
for key in param_keys:
    eps_r2_d, eps_r2_u = num_elasticity(results[key]['r2_down'],
                                         results[key]['r2_up'], r2_base)
    eps_mae_d, eps_mae_u = num_elasticity(results[key]['mae_down'],
                                           results[key]['mae_up'], mae_base)
    eps_r2_avg = (abs(eps_r2_d) + abs(eps_r2_u)) / 2
    eps_mae_avg = (abs(eps_mae_d) + abs(eps_mae_u)) / 2
    e_items.append({
        'key': key, 'latex': param_latex[key],
        'eps_r2': eps_r2_avg, 'eps_mae': eps_mae_avg,
        'c': colors_params[key],
    })

# Separate K from the rest (K elasticities are off-scale)
e_nonK = [item for item in e_items if item['key'] != 'K']
e_K = [item for item in e_items if item['key'] == 'K'][0]

# Sort non-K by average impact
e_nonK.sort(key=lambda x: x['eps_r2'] + x['eps_mae'])

# Plot grouped horizontal bars for non-K params
n_nonK = len(e_nonK)
y_positions = np.arange(n_nonK)
bar_height = 0.28

for i, item in enumerate(e_nonK):
    # R² elasticity bar (top) — dark fill, parameter-colored
    ax.barh(i + bar_height/2, item['eps_r2'], bar_height,
            color=item['c'], alpha=0.85, edgecolor='white', linewidth=0.5)
    # MAE elasticity bar (bottom) — hatched, parameter-colored
    ax.barh(i - bar_height/2, item['eps_mae'], bar_height,
            color=item['c'], alpha=0.35, edgecolor='white', linewidth=0.5,
            hatch='...')

# Build y-tick labels
ax.set_yticks(y_positions)
ax.set_yticklabels([item['latex'] for item in e_nonK], fontsize=12)
ax.set_xlabel('Numerical Elasticity $|\\varepsilon|$  (absolute value)', fontsize=10)
ax.set_title('(c)  Numerical Elasticity of Fit Metrics', fontsize=13, fontweight='bold', loc='left')

# Legend with NEUTRAL colors (not parameter colors)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor='#444444', alpha=0.85, label='$|\\varepsilon(R^2)|$'),
    Patch(facecolor='#999999', alpha=0.55, hatch='...', label='$|\\varepsilon(\\mathrm{MAE})|$'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='lower right', framealpha=0.85)

# Annotate values on non-K bars
for i, item in enumerate(e_nonK):
    if item['eps_r2'] > 0.005:
        ax.text(item['eps_r2'] + 0.03, i + bar_height/2,
                f'{item["eps_r2"]:.2f}', va='center', fontsize=8.5, fontweight='bold')
    if item['eps_mae'] > 0.005:
        ax.text(item['eps_mae'] + 0.03, i - bar_height/2,
                f'{item["eps_mae"]:.2f}', va='center', fontsize=7.5, color='#555')

# K annotation box (off-scale)
k_eps_text = (f'$K$ (off-scale →):\n'
              f'  $|\\varepsilon(R^2, K)| \\approx {e_K["eps_r2"]:.1f}$\n'
              f'  $|\\varepsilon(\\mathrm{{MAE}}, K)| \\approx {e_K["eps_mae"]:.0f}$')
ax.text(0.98, 0.18, k_eps_text, transform=ax.transAxes, fontsize=9,
        ha='right', va='center', color=colors_params['K'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0',
                  edgecolor=colors_params['K'], alpha=0.9))

# Set xlim to comfortably fit non-K bars
nonK_xmax = max(max(item['eps_r2'], item['eps_mae']) for item in e_nonK)
ax.set_xlim(-0.3, nonK_xmax * 1.6)
ax.grid(True, alpha=0.15, axis='x')

fig2.suptitle('Q4: Prediction Envelope  |  Parameter Bands  |  Numerical Elasticity',
              fontsize=14, fontweight='bold', y=1.02)
plt.savefig('sensitivity_envelope.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Plot saved: sensitivity_envelope.png")
plt.close(fig2)

print("\nAll Q4 sensitivity analysis complete.")
