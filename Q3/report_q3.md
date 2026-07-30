# Q3 Stability Analysis: Symmetric Antagonistic Model

## Model (from Q2)

$$\frac{dx}{dt} = \frac{\gamma}{1+\beta} \cdot x \cdot \left(1 - \frac{x}{K(1+\alpha)}\right)
= r \cdot x \cdot \left(1 - \frac{x}{K_{\text{eff}}}\right)$$

$$r = \frac{\gamma}{1+\beta}, \qquad K_{\text{eff}} = K(1+\alpha)$$

**By definition, $\alpha \geq 0$ (promotion) and $\beta \geq 0$ (inhibition).**

## Equilibrium Analysis

Setting $dx/dt = 0$:

| Equilibrium | Value | Eigenvalue | Stability |
|-------------|-------|-----------|-----------|
| $x_1^*$ (zero) | $0$ | $\lambda_1 = r = \gamma/(1+\beta)$ | **Unstable** ($r > 0$ always) |
| $x_2^*$ (sustainable) | $K_{\text{eff}} = K(1+\alpha)$ | $\lambda_2 = -r$ | **Asymptotically Stable** |

**Proof**: $\beta \geq 0 \Rightarrow 1+\beta \geq 1 \Rightarrow r = \gamma/(1+\beta) \in (0, \gamma]$.
Therefore $r > 0$ for all valid $\beta$, and $x_2^*$ is always stable.

## Structural Stability

Because $\alpha, \beta \geq 0$ **by definition** (they are promotional and inhibitory coefficients),
the entire valid parameter space satisfies $r > 0$. There is **no bifurcation anywhere**
in the design region. The recycling system is **structurally stable**:

> Any positive initial recovery volume $x(0) > 0$ will converge to
> $K_{\text{eff}}$ as $t \to \infty$, for **all** $\alpha \geq 0$, $\beta \geq 0$.

This is a strong result: "沪尚回收" will not experience mathematical collapse
(断崖下跌) as long as the promotion and inhibition coefficients remain non-negative
— which they do by construction. The operational risk is **performance degradation**,
not structural instability.

## Practical Safety Boundaries

While the system is mathematically stable everywhere, operational viability
requires **adequate performance**:

### Growth Rate Constraint

$$r = \frac{\gamma}{1+\beta} \geq r_{\min} \quad \Rightarrow \quad
\beta \leq \frac{\gamma}{r_{\min}} - 1$$

With $r_{\min} = 0.005$ (minimum acceptable growth rate):
$$\beta_{\max} = \frac{0.022}{0.005} - 1 = 3.40$$

### Capacity Constraint

$$K_{\text{eff}} = K(1+\alpha) \geq K_{\min} \quad \Rightarrow \quad
\alpha \geq \frac{K_{\min}}{K} - 1$$

With $K_{\min} = K$ (baseline capacity): $\alpha \geq 0$ — always satisfied.

### Safe Operating Ranges

| Parameter | Lower Bound | Upper Bound | Current | Status |
|-----------|-------------|-------------|---------|--------|
| $\alpha$ | $0$ (definition) | — | $0.0164$ | Safe (capacity at $101.6%$ of $K$) |
| $\beta$ | $0$ (definition) | $3.40$ (from $r_{\min}$) | $1.59$ | Safe (r at $38.6%$ of $\gamma$) |

### Safety Margins

| Metric | Value |
|--------|-------|
| $\beta$ margin to $\beta_{\max}$ | $1.81$ |
| $r$ margin above $r_{\min}$ | $0.0035$ |
| $r/\gamma$ (effective rate fraction) | $38.6\%$ |

## Decoupled Regulation

A key advantage of this model: $\alpha$ and $\beta$ are **fully decoupled**.

- **$\alpha$ only affects $K_{\text{eff}}$**: adjusting promotion changes
  the saturation ceiling without affecting convergence speed
- **$\beta$ only affects $r$**: controlling inhibition changes the convergence
  speed without affecting the saturation ceiling

This enables **independent tuning**: policymakers can raise $\alpha$ to expand
total capacity while separately managing $\beta$ to maintain adequate growth speed.

## Key Findings

1. **Structurally stable**: $r = \gamma/(1+\beta) > 0$ for all $\alpha,\beta \geq 0$ — no bifurcation exists.
2. **Collapse impossible by design**: as long as promotional and inhibitory effects remain positive by definition, $x(t) \to K_{\text{eff}} > 0$ always.
3. **Risk is performance, not stability**: the danger lies in $\beta$ growing large enough to make $r$ impractically small, or $\alpha$ shrinking to near zero.
4. **Decoupled control**: $\alpha \to K_{\text{eff}}$, $\beta \to r$ — independently adjustable.
5. **Safety boundary**: $\beta \leq \gamma/r_{\min} - 1 = 3.40$ for operational viability with $r_{\min} = 0.005$.
