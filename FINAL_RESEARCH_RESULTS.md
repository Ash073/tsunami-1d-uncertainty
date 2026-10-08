# Final Research Results — 1D Tsunami Uncertainty Study

## Main result

Across Monte Carlo uncertainty propagation, correlation/regression analysis, Morris screening, and replicated Sobol total-order analysis, the same sensitivity hierarchy was obtained:

**slope-end position > shallow-water depth > deep-water depth**

## Deterministic baseline

- Deep-water depth: 200 m
- Shallow-water depth: 50 m
- Slope end: 80 km
- 40 km amplitude: 0.98726 m
- 80 km amplitude: 1.34080 m
- Amplification: **1.35811×**

## Multivariate Monte Carlo (100 realizations)

- Mean amplification: **1.33455×**
- Standard deviation: **0.03266×**
- Minimum sampled amplification: **1.26544×**
- Maximum sampled amplification: **1.38875×**
- 5th percentile: **1.27592×**
- Median: **1.33681×**
- 95th percentile: **1.38127×**

## Exploratory sensitivity

Pearson correlations:

| Parameter | Pearson |
|---|---:|
| Deep depth | +0.29182 |
| Shallow depth | −0.53299 |
| Slope end | −0.66997 |

Spearman correlations:

| Parameter | Spearman |
|---|---:|
| Deep depth | +0.31034 |
| Shallow depth | −0.57026 |
| Slope end | −0.61089 |

Standardized regression coefficients:

| Parameter | Coefficient |
|---|---:|
| Deep depth | +0.19399 |
| Shallow depth | −0.49486 |
| Slope end | −0.65218 |

Linear-model R²: **0.75730**.

## Morris sensitivity

The 64-trajectory Morris experiment used 256 model evaluations.

| Parameter | μ* | σ |
|---|---:|---:|
| Slope end | **0.08702** | **0.06254** |
| Shallow depth | 0.05060 | 0.00897 |
| Deep depth | 0.02561 | 0.00479 |

The same ranking was obtained in all three Morris measures.

## Replicated Sobol sensitivity

Mean total-order sensitivities from the replicated experiment:

| Parameter | Mean ST |
|---|---:|
| Slope end | **0.72263** |
| Shallow depth | **0.22153** |
| Deep depth | **0.06146** |

These total-order results provide the most stable global-sensitivity ranking from the current experiments.

**Caution:** the first-order Sobol estimates were still noisy in the replicated calculation, so their exact numerical values should not be presented as converged final estimates. The total-order ranking is the result used here.

## Interpretation

For the synthetic uncertainty ranges used in this study, the position where the bathymetric slope reaches the shallow-water plateau is the dominant uncertain parameter controlling predicted wave amplification at the 80 km gauge. Shallow-water depth is the second most influential parameter, while deep-water depth has a smaller influence.

## Limitations

- The uncertainty ranges are synthetic development assumptions, not measured bathymetric error bounds.
- The model is 1D and does not represent directional spreading or full 2D coastal geometry.
- The amplification metric is defined from incident wave peaks at fixed gauges.
- Green's-law comparisons are theoretical reference checks, not exact validation of the nonlinear numerical solution.
- The solver remains a research prototype and should be benchmarked against additional established cases before publication-level claims.
- Sensitivity results depend on the chosen parameterization, uncertainty ranges, source, numerical resolution, and output metric.

## Resume-ready project description

> Developed a 1D nonlinear shallow-water tsunami simulator with MUSCL finite-volume discretization, hydrostatic reconstruction, and uncertainty propagation over variable bathymetry. Performed Monte Carlo UQ and global sensitivity analysis using Morris screening and replicated Sobol total-order estimates, identifying bathymetric slope-end position as the dominant uncertainty source for modeled wave amplification.
