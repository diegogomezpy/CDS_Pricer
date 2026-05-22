# Synthetic CDO Tranche Pricer

A Python tool for pricing multi-tranche synthetic CDOs on a loan portfolio. Built as an independent project to explore structured credit products — specifically how loss is allocated across a capital structure and how tranche premiums are determined.

## What it does

Given a portfolio modeled as a normally distributed cashflow, the pricer:

- Constructs a sequential waterfall of N tranches with configurable attachment and detachment points
- Computes the fair fixed premium for each tranche
- Derives per-tranche analytics: credit event probability, full loss probability, positive return probability, expected loss, and return on premium
- Computes equity tranche residual cashflow
- Visualizes the portfolio cashflow distribution with tranche overlays

## Pricing approach

Tranches are priced using **actuarial cost-plus pricing under the physical measure P**:

```
fixed_payment = E^P[exp(-rT) × payoff(x)] + expected_return × notional
```

where `payoff(x)` is the standard tranche payoff over the attachment-detachment interval and `x ~ N(mean, std_dev)` is the portfolio cashflow. Expected losses are computed via numerical integration (`scipy.quad`) over the payoff function. The risk premium (`expected_return`) is an exogenous input supplied by the user.

This is **not** risk-neutral pricing. Under a risk-neutral framework the premium would be `E^Q[exp(-rT) × payoff(x)]` with no separate risk loading, and the risk premium would be embedded in the P → Q change of measure, calibrated to observed market tranche spreads. That extension would require fitting the Q-measure distribution to market prices.

## Known limitations

- **Gaussian cashflow distribution** — underestimates tail risk; real portfolio losses have fat tails
- **No inter-obligor correlation** — a Gaussian copula model would be more realistic
- **Flat term structure** — single risk-free rate, no yield curve
- **Single-period model** — one cashflow realisation at maturity, no intermediate stochastic payments
- **Exogenous risk premium** — not derived from no-arbitrage arguments or market calibration

## Files

| File | Description |
|------|-------------|
| `cds_pricing_prototype.py` | Core module: `CreditDefaultSwap`, `Loan`, `SyntheticCDO` classes |
| `demo.ipynb` | Interactive Jupyter/Colab notebook with widget UI and visualisation |
| `requirements.txt` | Dependencies |

## Installation

```bash
git clone https://github.com/diegogomezpy/cds-tranche-pricer
cd cds-tranche-pricer
pip install -r requirements.txt
```

## Usage

### Interactive (recommended)

Open `demo.ipynb` in Jupyter or Colab, run all cells, and use the widgets to configure tranches and portfolio parameters.

**In Colab**, add this at the top before running:
```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/your-folder-name')
```

### Programmatic

```python
from cds_pricing_prototype import SyntheticCDO

cdo = SyntheticCDO(
    num_tranches=3,
    tranche_sizes=[30, 40, 30],       # must sum to notional
    expected_returns=[0.08, 0.05, 0.03],  # equity, mezz, senior
    mean=120,
    std_dev=30,
    notional=100,
    risk_free_rate=0.04,
    maturity=1,
    payment_frequency=4
)

cdo.summarize()
cdo.summarize_equity()
```

**Example output:**
```
=== Synthetic CDO Summary ===

Tranche 1: 0.00-30.00
  Target return:                  8.00%
  Fixed payment (t=0):            4.97
  Credit event probability:       0.09%
  Probability of positive return: 99.91%
  Return on premium:              48.30%
  Expected loss ($):              2.57
  Expected loss (%):              8.57%
  Full loss probability:          0.00%
  Benchmark loan return:          4.00%
...
```

## Per-tranche analytics

| Metric | Description |
|--------|-------------|
| Fixed payment | Upfront premium paid by protection buyer at t=0 |
| Credit event probability | P(portfolio cashflow < detachment) — any loss in tranche |
| Full loss probability | P(portfolio cashflow < attachment) — complete wipeout |
| Positive return probability | P(payout < FV of premium) — protection seller earns positive net return |
| Expected loss | Discounted expected payout under P |
| Return on premium | Net profit / fixed payment — fraction of premium that is profit vs. loss coverage |

## Dependencies

```
numpy
scipy
matplotlib
ipywidgets
pandas
```
