"""
Synthetic CDO Tranche Pricer
============================
Three classes implementing actuarial cost-plus pricing for synthetic CDO tranches:

    CreditDefaultSwap  — prices a single tranche
    Loan               — benchmark loan with simple interest
    SyntheticCDO       — builds a full waterfall of N tranches

Pricing approach
----------------
Tranches are priced under the physical measure P, NOT under a risk-neutral measure Q.
The fixed premium is:

    fixed_payment = E^P[exp(-rT) * payoff(x)] + expected_return * notional

where payoff(x) is the standard tranche payoff over the attachment-detachment interval,
x ~ N(mean, std_dev) is the portfolio cashflow, and expected_return is an exogenous
risk loading supplied by the user.

Under true risk-neutral pricing the premium would be E^Q[exp(-rT) * payoff(x)] with
no separate risk loading, and the risk premium would be embedded in the P -> Q change
of measure (calibrated to observed market spreads). That extension would require
fitting the Q-measure distribution to market tranche prices.

Known limitations
-----------------
- Gaussian cashflow distribution underestimates tail risk
- No inter-obligor correlation structure (a Gaussian copula model would be more realistic)
- Flat term structure; single risk-free rate
- Single-period model: one cashflow realisation at maturity
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


class CreditDefaultSwap:
    def __init__(self, attachment, detachment, mean, std_dev,
                 notional, risk_free_rate, maturity, expected_return):
        self.attachment = attachment
        self.detachment = detachment
        self.mean = mean
        self.std_dev = std_dev
        self.notional = notional
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        self.expected_return = expected_return

        # Computed attributes (populated by compute_fixed_payment)
        self.fixed_payment = None
        self.credit_event_probability = None
        self.positive_return_probability = None
        self.expected_loss = None
        self.expected_loss_pct = None
        self.expected_actual_return = None
        self.full_loss_probability = None

    def compute_fixed_payment(self):
        df = np.exp(-self.risk_free_rate * self.maturity)
        norm_dist = norm(loc=self.mean, scale=self.std_dev)

        def integrand(x):
            payoff = min(max(self.detachment - x, 0),
                         self.detachment - self.attachment)
            return payoff * norm_dist.pdf(x)

        expected_discounted_payout, _ = quad(integrand, -np.inf, np.inf)
        expected_discounted_payout *= df

        self.expected_loss = expected_discounted_payout
        self.expected_loss_pct = expected_discounted_payout / self.notional
        self.fixed_payment = expected_discounted_payout + self.expected_return * self.notional

        # Return on premium: net profit as fraction of total fixed payment received.
        # Varies meaningfully across tranches — senior tranches have low expected losses
        # so most of the premium is profit; equity tranches have high expected losses
        # so little is.
        net_profit = self.expected_return * self.notional
        self.expected_actual_return = net_profit / self.fixed_payment

        self.credit_event_probability = norm_dist.cdf(self.detachment)
        self.full_loss_probability = norm_dist.cdf(self.attachment)
        self._compute_positive_return_probability(norm_dist)

    def _compute_positive_return_probability(self, norm_dist):
        """P(payoff(x) < FV(fixed_payment)), i.e. protection seller earns positive net return."""
        df = np.exp(-self.risk_free_rate * self.maturity)
        x_star = self.detachment - (self.fixed_payment / df)
        x_star = np.clip(x_star, self.attachment, self.detachment)
        self.positive_return_probability = 1.0 - norm_dist.cdf(x_star)


class Loan:
    """
    Benchmark loan with simple interest.
    Principal repaid in full at maturity; interest accrues linearly.
    """
    def __init__(self, principal, interest_rate, maturity):
        self.principal = principal
        self.interest_rate = interest_rate
        self.maturity = maturity
        self.total_return = None
        self.yield_to_maturity = None

    def compute_return(self):
        self.total_return = self.principal * self.interest_rate * self.maturity
        self.yield_to_maturity = self.interest_rate


class SyntheticCDO:
    def __init__(self, num_tranches, tranche_sizes, expected_returns, mean, std_dev,
                 notional, risk_free_rate, maturity, payment_frequency):
        if len(tranche_sizes) != num_tranches:
            raise ValueError("Length of tranche_sizes must match num_tranches")
        if len(expected_returns) != num_tranches:
            raise ValueError("Length of expected_returns must match num_tranches")
        if not np.isclose(sum(tranche_sizes), notional):
            print(f"Warning: Sum of tranche sizes ({sum(tranche_sizes)}) != total notional ({notional})")

        self.tranches = []
        self.loans = []
        self.payment_frequency = payment_frequency
        self.total_notional = notional
        self.mean = mean
        self.std_dev = std_dev

        attach = 0.0
        for i in range(num_tranches):
            detach = attach + tranche_sizes[i]
            tranche_notional = tranche_sizes[i]

            cds = CreditDefaultSwap(
                attachment=attach,
                detachment=detach,
                mean=mean,
                std_dev=std_dev,
                notional=tranche_notional,
                risk_free_rate=risk_free_rate,
                maturity=maturity,
                expected_return=expected_returns[i]
            )
            cds.compute_fixed_payment()

            loan = Loan(principal=tranche_notional, interest_rate=risk_free_rate, maturity=maturity)
            loan.compute_return()

            self.tranches.append(cds)
            self.loans.append(loan)
            attach = detach

    def summarize(self, return_as_df=False):
        import pandas as pd
        print("=== Synthetic CDO Summary ===\n")
        results = []
        weighted_annuity_sum = 0.0
        total_tranche_notional = sum(cds.notional for cds in self.tranches)

        for i, cds in enumerate(self.tranches):
            annuity = cds.fixed_payment / (cds.maturity * self.payment_frequency)
            annualized_pct = annuity * self.payment_frequency / cds.notional
            weight = cds.notional / total_tranche_notional
            weighted_annuity_sum += weight * annualized_pct
            loan = self.loans[i]

            result = {
                "Tranche": f"{i + 1} ({cds.attachment:.2f}-{cds.detachment:.2f})",
                "Target Return %": cds.expected_return * 100,
                "Fixed Payment": cds.fixed_payment,
                "Annuity Per Period": annuity,
                "Annualized % of Notional": annualized_pct * 100,
                "Credit Event Probability %": cds.credit_event_probability * 100,
                "Positive Return Probability %": cds.positive_return_probability * 100,
                "Return on Premium %": cds.expected_actual_return * 100,
                "Expected Loss ($)": cds.expected_loss,
                "Expected Loss (%)": cds.expected_loss_pct * 100,
                "Full Loss Probability %": cds.full_loss_probability * 100,
                "Benchmark Loan Return %": loan.yield_to_maturity * 100
            }
            results.append(result)

            print(f"Tranche {i+1}: {cds.attachment:.2f}-{cds.detachment:.2f}")
            print(f"  Target return:                  {cds.expected_return * 100:.2f}%")
            print(f"  Fixed payment (t=0):            {cds.fixed_payment:.2f}")
            print(f"  Annuity per period:             {annuity:.2f}")
            print(f"  Annualized % of notional:       {annualized_pct * 100:.2f}%")
            print(f"  Credit event probability:       {cds.credit_event_probability * 100:.2f}%")
            print(f"  Probability of positive return: {cds.positive_return_probability * 100:.2f}%")
            print(f"  Return on premium:              {cds.expected_actual_return * 100:.2f}%")
            print(f"  Expected loss ($):              {cds.expected_loss:.2f}")
            print(f"  Expected loss (%):              {cds.expected_loss_pct * 100:.2f}%")
            print(f"  Full loss probability:          {cds.full_loss_probability * 100:.2f}%")
            print(f"  Benchmark loan return:          {loan.yield_to_maturity * 100:.2f}%\n")

        print(f"Weighted avg. annualized return:  {weighted_annuity_sum * 100:.2f}%")

        if return_as_df:
            return pd.DataFrame(results)

    def summarize_equity(self):
        r = self.loans[0].interest_rate
        T = self.loans[0].maturity
        freq = self.payment_frequency

        total_loan_principal = sum(loan.principal for loan in self.loans)
        total_loan_interest = sum(loan.principal * r * T for loan in self.loans)
        total_annuity_payments = sum(
            (cds.fixed_payment / (cds.maturity * freq)) * freq * T for cds in self.tranches
        )

        residual_cashflow = (
            self.mean
            - total_loan_principal
            - total_loan_interest
            - total_annuity_payments
        )

        highest_detachment = max(cds.detachment for cds in self.tranches)
        equity_notional = self.total_notional - highest_detachment
        prob_positive_cashflow = 1.0 - norm.cdf(highest_detachment, self.mean, self.std_dev)

        print("=== Equity Tranche Summary ===")
        print(f"  Implied attachment:            {highest_detachment:.2f}")
        print(f"  Implied notional:              {equity_notional:.2f}")
        print(f"  Expected residual cashflow:    {residual_cashflow:.2f}")
        print(f"  Probability of positive cash:  {prob_positive_cashflow * 100:.2f}%")

        if equity_notional > 0:
            expected_equity_return = residual_cashflow / equity_notional
            print(f"  Expected return:               {expected_equity_return * 100:.2f}%\n")
        else:
            print("  No expected return (no residual notional).\n")


# ── Interactive UI ──────────────────────────────────────────────────────────

def run_synthetic_cdo_ui():
    """
    Interactive ipywidgets UI for building and visualising a SyntheticCDO.
    Run inside a Jupyter notebook or Google Colab cell.
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    num_tranches = widgets.BoundedIntText(value=3, min=1, max=10, step=1, description='Tranches:')
    display(num_tranches)

    confirm_button = widgets.Button(description='Continue')
    display(confirm_button)

    def on_confirm_clicked(_):
        clear_output()
        display(num_tranches)
        n = num_tranches.value

        tranche_sizes_widgets    = [widgets.FloatText(description=f'Size T{i+1}')   for i in range(n)]
        expected_returns_widgets = [widgets.FloatText(description=f'Return T{i+1}') for i in range(n)]

        print('Enter tranche sizes and target returns (as decimals, e.g. 0.05 = 5%):')
        for w in tranche_sizes_widgets + expected_returns_widgets:
            display(w)

        mean           = widgets.FloatText(value=120,  description='Mean:')
        std_dev        = widgets.FloatText(value=30,   description='Std Dev:')
        notional       = widgets.FloatText(value=100,  description='Notional:')
        risk_free_rate = widgets.FloatText(value=0.04, description='r:')
        maturity       = widgets.FloatText(value=1,    description='Maturity:')
        payment_freq   = widgets.Dropdown(options=[1, 2, 4, 12], value=4, description='Frequency:')

        print('\nPortfolio parameters:')
        display(mean, std_dev, notional, risk_free_rate, maturity, payment_freq)

        build_button = widgets.Button(description='Build CDO')
        display(build_button)
        output_area = widgets.Output()
        display(output_area)

        def on_build_clicked(_):
            with output_area:
                clear_output()
                sizes   = [w.value for w in tranche_sizes_widgets]
                returns = [w.value for w in expected_returns_widgets]

                cdo = SyntheticCDO(
                    num_tranches=n,
                    tranche_sizes=sizes,
                    expected_returns=returns,
                    mean=mean.value,
                    std_dev=std_dev.value,
                    notional=notional.value,
                    risk_free_rate=risk_free_rate.value,
                    maturity=maturity.value,
                    payment_frequency=payment_freq.value
                )

                cdo.summarize()
                cdo.summarize_equity()

                mu, sigma = mean.value, std_dev.value
                xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
                ys = norm.pdf(xs, mu, sigma)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(xs, ys, label='Cashflow Distribution', lw=2, color='black')

                colors = plt.cm.Pastel1.colors
                for i, cds in enumerate(cdo.tranches):
                    a, d = cds.attachment, cds.detachment
                    label = (
                        f'Tranche {i+1}\n{a:.0f}-{d:.0f}\n'
                        f'Ret: {cds.expected_return*100:.1f}%\n'
                        f'P(Loss): {cds.credit_event_probability*100:.1f}%\n'
                        f'P(Profit): {cds.positive_return_probability*100:.1f}%'
                    )
                    ax.axvspan(a, d, alpha=0.3, color=colors[i % len(colors)])
                    ax.text((a + d) / 2, max(ys) * 0.6, label,
                            ha='center', va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.85))

                ax.axvline(mu, color='red', linestyle='--', lw=1.5, label='Expected Cashflow')
                ax.set_title('Portfolio Cashflow Distribution and Tranche Breakdown', fontsize=14)
                ax.set_xlabel('Portfolio Cashflow')
                ax.set_ylabel('Density')
                ax.grid(True)
                ax.legend(loc='upper right')
                plt.tight_layout()
                plt.show()

        build_button.on_click(on_build_clicked)

    confirm_button.on_click(on_confirm_clicked)
