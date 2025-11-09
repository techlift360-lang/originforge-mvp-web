import numpy as np
import pandas as pd
from agents import Household, Firm, Government


def gini_proxy(values):
    v = np.asarray(values) + 1e-9
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * v).sum() / (n * v.sum()))


class World:
    """
    A slightly more realistic toy world:

    - Households receive wage income + UBI.
    - Government sets: tax, UBI, education spend, resource cap, regime.
    - Firms produce output based on productivity and effective resources.
    - Policies affect: GDP, inequality, emissions, innovation, stability.

    This is still stylized, but more responsive to sliders and shocks.
    """

    def __init__(
        self,
        n_households=500,
        n_firms=20,
        seed=42,
        tax_rate=0.2,
        ubi_rate=0.1,
        education_spend=0.05,
        resource_cap=0.2,
        regime="democracy",
    ):
        self.rng = np.random.default_rng(seed)

        # Agents
        base_wage = 1000.0
        wage_spread = 300.0
        self.households = [
            Household(
                wage=float(self.rng.normal(base_wage, wage_spread)),
                education=float(self.rng.uniform(0.8, 1.2)),
            )
            for _ in range(n_households)
        ]
        self.firms = [
            Firm(productivity=float(self.rng.uniform(0.8, 1.2)))
            for _ in range(n_firms)
        ]
        self.gov = Government(
            tax_rate=tax_rate,
            ubi_rate=ubi_rate,
            education_spend=education_spend,
            resource_cap=resource_cap,
            regime=regime,
        )

        # State
        self.tick = 0
        self.history = []

    # -----------------------
    # Core step + dynamics
    # -----------------------
    def step(self):
        self.tick += 1

        # 1) Wage & income basics
        wages = np.array([h.wage for h in self.households])
        median_income = float(np.median(wages))
        ubi_amount = self.gov.ubi_amount(median_income)

        # 2) Households react to policy: income, wealth, unrest
        total_tax_collected = 0.0
        for h in self.households:
            # tax on wage income only
            income = h.wage
            tax = self.gov.tax_rate * income
            total_tax_collected += tax
            transfer = ubi_amount
            net_income = income - tax + transfer

            # update wealth & unrest based on net income vs wage baseline
            h.wealth += net_income
            # if net_income < wage -> more unrest; if > wage -> less unrest
            h.unrest = max(0.0, min(1.0, 1.0 - (net_income / (h.wage + 1e-9))))

        # 3) Firms respond to policy: output, innovation, emissions
        total_output = 0.0
        total_emissions = 0.0
        innovations = []

        # policy to macro conversion:
        # - higher tax slightly hurts firm incentives
        # - more education spend boosts innovation
        # - stronger resource cap reduces output & emissions
        tax_drag = 1.0 - 0.5 * self.gov.tax_rate  # 0.9 at 0.2, 0.75 at 0.5
        edu_boost = 1.0 + 2.0 * self.gov.education_spend  # +10% at 0.05
        resource_factor = 1.0 - self.gov.resource_cap  # 0.8 if cap=0.2

        for f in self.firms:
            # base potential output
            base_output = 100.0 * f.productivity
            shock = float(self.rng.normal(1.0, 0.05))  # small noise
            output = base_output * tax_drag * resource_factor * shock
            output = max(0.0, output)

            emissions = 0.15 * output * (1.0 - self.gov.resource_cap)

            # innovation responds strongly to education spend
            innovation = 0.01 + 0.4 * self.gov.education_spend
            f.productivity *= (1.0 + innovation * 0.05)

            f.output = output
            f.emissions = emissions
            f.innovation = innovation

            total_output += output
            total_emissions += emissions
            innovations.append(innovation)

        gdp = total_output
        avg_innovation = float(np.mean(innovations))

        # 4) Stability & inequality metrics
        wealths = np.array([h.wealth for h in self.households])
        unrest = np.array([h.unrest for h in self.households])

        gini = gini_proxy(wealths)
        avg_unrest = float(np.mean(unrest))

        # regime slightly affects stability (toy assumption)
        regime_bonus = 0.05 if self.gov.regime == "democracy" else -0.02
        stability = float(
            max(0.0, min(1.0, 1.0 - avg_unrest + regime_bonus))
        )

        # 5) Random macro shock every ~50–100 ticks (very simple)
        # (This adds realism & variety)
        shock_prob = 0.01  # 1% chance per tick
        if self.rng.random() < shock_prob:
            # apply a negative shock to GDP and stability
            gdp *= 0.9
            stability *= 0.9

        # 6) Record snapshot
        snapshot = {
            "tick": self.tick,
            "population": len(self.households),
            "gdp": gdp,
            "emissions": total_emissions,
            "innovation": avg_innovation,
            "median_income": median_income,
            "ubi_amount": ubi_amount,
            "gini_proxy": gini,
            "stability": stability,
            "tax_rate": self.gov.tax_rate,
            "ubi_rate": self.gov.ubi_rate,
            "education_spend": self.gov.education_spend,
            "resource_cap": self.gov.resource_cap,
        }
        self.history.append(snapshot)

    def run(self, ticks=100):
        for _ in range(int(ticks)):
            self.step()
        return pd.DataFrame(self.history)
