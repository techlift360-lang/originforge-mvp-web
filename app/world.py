import numpy as np
import pandas as pd
from agents import Household, Firm, Government

class World:
    def __init__(self, n_households=500, n_firms=20, seed=42,
                 tax_rate=0.2, ubi_rate=0.1, education_spend=0.05, resource_cap=0.2, regime="democracy"):
        self.rng = np.random.default_rng(seed)
        self.households = [
            Household(
                wage=self.rng.normal(1000, 200),
                education=self.rng.uniform(0.8, 1.2)
            )
            for _ in range(n_households)
        ]
        self.firms = [
            Firm(productivity=self.rng.uniform(0.8, 1.2))
            for _ in range(n_firms)
        ]
        self.gov = Government(tax_rate, ubi_rate, education_spend, resource_cap, regime)
        self.tick = 0
        self.history = []

    def metrics_snapshot(self):
        wages = np.array([h.wage for h in self.households])
        incomes = wages
        median_income = float(np.median(incomes))
        ubi_amt = self.gov.ubi_amount(median_income)
        wealths = np.array([h.wealth for h in self.households])
        unrest = np.array([h.unrest for h in self.households])
        gdp = sum(f.output for f in self.firms)
        emissions = sum(f.emissions for f in self.firms)
        innovation = np.mean([f.innovation for f in self.firms])
        return {
            "tick": self.tick,
            "population": len(self.households),
            "gdp": gdp,
            "emissions": emissions,
            "innovation": innovation,
            "median_income": median_income,
            "ubi_amount": ubi_amt,
            "gini_proxy": gini_proxy(wealths),
            "stability": float(1.0 - np.clip(np.mean(unrest), 0, 1)),
        }

    def step(self):
        self.tick += 1
        wages = np.array([h.wage for h in self.households])
        median_income = float(np.median(wages))
        ubi_amt = self.gov.ubi_amount(median_income)

        for h in self.households:
            h.step(self.gov.tax_rate, ubi_amt)
        for f in self.firms:
            f.step(self.gov.resource_cap, self.gov.education_spend)

        self.history.append(self.metrics_snapshot())

    def run(self, ticks=100):
        for _ in range(ticks):
            self.step()
        import pandas as pd  # local import for safety
        return pd.DataFrame(self.history)

def gini_proxy(values):
    v = np.asarray(values) + 1e-9
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * v).sum() / (n * v.sum()))
