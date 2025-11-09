"""
world.py

Core simulation engine for OriginForge — Policy Sandbox.

This is a stylized, toy model of a "world" with:
- A population (num_agents)
- Macro metrics: GDP, inequality (Gini proxy), stability, innovation, emissions
- Policy levers: tax_rate, ubi_rate, education_spend, resource_cap, regime
- Optional shocks: recession, climate_shock

It is designed for:
- Educational use
- Fast experimentation
- Visual inspection of trends

It is NOT calibrated to any real country or dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import random


@dataclass
class WorldConfig:
    tax_rate: float
    ubi_rate: float
    education_spend: float
    resource_cap: float
    regime: str
    num_agents: int = 500


class World:
    def __init__(
        self,
        tax_rate: float,
        ubi_rate: float,
        education_spend: float,
        resource_cap: float,
        regime: str,
        num_agents: int = 500,
    ) -> None:
        """
        Initialize a new world with given policy parameters.

        num_agents is used primarily for the population metric.
        The underlying model is aggregate, not fully micro-agent-based.
        """
        self.config = WorldConfig(
            tax_rate=float(tax_rate),
            ubi_rate=float(ubi_rate),
            education_spend=float(education_spend),
            resource_cap=float(resource_cap),
            regime=str(regime),
            num_agents=max(50, min(int(num_agents), 5000)),
        )

        # Baseline macro state (stylized)
        self.population = float(self.config.num_agents)
        self.gdp = 1000.0
        self.gini = 0.35  # inequality proxy
        self.stability = 0.7
        self.innovation = 0.3
        self.emissions = 1.0

        # Small random seed for reproducibility of noise
        random.seed(42)
        np.random.seed(42)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _apply_policy_effects(self) -> None:
        """
        Apply gradual effects of policies on macro variables.

        The equations here are deliberately simple and smooth, with some
        loosely intuitive directions:
        - Higher tax: slightly lower GDP growth, lower inequality
        - Higher UBI: lower inequality, slight drag on GDP, stabilizes society
        - Higher education: boosts innovation + GDP, slowly reduces inequality
        - Higher resource cap: reduces emissions but also GDP a bit
        - Regime: democracy tends to higher stability, autocracy higher short-run GDP
        """
        cfg = self.config

        # Base drifts
        base_gdp_growth = 0.012  # 1.2% per tick baseline
        base_innovation_drift = 0.002
        base_emissions_drift = 0.003

        # Policy multipliers
        tax = cfg.tax_rate
        ubi = cfg.ubi_rate
        edu = cfg.education_spend
        cap = cfg.resource_cap

        # Regime effects
        if cfg.regime == "democracy":
            regime_stab_bonus = 0.01
            regime_gdp_bonus = -0.002
        else:  # "autocracy"
            regime_stab_bonus = -0.005
            regime_gdp_bonus = 0.004

        # GDP: base growth + edu boost - tax drag - cap drag + regime
        gdp_growth = (
            base_gdp_growth
            + 0.10 * edu       # education supports growth
            - 0.06 * tax       # taxes slightly reduce growth
            - 0.04 * cap       # resource caps reduce growth
            + regime_gdp_bonus
        )

        # Some noise
        gdp_noise = np.random.normal(0.0, 0.01)
        gdp_growth += gdp_noise
        self.gdp *= (1.0 + gdp_growth)
        self.gdp = max(100.0, min(self.gdp, 100000.0))

        # Inequality: tax & UBI reduce it, weak upward drift with growth
        gini_drift = 0.002  # baseline upward drift
        gini_drift -= 0.05 * tax
        gini_drift -= 0.08 * ubi
        gini_drift -= 0.03 * edu
        gini_noise = np.random.normal(0.0, 0.002)
        self.gini += gini_drift + gini_noise
        self.gini = max(0.15, min(self.gini, 0.80))

        # Stability: hurt by inequality and low GDP per capita; helped by UBI & democracy
        income_per_capita = self.gdp / max(1.0, self.population)
        income_norm = income_per_capita / 10.0  # rough normalization

        stab_drift = 0.0
        stab_drift += 0.05 * ubi
        stab_drift += regime_stab_bonus
        stab_drift += 0.01 * edu
        stab_drift -= 0.10 * (self.gini - 0.3)
        stab_drift += 0.02 * (income_norm - 1.0)

        stab_noise = np.random.normal(0.0, 0.01)
        self.stability += stab_drift + stab_noise
        self.stability = max(0.0, min(self.stability, 1.0))

        # Innovation: boosted by education & stability, slightly hurt by very high tax
        innov_drift = base_innovation_drift
        innov_drift += 0.10 * edu
        innov_drift += 0.05 * (self.stability - 0.5)
        innov_drift -= 0.04 * max(0, tax - 0.3)
        innov_noise = np.random.normal(0.0, 0.01)
        self.innovation += innov_drift + innov_noise
        self.innovation = max(0.0, min(self.innovation, 1.0))

        # Emissions: tied to GDP and resource caps, mitigated by innovation
        emissions_growth = base_emissions_drift
        emissions_growth += 0.04 * (self.gdp / 5000.0)
        emissions_growth -= 0.06 * cap
        emissions_growth -= 0.03 * self.innovation
        emissions_noise = np.random.normal(0.0, 0.01)
        self.emissions *= (1.0 + emissions_growth + emissions_noise)
        self.emissions = max(0.1, min(self.emissions, 100.0))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        ticks: int,
        recession: bool = False,
        climate_shock: bool = False,
    ) -> pd.DataFrame:
        """
        Run the world forward for a given number of ticks, recording metrics.

        Shocks:
        - recession: one-time GDP & stability hit at mid-run.
        - climate_shock: emissions spike & stability hit at 80% of the run.
        """
        ticks = max(1, min(int(ticks), 5000))

        records: List[Dict[str, Any]] = []

        for t in range(ticks):
            # 1) Apply normal policy-driven dynamics
            self._apply_policy_effects()

            # 2) Apply shocks at specific times
            if recession and t == ticks // 2:
                # Temporary GDP + stability hit
                self.gdp *= 0.85
                self.stability *= 0.9

            if climate_shock and t == int(ticks * 0.8):
                # Emissions spike, stability falls
                self.emissions *= 1.3
                self.stability *= 0.9

            # 3) Record snapshot
            records.append(
                {
                    "tick": t,
                    "population": float(self.population),
                    "gdp": float(self.gdp),
                    "gini_proxy": float(self.gini),
                    "stability": float(self.stability),
                    "innovation": float(self.innovation),
                    "emissions": float(self.emissions),
                }
            )

        df = pd.DataFrame.from_records(records)
        return df
