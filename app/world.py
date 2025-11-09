"""
world.py

Core simulation engine for OriginForge — Policy Sandbox.

This is a stylized, toy model of a "world" with:
- A population (num_agents)
- Macro metrics: GDP, inequality (Gini proxy), stability, innovation, emissions
- Policy levers: tax_rate, ubi_rate, education_spend, resource_cap, regime
- Optional shocks: recession, climate_shock
- Optional policy feedback: government adapts policies over time
- Sectoral GDP: industry, services, green
- A simple "memory" / frustration index

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

        # Sectoral GDP decomposition (proxies)
        self.gdp_industry = self.gdp * 0.4
        self.gdp_services = self.gdp * 0.4
        self.gdp_green = self.gdp * 0.2

        # A simple "memory" / frustration index (0–1)
        # Higher when inequality is high and stability is low
        self.frustration = 0.3

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

    def _update_frustration(self) -> None:
        """
        Update a simple "frustration" index (0–1) based on inequality and stability.
        This acts like a memory of how citizens feel over time.
        """
        # Higher inequality -> more frustration
        # Higher stability -> less frustration
        self.frustration += 0.02 * (self.gini - 0.35)
        self.frustration -= 0.02 * (self.stability - 0.7)
        self.frustration = max(0.0, min(self.frustration, 1.0))

    def _adapt_policies(self, gdp_growth_real: float) -> None:
        """
        Very simple "policy feedback" system:
        - If inequality is high, government nudges tax and UBI up.
        - If GDP growth is weak/negative, it eases tax and boosts education.
        - If emissions are high, it tightens resource caps.
        - If stability is low, it boosts UBI and education.
        All adjustments are small per tick, creating slow-moving policy paths.
        """
        cfg = self.config

        # Inequality-driven adjustments
        if self.gini > 0.42:
            cfg.tax_rate += 0.0008
            cfg.ubi_rate += 0.0008
        elif self.gini < 0.30:
            cfg.tax_rate -= 0.0006
            cfg.ubi_rate -= 0.0004

        # Growth-driven adjustments
        if gdp_growth_real < -0.002:  # strong contraction
            cfg.tax_rate -= 0.0010
            cfg.education_spend += 0.0008
        elif gdp_growth_real < 0.002:  # weak growth
            cfg.education_spend += 0.0003

        # Emissions-driven adjustments
        if self.emissions > 5.0:
            cfg.resource_cap += 0.0009
        elif self.emissions < 1.5:
            cfg.resource_cap -= 0.0004

        # Stability-driven adjustments (via frustration)
        if self.frustration > 0.6:
            cfg.ubi_rate += 0.0007
            cfg.education_spend += 0.0005
        elif self.frustration < 0.3:
            cfg.resource_cap -= 0.0003

        # Clamp policies to allowed ranges
        cfg.tax_rate = float(max(0.0, min(cfg.tax_rate, 0.50)))
        cfg.ubi_rate = float(max(0.0, min(cfg.ubi_rate, 0.30)))
        cfg.education_spend = float(max(0.0, min(cfg.education_spend, 0.10)))
        cfg.resource_cap = float(max(0.0, min(cfg.resource_cap, 0.40)))

    def _update_sectors(self) -> None:
        """
        Update sectoral GDP (industry, services, green) based on current
        GDP, innovation, resource caps, and stability.

        This is a stylized decomposition; sums back to total GDP.
        """
        if self.gdp <= 0:
            self.gdp_industry = 0.0
            self.gdp_services = 0.0
            self.gdp_green = 0.0
            return

        # Baseline shares
        base_ind = 0.40
        base_serv = 0.40
        base_green = 0.20

        # Adjustments:
        # - Higher resource caps + innovation favor green sector
        # - Strong stability + education (proxied by innovation) support services
        # - Looser caps and lower innovation favor industry
        cap = self.config.resource_cap
        innov = self.innovation
        stab = self.stability

        share_green = base_green + 0.10 * innov + 0.08 * cap
        share_ind = base_ind + 0.06 * (1.0 - cap) - 0.04 * innov
        share_serv = base_serv + 0.05 * stab + 0.02 * innov

        # Normalize to sum to 1 and avoid negatives
        shares = np.array([share_ind, share_serv, share_green])
        shares = np.maximum(shares, 0.02)  # avoid zeros/negatives
        shares = shares / shares.sum()

        self.gdp_industry = float(self.gdp * shares[0])
        self.gdp_services = float(self.gdp * shares[1])
        self.gdp_green = float(self.gdp * shares[2])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        ticks: int,
        recession: bool = False,
        climate_shock: bool = False,
        adapt_policies: bool = False,
    ) -> pd.DataFrame:
        """
        Run the world forward for a given number of ticks, recording metrics.

        Shocks:
        - recession: one-time GDP & stability hit at mid-run.
        - climate_shock: emissions spike & stability hit at 80% of the run.

        If adapt_policies is True, the government gradually adjusts tax, UBI,
        education, and resource caps based on current conditions.
        """
        ticks = max(1, min(int(ticks), 5000))

        records: List[Dict[str, Any]] = []

        for t in range(ticks):
            prev_gdp = self.gdp

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

            # 3) Update frustration (memory)
            self._update_frustration()

            # 4) Policy feedback (if enabled)
            if prev_gdp > 0:
                gdp_growth_real = (self.gdp - prev_gdp) / prev_gdp
            else:
                gdp_growth_real = 0.0

            if adapt_policies:
                self._adapt_policies(gdp_growth_real)

            # 5) Sectoral decomposition
            self._update_sectors()

            # 6) Record snapshot, including current policy settings and sectors
            records.append(
                {
                    "tick": t,
                    "population": float(self.population),
                    "gdp": float(self.gdp),
                    "gini_proxy": float(self.gini),
                    "stability": float(self.stability),
                    "innovation": float(self.innovation),
                    "emissions": float(self.emissions),
                    "tax_rate": float(self.config.tax_rate),
                    "ubi_rate": float(self.config.ubi_rate),
                    "education_spend": float(self.config.education_spend),
                    "resource_cap": float(self.config.resource_cap),
                    "frustration": float(self.frustration),
                    "gdp_industry": float(self.gdp_industry),
                    "gdp_services": float(self.gdp_services),
                    "gdp_green": float(self.gdp_green),
                }
            )

        df = pd.DataFrame.from_records(records)
        return df
