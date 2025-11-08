import numpy as np

class Household:
    def __init__(self, wage=1000.0, education=1.0):
        self.wage = wage
        self.education = education
        self.wealth = 0.0
        self.unrest = 0.0

    def step(self, tax_rate, ubi_amount):
        income = self.wage
        tax = tax_rate * income
        transfer = ubi_amount
        net = income - tax + transfer
        self.wealth += net
        self.unrest = max(0.0, 1.0 - (net / (self.wage + 1e-9)))

class Firm:
    def __init__(self, productivity=1.0):
        self.productivity = productivity
        self.output = 0.0
        self.emissions = 0.0
        self.innovation = 0.0

    def step(self, resource_cap, education_spend):
        base_output = 100.0 * self.productivity
        cap_factor = (1.0 - resource_cap)
        self.output = base_output * cap_factor
        self.emissions = 0.1 * self.output / max(0.1, self.productivity)
        self.innovation = 0.02 + 0.5 * education_spend
        self.productivity *= (1.0 + self.innovation * 0.01)

class Government:
    def __init__(self, tax_rate=0.2, ubi_rate=0.1, education_spend=0.05, resource_cap=0.2, regime="democracy"):
        self.tax_rate = tax_rate
        self.ubi_rate = ubi_rate
        self.education_spend = education_spend
        self.resource_cap = resource_cap
        self.regime = regime

    def ubi_amount(self, median_income):
        return self.ubi_rate * median_income
