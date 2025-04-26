from math import ceil

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


class ABTestResults:
    def __init__(
        self,
        df: pd.DataFrame,
        group_col: str = "group",
        conversion_col: str = "converted",
        expected_control_conversion_rate: float = 0,
        expected_treatment_conversion_rate: float = 0,
        desired_power: float = 0.8,
        alpha: float = 0.05,
    ):
        self.df = df
        self.group_col = group_col
        self.conversion_col = conversion_col
        self.expected_control_conversion_rate = expected_control_conversion_rate
        self.expected_treatment_conversion_rate = expected_treatment_conversion_rate
        self.desired_power = desired_power
        self.alpha = alpha
        self.required_n = self.get_required_samples(
            self.expected_control_conversion_rate,
            self.expected_treatment_conversion_rate,
            self.desired_power,
            self.alpha,
        )
        self.conversion_rates = self.get_converson_rate(self.df)

    def get_required_samples(
        self,
        expected_control_conversion_rate,
        expected_treatment_conversion_rate,
        desired_power=0.8,
        alpha=0.05,
    ):
        self.effect_size = sms.proportion_effectsize(
            expected_control_conversion_rate, expected_treatment_conversion_rate
        )  # Calculating effect size based on our expected rates

        required_n = sms.NormalIndPower().solve_power(
            self.effect_size, power=desired_power, alpha=alpha, ratio=1
        )  # Calculating sample size needed

        required_n = ceil(required_n)  # Rounding up to next whole number

        return required_n

    def get_converson_rate(self, ab_test):
        conversion_rates = ab_test.groupby(self.group_col)[self.conversion_col]

        def std_p(x):
            return np.std(x, ddof=0)  # Std. deviation of the proportion

        def se_p(x):
            return stats.sem(x, ddof=0)  # Std. error of the proportion (std / sqrt(n))

        conversion_rates = conversion_rates.agg(["mean", std_p, se_p])
        conversion_rates.columns = ["conversion_rate", "std_deviation", "std_error"]
        return conversion_rates

    def experiment_results(self, ab_test):
        control_results = ab_test[ab_test[self.group_col] == "control"][self.conversion_col]
        treatment_results = ab_test[ab_test[self.group_col] == "treatment"][self.conversion_col]

        n_con = control_results.count()
        n_treat = treatment_results.count()
        successes = [control_results.sum(), treatment_results.sum()]
        nobs = [n_con, n_treat]

        z_stat, pval = proportions_ztest(successes, nobs=nobs)
        (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

        print(f"z statistic: {z_stat:.2f}")
        print(f"p-value: {pval:.3f}")
        print(f"ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]")
        print(f"ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]")
        return {
            "z_stat": z_stat,
            "pval": pval,
            "ci_95_lower_con": lower_con,
            "ci_95_upper_con": upper_con,
            "ci_95_lower_treat": lower_treat,
            "ci_95_upper_treat": upper_treat,
        }


if __name__ == "__main__":
    df = pd.DataFrame()
    abres = ABTestResults(
        df,
        group_col="group",
        conversion_col="converted",
        expected_control_conversion_rate=0.13,
        expected_treatment_conversion_rate=0.15,
        desired_power=0.8,
        alpha=0.05,
    )
    abres.experiment_results(df)
