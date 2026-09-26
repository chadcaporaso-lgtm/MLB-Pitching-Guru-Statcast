import numpy as np
from scipy.stats import nbinom

class DynamicNegBinomialEngine:
    def __init__(self):
        self.theta_0 = 2.67       # ln(14.5) ~ 2.67 baseline dispersion
        self.theta_var = -0.42    # High starter exit variance fattens tails (reduces r)
        self.theta_adi = -0.85    # Thin/warm air increases scoring variance
        self.theta_wind = -0.045  # Tailwind vector along CF axis amplifies run volatility

    def compute_dispersion(
        self,
        starter_exit_var: float,
        adi_delta: float,
        wind_speed_mph: float,
        wind_angle_deg: float,
        cf_azimuth_deg: float
    ) -> float:
        """Dynamically computes parameter r for discrete game and F5 totals."""
        angle_diff = np.radians(wind_angle_deg - cf_azimuth_deg)
        tailwind_vector = wind_speed_mph * np.cos(angle_diff)
        
        log_r = (
            self.theta_0 
            + self.theta_var * starter_exit_var 
            + self.theta_adi * adi_delta 
            + self.theta_wind * tailwind_vector
        )
        r = np.clip(np.exp(log_r), 4.5, 25.0)
        return float(r)

    def price_total_market(self, projected_mu: float, r: float, line: float) -> dict:
        """Calculates exact Over/Under probabilities using dynamically dispersed NegBin."""
        p = r / (r + projected_mu)
        under_k = int(np.floor(line))
        p_under = nbinom.cdf(under_k, r, p)
        p_over = 1.0 - p_under
        
        return {
            "projected_mean": projected_mu,
            "dynamic_r": round(r, 3),
            "line": line,
            "p_over": round(float(p_over), 4),
            "p_under": round(float(p_under), 4),
            "fair_over_dec": round(1.0 / p_over, 3),
            "fair_under_dec": round(1.0 / p_under, 3)
        }
