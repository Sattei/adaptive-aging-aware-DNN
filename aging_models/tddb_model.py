import numpy as np


class TDDBModel:
    """Time-dependent dielectric-breakdown probability using a Weibull model."""

    K_B_EV = 8.617333262145e-5

    def __init__(
        self,
        k: float = 2.5,
        beta: float = 2.0,
        eta_ref_s: float = 5_000_000.0,
        field_ref: float = 0.8,
        temperature_ref_k: float = 373.0,
        activation_energy_ev: float = 0.30,
    ):
        self.k = k
        self.beta = beta
        self.eta_ref_s = eta_ref_s
        self.field_ref = field_ref
        self.temperature_ref_k = temperature_ref_k
        self.activation_energy_ev = activation_energy_ev

    def characteristic_lifetime(
        self,
        electric_field_proxy: np.ndarray,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        electric_field_proxy = np.asarray(electric_field_proxy, dtype=np.float64)

        if temperature_k is None:
            temperature_k = np.full_like(
                electric_field_proxy,
                self.temperature_ref_k,
                dtype=np.float64,
            )
        else:
            temperature_k = np.asarray(temperature_k, dtype=np.float64)

        if np.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be greater than 0 K")

        field_term = -self.k * (electric_field_proxy - self.field_ref)

        temp_term = (
            self.activation_energy_ev
            / self.K_B_EV
            * (1.0 / temperature_k - 1.0 / self.temperature_ref_k)
        )

        exponent = np.clip(field_term + temp_term, -50.0, 50.0)

        return np.clip(
            self.eta_ref_s * np.exp(exponent),
            1e-12,
            None,
        )

    def failure_probability(
        self,
        electric_field_proxy: np.ndarray,
        stress_time_s: np.ndarray,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        stress_time_s = np.asarray(stress_time_s, dtype=np.float64)
        eta = self.characteristic_lifetime(
            electric_field_proxy,
            temperature_k,
        )

        scaled_time = np.clip(stress_time_s, 0.0, None) / eta
        weibull_argument = np.power(scaled_time, self.beta)

        return np.clip(
            -np.expm1(-np.clip(weibull_argument, 0.0, 700.0)),
            0.0,
            1.0,
        )

    def time_to_failure(
        self,
        electric_field_proxy: np.ndarray,
        target_failure_prob: float = 0.001,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        eta = self.characteristic_lifetime(
            electric_field_proxy,
            temperature_k,
        )

        p = np.clip(target_failure_prob, 1e-9, 1.0 - 1e-9)

        return eta * np.power(
            -np.log1p(-p),
            1.0 / self.beta,
        )
    def accumulate_hazard(
        self,
        existing_hazard: np.ndarray,
        electric_field_proxy: np.ndarray,
        delta_t: float,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance cumulative TDDB hazard by one stress interval."""
        existing_hazard = np.asarray(existing_hazard, dtype=np.float64)
        electric_field_proxy = np.asarray(electric_field_proxy, dtype=np.float64)

        if existing_hazard.shape != electric_field_proxy.shape:
            raise ValueError("existing_hazard and electric_field_proxy must have the same shape")
        if delta_t < 0.0:
            raise ValueError("delta_t must be nonnegative")
        if self.beta <= 0.0:
            raise ValueError("TDDB beta must be positive")
        if np.any(existing_hazard < 0.0):
            raise ValueError("existing_hazard must be nonnegative")

        if temperature_k is None:
            temperature_k = np.full_like(
                electric_field_proxy,
                self.temperature_ref_k,
                dtype=np.float64,
            )
        else:
            temperature_k = np.asarray(temperature_k, dtype=np.float64)

        if temperature_k.shape != electric_field_proxy.shape:
            raise ValueError("temperature_k must match electric_field_proxy shape")
        if np.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be greater than 0 K")
        if delta_t == 0.0:
            return existing_hazard.copy()

        field = np.clip(electric_field_proxy, 0.0, None)
        eta = self.characteristic_lifetime(field, temperature_k)
        exposure = np.power(
            np.clip(existing_hazard, 0.0, None),
            1.0 / self.beta,
        )
        increment = np.where(
            field > 0.0,
            float(delta_t) / eta,
            0.0,
        )
        return np.power(exposure + increment, self.beta)

    @staticmethod
    def hazard_to_probability(hazard: np.ndarray) -> np.ndarray:
        hazard = np.asarray(hazard, dtype=np.float64)
        if np.any(hazard < 0.0):
            raise ValueError("hazard must be nonnegative")
        return np.clip(
            -np.expm1(-np.clip(hazard, 0.0, 700.0)),
            0.0,
            1.0,
        )
