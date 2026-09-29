import numpy as np


class NBTIModel:
    """Physics-informed NBTI degradation proxy."""

    K_B_EV = 8.617333262145e-5

    def __init__(
        self,
        A: float,
        n: float,
        temperature_K: float = 373.0,
        voltage_ref_v: float = 0.8,
        voltage_exp: float = 4.0,
        activation_energy_ev: float = 0.15,
    ):
        self.A = A
        self.n = n
        self.temperature_K = temperature_K
        self.voltage_ref_v = voltage_ref_v
        self.voltage_exp = voltage_exp
        self.activation_energy_ev = activation_energy_ev

    def compute_degradation(
        self,
        stress_time_s: np.ndarray,
        switching_activity: np.ndarray,
        voltage: np.ndarray | None = None,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        stress_time_s = np.asarray(stress_time_s, dtype=np.float64)
        switching_activity = np.asarray(switching_activity, dtype=np.float64)

        if voltage is None:
            voltage = np.full_like(switching_activity, self.voltage_ref_v, dtype=np.float64)
        else:
            voltage = np.asarray(voltage, dtype=np.float64)

        if temperature_k is None:
            temperature_k = np.full_like(switching_activity, self.temperature_K, dtype=np.float64)
        else:
            temperature_k = np.asarray(temperature_k, dtype=np.float64)

        if np.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be greater than 0 K")

        effective_stress = np.clip(
            switching_activity * stress_time_s,
            1e-12,
            None,
        )

        voltage_factor = np.power(
            np.clip(voltage, 1e-12, None) / self.voltage_ref_v,
            self.voltage_exp,
        )

        temp_exponent = (
            self.activation_energy_ev
            / self.K_B_EV
            * (1.0 / self.temperature_K - 1.0 / temperature_k)
        )
        temperature_factor = np.exp(np.clip(temp_exponent, -50.0, 50.0))

        return (
            self.A
            * np.power(effective_stress, self.n)
            * voltage_factor
            * temperature_factor
        )

    def accumulate(
        self,
        existing_degradation: np.ndarray,
        new_stress: np.ndarray,
        delta_t: float,
        voltage: np.ndarray | None = None,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance raw NBTI degradation by one stress interval."""
        existing_degradation = np.asarray(existing_degradation, dtype=np.float64)
        new_stress = np.asarray(new_stress, dtype=np.float64)

        if existing_degradation.shape != new_stress.shape:
            raise ValueError("existing_degradation and new_stress must have the same shape")
        if delta_t < 0.0:
            raise ValueError("delta_t must be nonnegative")
        if self.A <= 0.0 or self.n <= 0.0:
            raise ValueError("NBTI A and n must be positive")

        if voltage is None:
            voltage = np.full_like(new_stress, self.voltage_ref_v, dtype=np.float64)
        else:
            voltage = np.asarray(voltage, dtype=np.float64)

        if temperature_k is None:
            temperature_k = np.full_like(new_stress, self.temperature_K, dtype=np.float64)
        else:
            temperature_k = np.asarray(temperature_k, dtype=np.float64)

        if voltage.shape != new_stress.shape:
            raise ValueError("voltage must match new_stress shape")
        if temperature_k.shape != new_stress.shape:
            raise ValueError("temperature_k must match new_stress shape")
        if np.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be greater than 0 K")
        if np.any(existing_degradation < 0.0):
            raise ValueError("existing_degradation must be nonnegative")
        if delta_t == 0.0:
            return existing_degradation.copy()

        voltage_factor = np.power(
            np.clip(voltage, 1e-12, None) / self.voltage_ref_v,
            self.voltage_exp,
        )
        temp_exponent = (
            self.activation_energy_ev
            / self.K_B_EV
            * (1.0 / self.temperature_K - 1.0 / temperature_k)
        )
        temperature_factor = np.exp(np.clip(temp_exponent, -50.0, 50.0))

        exposure = np.power(
            np.clip(existing_degradation, 0.0, None) / self.A,
            1.0 / self.n,
        )
        activity = np.clip(new_stress, 0.0, None)
        rate = activity * np.power(
            voltage_factor * temperature_factor,
            1.0 / self.n,
        )
        exposure_new = exposure + rate * float(delta_t)
        return self.A * np.power(exposure_new, self.n)
