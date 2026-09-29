import numpy as np


class HCIModel:
    """Physics-informed hot-carrier degradation proxy."""

    K_B_EV = 8.617333262145e-5

    def __init__(
        self,
        B: float,
        m: float,
        time_exp: float = 0.5,
        voltage_ref_v: float = 0.8,
        voltage_exp: float = 3.0,
        temperature_ref_k: float = 373.0,
        activation_energy_ev: float = 0.10,
    ):
        self.B = B
        self.m = m
        self.time_exp = time_exp
        self.voltage_ref_v = voltage_ref_v
        self.voltage_exp = voltage_exp
        self.temperature_ref_k = temperature_ref_k
        self.activation_energy_ev = activation_energy_ev

    def compute_degradation(
        self,
        current_density: np.ndarray,
        stress_time_s: np.ndarray,
        voltage: np.ndarray | None = None,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        current_density = np.asarray(current_density, dtype=np.float64)
        stress_time_s = np.asarray(stress_time_s, dtype=np.float64)

        if voltage is None:
            voltage = np.full_like(current_density, self.voltage_ref_v, dtype=np.float64)
        else:
            voltage = np.asarray(voltage, dtype=np.float64)

        if temperature_k is None:
            temperature_k = np.full_like(current_density, self.temperature_ref_k, dtype=np.float64)
        else:
            temperature_k = np.asarray(temperature_k, dtype=np.float64)

        if np.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be greater than 0 K")

        stress_factor = np.power(
            np.clip(current_density, 0.0, None) + 1e-12,
            self.m,
        )

        time_factor = np.power(
            np.clip(stress_time_s, 0.0, None),
            self.time_exp,
        )

        voltage_factor = np.power(
            np.clip(voltage, 1e-12, None) / self.voltage_ref_v,
            self.voltage_exp,
        )

        temp_exponent = (
            self.activation_energy_ev
            / self.K_B_EV
            * (1.0 / self.temperature_ref_k - 1.0 / temperature_k)
        )
        temperature_factor = np.exp(np.clip(temp_exponent, -50.0, 50.0))

        return (
            self.B
            * stress_factor
            * time_factor
            * voltage_factor
            * temperature_factor
        )
    def accumulate(
        self,
        existing_degradation: np.ndarray,
        current_density: np.ndarray,
        delta_t: float,
        voltage: np.ndarray | None = None,
        temperature_k: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance raw HCI degradation by one stress interval."""
        existing_degradation = np.asarray(existing_degradation, dtype=np.float64)
        current_density = np.asarray(current_density, dtype=np.float64)

        if existing_degradation.shape != current_density.shape:
            raise ValueError("existing_degradation and current_density must have the same shape")
        if delta_t < 0.0:
            raise ValueError("delta_t must be nonnegative")
        if self.B <= 0.0 or self.time_exp <= 0.0:
            raise ValueError("HCI B and time_exp must be positive")

        if voltage is None:
            voltage = np.full_like(current_density, self.voltage_ref_v, dtype=np.float64)
        else:
            voltage = np.asarray(voltage, dtype=np.float64)

        if temperature_k is None:
            temperature_k = np.full_like(current_density, self.temperature_ref_k, dtype=np.float64)
        else:
            temperature_k = np.asarray(temperature_k, dtype=np.float64)

        if voltage.shape != current_density.shape:
            raise ValueError("voltage must match current_density shape")
        if temperature_k.shape != current_density.shape:
            raise ValueError("temperature_k must match current_density shape")
        if np.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be greater than 0 K")
        if np.any(existing_degradation < 0.0):
            raise ValueError("existing_degradation must be nonnegative")
        if delta_t == 0.0:
            return existing_degradation.copy()

        current = np.clip(current_density, 0.0, None)
        stress_factor = np.where(
            current > 0.0,
            np.power(current + 1e-12, self.m),
            0.0,
        )
        voltage_factor = np.power(
            np.clip(voltage, 1e-12, None) / self.voltage_ref_v,
            self.voltage_exp,
        )
        temp_exponent = (
            self.activation_energy_ev
            / self.K_B_EV
            * (1.0 / self.temperature_ref_k - 1.0 / temperature_k)
        )
        temperature_factor = np.exp(np.clip(temp_exponent, -50.0, 50.0))

        exposure = np.power(
            np.clip(existing_degradation, 0.0, None) / self.B,
            1.0 / self.time_exp,
        )
        rate = np.power(
            stress_factor * voltage_factor * temperature_factor,
            1.0 / self.time_exp,
        )
        exposure_new = exposure + rate * float(delta_t)
        return self.B * np.power(exposure_new, self.time_exp)
