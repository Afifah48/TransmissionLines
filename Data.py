import numpy as np
import pandas as pd

# === formulas ===
def characteristic_impedance(Rp, Lp, Gp, Cp, freq):
    omega = 2 * np.pi * freq
    Zp = Rp + 1j * omega * Lp
    Yp = Gp + 1j * omega * Cp
    return np.sqrt(Zp / Yp)

def propagation_constant(Rp, Lp, Gp, Cp, freq):
    omega = 2 * np.pi * freq
    Zp = Rp + 1j * omega * Lp
    Yp = Gp + 1j * omega * Cp
    return np.sqrt(Zp * Yp)

def velocity(beta, freq):
    omega = 2 * np.pi * freq
    return omega / beta

def vswr(refl_coeff):
    mag = np.abs(refl_coeff)
    return (1 + mag) / (1 - mag) if mag != 1 else np.inf

def reflection_coeff(ZL, Z0):
    return (ZL - Z0) / (ZL + Z0)

# === dataset generator ===
def generate_dataset(n_samples=5000):
    records = []

    for _ in range(n_samples):
        # random inputs
        Rp = np.random.uniform(0.1, 5.0)
        Lp = np.random.uniform(1e-8, 1e-6)
        Gp = np.random.uniform(1e-9, 1e-6)
        Cp = np.random.uniform(1e-12, 1e-10)
        freq = np.random.uniform(1e6, 1e9)
        length = np.random.uniform(0.01, 1.0)
        ZL_real = np.random.choice([25, 50, 75, 100])
        ZL_imag = np.random.choice([0, 10, -10, 20, -20])
        ZL = ZL_real + 1j * ZL_imag

        # outputs from formulas
        Z0 = characteristic_impedance(Rp, Lp, Gp, Cp, freq)
        gamma = propagation_constant(Rp, Lp, Gp, Cp, freq)
        alpha, beta = gamma.real, gamma.imag
        vp = velocity(beta, freq)
        refl = reflection_coeff(ZL, Z0)
        vswr = vswr(refl)

        records.append({
            # inputs
            "Rp": Rp, "Lp": Lp, "Gp": Gp, "Cp": Cp,
            "freq": freq, "ZL_real": ZL.real, "ZL_imag": ZL.imag, "length": length,
            # outputs
            "Z0_real": Z0.real, "Z0_imag": Z0.imag,
            "gamma_real": gamma.real, "gamma_imag": gamma.imag,
            "alpha": alpha, "beta": beta,
            "vp": vp,
            "refl_real": refl.real, "refl_imag": refl.imag,
            "vswr": vswr
        })

    df = pd.DataFrame(records)
    return df

# === create dataset & save ===
df = generate_dataset(10000)   # 10k samples
df.to_csv("txline_dataset.csv", index=False)

print("Dataset saved to txline_dataset.csv")
print(df.head())
