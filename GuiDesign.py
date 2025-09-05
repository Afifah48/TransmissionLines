import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib


# Load model & scaler

model = joblib.load("txline_model.pkl")
scaler = joblib.load("txline_scaler.pkl")

# Output names
output_names = [
    "Z0_real", "Z0_imag",
    "gamma_real", "gamma_imag",
    "alpha", "beta", "vp",
    "refl_real", "refl_imag", "vswr"
]


# GUI app

class TxLineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transmission Line ML Predictor")
        self.root.geometry("500x600")

        # Input fields
        self.entries = {}
        inputs = ["Rp", "Lp", "Gp", "Cp", "freq", "ZL_real", "ZL_imag", "length"]

        frm_inputs = ttk.LabelFrame(root, text="Input Parameters")
        frm_inputs.pack(fill="x", padx=10, pady=10)

        for i, name in enumerate(inputs):
            ttk.Label(frm_inputs, text=name + ":").grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(frm_inputs, width=20)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[name] = entry

        # Predict button
        self.btn_predict = ttk.Button(root, text="Predict", command=self.predict)
        self.btn_predict.pack(pady=10)

        # Results frame
        self.frm_results = ttk.LabelFrame(root, text="Predicted Outputs")
        self.frm_results.pack(fill="both", expand=True, padx=10, pady=10)

        self.text_results = tk.Text(self.frm_results, height=20, wrap="word")
        self.text_results.pack(fill="both", expand=True)

    def predict(self):
        try:
            # Collect input values
            values = [float(self.entries[name].get()) for name in self.entries]
            X_input = np.array([values])
            X_scaled = scaler.transform(X_input)

            # Predict
            y_pred = model.predict(X_scaled)[0]

            # Show results
            self.text_results.delete("1.0", tk.END)
            self.text_results.insert(tk.END, "Predicted Results:\n\n")
            for name, val in zip(output_names, y_pred):
                self.text_results.insert(tk.END, f"{name:12s}: {val:.6e}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))


# Run the GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = TxLineApp(root)
    root.mainloop()
