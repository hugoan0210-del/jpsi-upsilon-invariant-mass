import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = r"location of the file containing the muon data"

# 0.1 MeV -> GeV
SCALE = 1e-4

SAVE_PDFS = True

DO_CLASS_PLOT = True
DO_SIGMA_PLOTS = True 

# Zoom windows
WIN_JPSI = (2.8, 3.4)
WIN_UPS  = (9.0, 10.2)

# PDF outputs
OUT_GLOBAL = "histograma_global.pdf"
OUT_ZOOM_J = "zoom_jpsi.pdf"
OUT_ZOOM_U = "zoom_upsilon.pdf"
OUT_CLASS  = "histograma_por_clase.pdf"
OUT_SIG_J  = "zoom_jpsi_sigmas.pdf"
OUT_SIG_U  = "zoom_upsilon_sigmas.pdf"
OUT_Z_J    = "zscore_jpsi.pdf"
OUT_Z_U    = "zscore_upsilon.pdf"

# =========================================================
# 1) Load data
# =========================================================
df = pd.read_excel(FILE_PATH)

required = {"Q1", "Q2", "E1", "E2", "px1", "py1", "pz1", "px2", "py2", "pz2"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in Excel: {missing}")

# =========================================================
# 2) Filter opposite charges
# =========================================================
df = df[(df["Q1"] + df["Q2"]) == 0].copy()

# =========================================================
# 3) Scaling energies and momenta to GeV (using E from the dataset)
# =========================================================
E1  = df["E1"].to_numpy()  * SCALE
E2  = df["E2"].to_numpy()  * SCALE

px1 = df["px1"].to_numpy() * SCALE
py1 = df["py1"].to_numpy() * SCALE
pz1 = df["pz1"].to_numpy() * SCALE

px2 = df["px2"].to_numpy() * SCALE
py2 = df["py2"].to_numpy() * SCALE
pz2 = df["pz2"].to_numpy() * SCALE

# =========================================================
# 4) invariant mass
# =========================================================
E  = E1 + E2
px = px1 + px2
py = py1 + py2
pz = pz1 + pz2

m2 = E**2 - (px**2 + py**2 + pz**2)

# Physical truncation prevents artificial peak at 0
masses = np.sqrt(m2[m2 > 0])

# =========================================================
# 5) Global histogram
# =========================================================
plt.figure(figsize=(10, 5))
plt.hist(masses, bins=100, range=(0, 12), alpha=0.7, label="Datos")

plt.axvline(3.0969, linestyle="--", linewidth=1, label="J/ψ esperado")
plt.axvline(9.4600, linestyle="--", linewidth=1, label="ϒ esperado")

plt.xlabel("Masa invariante (GeV)")
plt.ylabel("Número de eventos")
plt.title("Masa invariante de pares de muones")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PDFS:
    plt.savefig(OUT_GLOBAL, bbox_inches="tight")
plt.show()

# =========================================================
# 6) Zooms
# =========================================================

# --- Zoom J/ψ ---
plt.figure(figsize=(8, 4))
plt.hist(masses, bins=60, range=WIN_JPSI, alpha=0.7, label="Data")
plt.axvline(3.0969, linestyle="--", linewidth=1, label="J/ψ expected")

plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Number of events")
plt.title("Zoom region J/ψ → μ⁺μ⁻")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PDFS:
    plt.savefig(OUT_ZOOM_J, bbox_inches="tight")
plt.show()

# --- Zoom ϒ ---
plt.figure(figsize=(8, 4))
plt.hist(masses, bins=60, range=WIN_UPS, alpha=0.7, label="Data")
plt.axvline(9.4600, linestyle="--", linewidth=1, label="ϒ expected")

plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Number of events")
plt.title("Zoom region ϒ → μ⁺μ⁻")
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PDFS:
    plt.savefig(OUT_ZOOM_U, bbox_inches="tight")
plt.show()

# =========================================================
# 7) Histogram by class
# =========================================================
if DO_CLASS_PLOT and ("class" in df.columns):
    df["m2"] = m2
    df["m"] = np.where(df["m2"] > 0, np.sqrt(df["m2"]), np.nan)

    jpsi = df[df["class"] == "J/psi"]["m"].dropna()
    ups  = df[df["class"] == "upsilon"]["m"].dropna()

    plt.figure(figsize=(10, 5))
    if len(jpsi) > 0:
        plt.hist(jpsi, bins=80, range=(0, 12), alpha=0.5, label="J/ψ (labeled)")
    if len(ups) > 0:
        plt.hist(ups, bins=80, range=(0, 12), alpha=0.5, label="ϒ (labeled)")

    plt.axvline(3.0969, linestyle="--", linewidth=1, label="J/ψ expected")
    plt.axvline(9.4600, linestyle="--", linewidth=1, label="ϒ expected")

    plt.xlabel("Invariant mass (GeV)")
    plt.ylabel("Number of events")
    plt.title("invariant mass per class")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if SAVE_PDFS:
        plt.savefig(OUT_CLASS, bbox_inches="tight")
    plt.show()

# =========================================================
# 8) σ and z-score bands
#    Now we use: Gauss + linear background
# =========================================================
if DO_SIGMA_PLOTS:
    try:
        from scipy.optimize import curve_fit
        from scipy.stats import norm

        def gauss(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        def gauss_lin_bkg(x, A, mu, sigma, c0, c1):
            return gauss(x, A, mu, sigma) + (c0 + c1*x)

        def fit_signal_bkg_window(masses_arr, window, mu_guess, bins=60):
            sel = masses_arr[(masses_arr >= window[0]) & (masses_arr <= window[1])]

            counts, edges = np.histogram(sel, bins=bins, range=window)
            centers = 0.5*(edges[1:] + edges[:-1])
            yerr = np.sqrt(np.maximum(counts, 1))

            A0 = counts.max() if len(counts) else 1
            sigma0 = 0.03 if mu_guess < 5 else 0.08
            c00 = np.median(counts) if len(counts) else 1
            c10 = 0.0

            p0 = [A0, mu_guess, sigma0, c00, c10]

            popt, pcov = curve_fit(
                gauss_lin_bkg, centers, counts, p0=p0,
                sigma=yerr, absolute_sigma=True, maxfev=10000
            )
            perr = np.sqrt(np.diag(pcov))
            return sel, counts, edges, popt, perr

        def plot_zoom_with_sigma_lines(masses_arr, window, mu_guess, title,
                                       bins=60, save_pdf=None):
            sel, counts, edges, popt, perr = fit_signal_bkg_window(
                masses_arr, window, mu_guess, bins=bins
            )
            A, mu, sigma, c0, c1 = popt

            xfit = np.linspace(window[0], window[1], 400)

            plt.figure(figsize=(8, 4))
            plt.hist(sel, bins=bins, range=window, alpha=0.7, label="Datos")
            plt.plot(xfit, gauss_lin_bkg(xfit, *popt),
                     label=f"Fit sign+background: μ={mu:.4f} GeV, σ={sigma:.4f} GeV")

            # Líneas en μ y ±nσ
            plt.axvline(mu, linestyle="--", linewidth=1)
            for n in [1, 2, 3]:
                plt.axvline(mu + n*sigma, linestyle=":", linewidth=1)
                plt.axvline(mu - n*sigma, linestyle=":", linewidth=1)

            # Evita que el eje se estire raro
            plt.xlim(window)

            plt.xlabel("Invariant mass (GeV)")
            plt.ylabel("Number of events")
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if save_pdf:
                plt.savefig(save_pdf, bbox_inches="tight")
            plt.show()

            return mu, sigma, perr

        def plot_zscore_hist(masses_arr, window, mu, sigma, title,
                             bins=50, save_pdf=None):
            sel = masses_arr[(masses_arr >= window[0]) & (masses_arr <= window[1])]
            z = (sel - mu) / sigma

            x = np.linspace(-5, 5, 400)

            plt.figure(figsize=(6, 4))
            plt.hist(z, bins=bins, range=(-5, 5), alpha=0.7,
                     density=True, label="Data")
            plt.plot(x, norm.pdf(x), label="N(0,1)")

            plt.axvline(0, linestyle="--", linewidth=1)
            for n in [1, 2, 3]:
                plt.axvline(n, linestyle=":", linewidth=1)
                plt.axvline(-n, linestyle=":", linewidth=1)

            plt.xlabel(r"$(m-\mu)/\sigma$")
            plt.ylabel("Density")
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if save_pdf:
                plt.savefig(save_pdf, bbox_inches="tight")
            plt.show()

        # --- J/ψ ---
        mu_j, sigma_j, perr_j = plot_zoom_with_sigma_lines(
            masses, WIN_JPSI, 3.0969,
            "J/ψ with standard deviation bands",
            save_pdf=OUT_SIG_J if SAVE_PDFS else None
        )

        # --- ϒ ---
        mu_u, sigma_u, perr_u = plot_zoom_with_sigma_lines(
            masses, WIN_UPS, 9.4600,
            "ϒ with standard deviation bands",
            save_pdf=OUT_SIG_U if SAVE_PDFS else None
        )

        # --- z-scores ---
        plot_zscore_hist(
            masses, WIN_JPSI, mu_j, sigma_j,
            "Distribution in units of σ (J/ψ)",
            save_pdf=OUT_Z_J if SAVE_PDFS else None
        )

        plot_zscore_hist(
            masses, WIN_UPS, mu_u, sigma_u,
            "Distribution in units of σ (ϒ)",
            save_pdf=OUT_Z_U if SAVE_PDFS else None
        )

        print("=== Local signal+background adjustment ===")
        print(f"J/ψ: μ = {mu_j:.5f} ± {perr_j[1]:.5f} GeV | σ = {sigma_j:.5f} ± {perr_j[2]:.5f} GeV")
        print(f"ϒ : μ = {mu_u:.5f} ± {perr_u[1]:.5f} GeV | σ = {sigma_u:.5f} ± {perr_u[2]:.5f} GeV")

    except ImportError:
        print("SciPy is not installed. Sigma and z-score graphs were omitted.")
