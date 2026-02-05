import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
# HW4 Q4: Critical Shear Stress for Dislocations Passing

> **Problem (WIRIS):** Consider two oppositely signed edge dislocations A and B with applied $\tau_{xy}$. A positive $\tau_{xy}$ moves A right and B left. B exerts stress on A, so A can only pass B if total $\tau_{xy}$ stays positive. **Find critical $\tau_{xy}$** with $\mu=31$ GPa, $\nu=1/3$, $h=6b$, $b=3$ Å. *Answer in MPa, 2 sig figs.*

## Stress Field (Eq. 2.13, Hertzberg)

$$\tau_{xy}^{B \to A}(x) = \frac{\mu(-b)}{2\pi(1-\nu)} \cdot \frac{x(x^2 - h^2)}{(x^2 + h^2)^2}$$

$$\boxed{\tau_{\text{applied}} \geq \left|\min_x \tau_{xy}^{B \to A}(x)\right|}$$
""")
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(start=10, stop=80, value=31, step=1, label="μ (GPa)")
    nu_slider = mo.ui.slider(start=0.10, stop=0.49, value=0.333, step=0.01, label="ν")
    h_slider = mo.ui.slider(start=1, stop=20, value=6, step=0.5, label="h / b")
    mo.md(f"""
## Parameters

| | Slider | Value |
|--|--------|-------|
| μ | {mu_slider} | **{mu_slider.value} GPa** |
| ν | {nu_slider} | **{nu_slider.value:.2f}** |
| h/b | {h_slider} | **{h_slider.value}** |
""")
    return mu_slider, nu_slider, h_slider


@app.cell
def _(mo, np, plt, mu_slider, nu_slider, h_slider):
    b = 3e-10
    mu, nu, h = mu_slider.value * 1e9, nu_slider.value, h_slider.value * b

    x = np.linspace(-30*b, 30*b, 5000)
    tau_MPa = mu * (-b) / (2*np.pi*(1-nu)) * x*(x**2 - h**2) / (x**2 + h**2)**2 / 1e6

    idx = np.argmin(tau_MPa)
    x_crit, tau_min = x[idx], tau_MPa[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x/b, tau_MPa, 'b-', lw=2, label=r'$\tau_{xy}^{B \to A}$')
    ax.plot(x_crit/b, tau_min, 'ro', ms=10, zorder=5,
            label=f'Min = {tau_min:.1f} MPa at x/b = {x_crit/b:.1f}')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set(xlabel='x / b', ylabel=r'$\tau_{xy}^{B \to A}$ [MPa]')
    ax.set_title(f'τ_crit = {abs(tau_min):.1f} MPa  (μ={mu_slider.value}, ν={nu_slider.value:.2f}, h={h_slider.value}b)')
    ax.legend(); ax.grid(True, alpha=0.3)

    mo.vstack([
        mo.md(f"## Result: **{abs(tau_min):.2f} MPa ≈ {abs(tau_min):.1e} MPa** &nbsp; WIRIS: `3.1×10²`"),
        fig
    ])
    return


if __name__ == "__main__":
    app.run()
