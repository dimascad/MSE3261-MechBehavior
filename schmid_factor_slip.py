# ================================================================
# SCHMID FACTOR & CRYSTAL SLIP - INTERACTIVE STEREOGRAPHIC PROJECTION
# ================================================================
#
# Visualizes how the tensile axis P moves on the stereographic
# triangle during plastic deformation, alternating between
# primary and secondary (conjugate) slip systems.
#
# PANEL 1: Stereographic triangle with P trajectory
# PANEL 2: Schmid factor vs deformation step
# PANEL 3: Individual cos(lambda) and cos(phi) breakdown
#
# Based on Hertzberg Figs 3.8 & 3.9 — MSE 3261 Lecture 12
# ================================================================

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


# ======================= CELL 1: IMPORTS =======================
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    return mo, np, plt, FancyArrowPatch


# ======================= CELL 2: TITLE =======================
@app.cell
def _(mo):
    mo.md(r"""
# Schmid Factor Evolution During Crystal Slip

*Interactive stereographic projection showing primary/secondary slip system alternation*

---

**The Schmid factor** $S = \cos\lambda\,\cos\phi$ determines which slip system is active.
During tensile deformation, the tensile axis **P** rotates toward the active slip direction,
changing $\lambda$ and $\phi$ — and eventually the *conjugate* system takes over.

This back-and-forth produces the classic **zigzag path** toward $\langle 112 \rangle$ (Hertzberg Fig. 3.9).

---
""")
    return


# ======================= CELL 3: MATH UTILITIES =======================
@app.cell
def _(np):
    def normalize(v):
        """Normalize a vector to unit length."""
        v = np.array(v, dtype=float)
        return v / np.linalg.norm(v)

    def stereo_project(uvw):
        """Project a unit-direction [uvw] onto the [001] stereographic plane.
        Uses r = tan(theta/2) where theta = angle from [001]."""
        v = normalize(uvw)
        # Colatitude from [001]
        cos_theta = v[2]
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        r = np.tan(theta / 2)
        # Azimuth in xy plane
        phi = np.arctan2(v[1], v[0])
        return r * np.cos(phi), r * np.sin(phi)

    def angle_between(a, b):
        """Angle in degrees between two direction vectors."""
        a, b = normalize(a), normalize(b)
        return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))

    def schmid_factor(P, slip_dir, slip_normal):
        """Compute Schmid factor S = cos(lambda) * cos(phi)."""
        lam = np.radians(angle_between(P, slip_dir))
        phi = np.radians(angle_between(P, slip_normal))
        return np.cos(lam) * np.cos(phi), np.degrees(lam), np.degrees(phi)

    return normalize, stereo_project, angle_between, schmid_factor


# ======================= CELL 4: CRYSTAL SLIP SYSTEM DEFINITIONS =======================
@app.cell
def _(mo):
    mo.md(r"""
## The Setup: FCC Single Crystal Under Tension

For an FCC crystal with tensile axis in the **standard stereographic triangle** (001–011–111):

| System | Slip Plane Normal $\mathbf{n}$ | Slip Direction $\mathbf{b}$ | Role |
|--------|-------------------------------|----------------------------|------|
| **Primary (I)** | $(111)$ | $[\bar{1}01]$ | Highest initial Schmid factor |
| **Secondary (II)** | $(\bar{1}\bar{1}1)$ | $[0\bar{1}1]$ | Takes over after P crosses boundary |

**During deformation:**
- Primary slip → $\lambda$ decreases → P moves toward $[\bar{1}01]$
- When P crosses the **line of equal Schmid factor**, secondary takes over
- Secondary slip → P moves toward $[0\bar{1}1]$
- This **zigzag converges** to $[\bar{1}12]$

---
""")
    return


# ======================= CELL 5: TRAJECTORY COMPUTATION =======================
@app.cell
def _(np, normalize, angle_between, schmid_factor):
    # Define the slip systems (Hertzberg Fig 3.8)
    # Primary: (111)[T01]    Secondary: (TT1)[0T1]
    n_I = normalize([1, 1, 1])       # primary slip plane normal
    b_I = normalize([-1, 0, 1])      # primary slip direction
    n_II = normalize([-1, -1, 1])    # secondary slip plane normal
    b_II = normalize([0, -1, 1])     # secondary slip direction

    # Starting tensile axis P0 — inside the standard triangle
    # Choosing a point that gives a nice trajectory
    P0 = normalize([1, 3, 5])

    def compute_trajectory(P0, n_steps=300, step_size=0.012):
        """Simulate tensile axis rotation during slip.

        At each step:
        - Compute Schmid factor for both systems
        - Active system = higher S
        - P rotates toward the active slip direction
        """
        trajectory = [normalize(P0)]
        active_system = []  # 0=primary, 1=secondary
        S_I_list, S_II_list = [], []
        lam_I_list, phi_I_list = [], []
        lam_II_list, phi_II_list = [], []

        for i in range(n_steps):
            P = trajectory[-1]

            # Schmid factors
            s1, l1, p1 = schmid_factor(P, b_I, n_I)
            s2, l2, p2 = schmid_factor(P, b_II, n_II)

            S_I_list.append(s1)
            S_II_list.append(s2)
            lam_I_list.append(l1)
            phi_I_list.append(p1)
            lam_II_list.append(l2)
            phi_II_list.append(p2)

            # Which system is active?
            if s1 >= s2:
                active = 0
                slip_dir = b_I
            else:
                active = 1
                slip_dir = b_II
            active_system.append(active)

            # Rotate P toward active slip direction
            # P_new = normalize(P + step_size * (slip_dir - P*dot(P,slip_dir)))
            # This moves P along the great circle toward slip_dir
            proj = np.dot(P, slip_dir) * P
            tangent = slip_dir - proj
            if np.linalg.norm(tangent) > 1e-10:
                tangent = tangent / np.linalg.norm(tangent)
            P_new = normalize(P + step_size * tangent)
            trajectory.append(P_new)

        return (np.array(trajectory), np.array(active_system),
                np.array(S_I_list), np.array(S_II_list),
                np.array(lam_I_list), np.array(phi_I_list),
                np.array(lam_II_list), np.array(phi_II_list))

    traj, active, S_I, S_II, lam_I, phi_I, lam_II, phi_II = compute_trajectory(P0)

    # Also compute convergence point [T12]
    P_converge = normalize([-1, 1, 2])

    return (n_I, b_I, n_II, b_II, P0, traj, active,
            S_I, S_II, lam_I, phi_I, lam_II, phi_II, P_converge,
            compute_trajectory)


# ======================= CELL 6: SLIDER =======================
@app.cell
def _(mo, S_I):
    n_total = len(S_I)
    step_slider = mo.ui.slider(
        start=0,
        stop=n_total - 1,
        step=1,
        value=0,
        label="Deformation step",
        full_width=True
    )
    step_slider
    return step_slider, n_total


# ======================= CELL 7: THREE-PANEL VISUALIZATION =======================
@app.cell
def _(mo, np, plt, stereo_project,
      n_I, b_I, n_II, b_II, traj, active,
      S_I, S_II, lam_I, phi_I, lam_II, phi_II,
      P_converge, step_slider, n_total):

    step = step_slider.value

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.32)

    # ─── PANEL 1: Stereographic Triangle ───
    ax1 = axes[0]

    # Draw the standard triangle boundary (001-011-111)
    corners = {
        '001': [0, 0, 1],
        '011': [0, 1, 1],
        '111': [1, 1, 1],
        '101': [1, 0, 1],
    }
    # Triangle edges
    def draw_great_circle_arc(ax, v1, v2, n_pts=80, **kwargs):
        """Draw arc between two directions on stereographic projection."""
        from numpy import linspace
        pts = []
        for t in linspace(0, 1, n_pts):
            v = (1 - t) * np.array(v1, float) + t * np.array(v2, float)
            v = v / np.linalg.norm(v)
            pts.append(stereo_project(v))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], **kwargs)

    # Triangle: 001-011, 011-111, 111-001
    draw_great_circle_arc(ax1, [0,0,1], [0,1,1], color='#444', lw=1.5, ls='-')
    draw_great_circle_arc(ax1, [0,1,1], [1,1,1], color='#444', lw=1.5, ls='-')
    draw_great_circle_arc(ax1, [1,1,1], [0,0,1], color='#444', lw=1.5, ls='-')

    # Extended triangle for context
    draw_great_circle_arc(ax1, [0,0,1], [1,0,1], color='#bbb', lw=1, ls='--')
    draw_great_circle_arc(ax1, [1,0,1], [1,1,1], color='#bbb', lw=1, ls='--')

    # Line of equal Schmid factor (001-T11 boundary → passes through T12)
    draw_great_circle_arc(ax1, [0,0,1], [-1,1,1], color='#22AA88', lw=2, ls='-', alpha=0.6)

    # Label key directions
    labels = {
        '001': [0, 0, 1], '011': [0, 1, 1], '111': [1, 1, 1],
        '101': [1, 0, 1], r'$\bar{1}$01': [-1, 0, 1], r'$\bar{1}\bar{1}$1': [-1, -1, 1],
        r'$\bar{1}$12': [-1, 1, 2],
    }
    for name, uvw in labels.items():
        sx, sy = stereo_project(uvw)
        ax1.plot(sx, sy, 'ko', ms=4, zorder=5)
        ax1.annotate(name, (sx, sy), textcoords="offset points",
                     xytext=(6, 6), fontsize=9, fontweight='bold')

    # Mark slip directions with colored symbols
    sx, sy = stereo_project(b_I)
    ax1.plot(sx, sy, 's', color='#CC2200', ms=10, zorder=6, label=r'$\mathbf{b}_I$ [$\bar{1}$01]')
    sx, sy = stereo_project(b_II)
    ax1.plot(sx, sy, 'D', color='#0055CC', ms=10, zorder=6, label=r'$\mathbf{b}_{II}$ [0$\bar{1}$1]')

    # Convergence point
    sx, sy = stereo_project(P_converge)
    ax1.plot(sx, sy, '*', color='#22AA88', ms=14, zorder=6, label=r'$[\bar{1}12]$ target')

    # Draw trajectory up to current step
    for i in range(min(step, len(active))):
        x0, y0 = stereo_project(traj[i])
        x1, y1 = stereo_project(traj[i + 1])
        color = '#CC2200' if active[i] == 0 else '#0055CC'
        ax1.plot([x0, x1], [y0, y1], '-', color=color, lw=1.8, alpha=0.7)

    # Current P position
    if step < len(traj):
        px, py = stereo_project(traj[step])
        ax1.plot(px, py, 'o', color='black', ms=10, zorder=10)
        ax1.plot(px, py, 'o', color='#FFD700', ms=7, zorder=11)
        ax1.annotate('P', (px, py), textcoords="offset points",
                     xytext=(-12, -12), fontsize=12, fontweight='bold', color='black')

    # Starting P
    px0, py0 = stereo_project(traj[0])
    ax1.plot(px0, py0, 'o', color='black', ms=8, zorder=9)
    ax1.annotate(r'$P_0$', (px0, py0), textcoords="offset points",
                 xytext=(-14, 8), fontsize=10, color='#666')

    ax1.set_xlim(-0.55, 0.55)
    ax1.set_ylim(-0.15, 0.65)
    ax1.set_aspect('equal')
    ax1.set_title('Stereographic Triangle [001]', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.15)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # ─── PANEL 2: Schmid Factor vs Step ───
    ax2 = axes[1]
    steps_arr = np.arange(len(S_I))

    ax2.plot(steps_arr, S_I, '-', color='#CC2200', lw=2, label=r'$S_I$ (primary)')
    ax2.plot(steps_arr, S_II, '-', color='#0055CC', lw=2, label=r'$S_{II}$ (secondary)')

    # Fill regions showing which is active
    for i in range(len(active) - 1):
        color = '#CC2200' if active[i] == 0 else '#0055CC'
        ax2.axvspan(i, i+1, alpha=0.06, color=color)

    # Current step marker
    if step < len(S_I):
        ax2.axvline(step, color='black', lw=1.5, ls='--', alpha=0.7)
        ax2.plot(step, S_I[step], 'o', color='#CC2200', ms=8, zorder=10)
        ax2.plot(step, S_II[step], 'o', color='#0055CC', ms=8, zorder=10)
        ax2.annotate(f'S_I = {S_I[step]:.3f}', (step, S_I[step]),
                     textcoords="offset points", xytext=(10, 8), fontsize=9, color='#CC2200')
        ax2.annotate(f'S_II = {S_II[step]:.3f}', (step, S_II[step]),
                     textcoords="offset points", xytext=(10, -12), fontsize=9, color='#0055CC')

    ax2.set_xlabel('Deformation step', fontsize=11)
    ax2.set_ylabel(r'Schmid factor $S = \cos\lambda\,\cos\phi$', fontsize=11)
    ax2.set_title('Schmid Factor Evolution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.grid(True, which='minor', alpha=0.1)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # ─── PANEL 3: cos(lambda) and cos(phi) breakdown ───
    ax3 = axes[2]

    ax3.plot(steps_arr, np.cos(np.radians(lam_I)), '-', color='#CC2200', lw=2,
             label=r'$\cos\lambda_I$')
    ax3.plot(steps_arr, np.cos(np.radians(phi_I)), '--', color='#CC2200', lw=1.5,
             label=r'$\cos\phi_I$', alpha=0.7)
    ax3.plot(steps_arr, np.cos(np.radians(lam_II)), '-', color='#0055CC', lw=2,
             label=r'$\cos\lambda_{II}$')
    ax3.plot(steps_arr, np.cos(np.radians(phi_II)), '--', color='#0055CC', lw=1.5,
             label=r'$\cos\phi_{II}$', alpha=0.7)

    if step < len(lam_I):
        ax3.axvline(step, color='black', lw=1.5, ls='--', alpha=0.7)
        # Annotations for current values
        ax3.annotate(
            f'$\\lambda_I$={lam_I[step]:.1f}°  $\\phi_I$={phi_I[step]:.1f}°',
            xy=(0.02, 0.96), xycoords='axes fraction', fontsize=9,
            color='#CC2200', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax3.annotate(
            f'$\\lambda_{{II}}$={lam_II[step]:.1f}°  $\\phi_{{II}}$={phi_II[step]:.1f}°',
            xy=(0.02, 0.85), xycoords='axes fraction', fontsize=9,
            color='#0055CC', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax3.set_xlabel('Deformation step', fontsize=11)
    ax3.set_ylabel(r'$\cos\lambda$, $\cos\phi$', fontsize=11)
    ax3.set_title(r'Trig Breakdown: $\cos\lambda$ (solid) vs $\cos\phi$ (dashed)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='center right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.grid(True, which='minor', alpha=0.1)
    ax3.minorticks_on()
    for spine in ax3.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f'Step {step}/{n_total-1}  —  '
        + ('Primary (I) active' if step < len(active) and active[step] == 0 else 'Secondary (II) active'),
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    mo.vstack([fig])
    return


# ======================= CELL 8: KEY TAKEAWAYS =======================
@app.cell
def _(mo):
    mo.md(r"""
---

## Key Takeaways

| Observation | Why? |
|-------------|------|
| **P moves toward active slip direction** | $\lambda$ decreases during slip — the crystal physically rotates |
| **S changes during deformation** | Both $\lambda$ and $\phi$ change as P moves |
| **Systems alternate** | When P crosses the equal-S boundary, the conjugate system has higher S |
| **Convergence to $\langle 112 \rangle$** | The zigzag dampens — both systems have equal S at $[\bar{1}12]$ |
| **Texture develops** | Many grains undergoing this → preferred orientation along $\langle 112 \rangle$ |

### The Unit Circle Connection

The stereographic projection maps a **unit sphere** onto a plane via $r = \tan(\theta/2)$.
Every crystallographic direction is a **unit vector**, and the Schmid factor is a product of
**dot products** (cosines) between unit vectors — pure unit-circle trig:

$$S = \cos\lambda\,\cos\phi = \frac{\mathbf{P}\cdot\mathbf{b}}{|\mathbf{P}||\mathbf{b}|} \cdot \frac{\mathbf{P}\cdot\mathbf{n}}{|\mathbf{P}||\mathbf{n}|}$$

The stereographic plot is just a **map** that lets you visualize where P sits relative to b and n on that sphere.

---
*Based on Hertzberg Figs 3.8 & 3.9 — MSE 3261 Lecture 12 (PM Anderson, Feb 2026)*
""")
    return


if __name__ == "__main__":
    app.run()
