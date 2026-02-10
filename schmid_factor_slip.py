# ================================================================
# SCHMID FACTOR & CRYSTAL SLIP - INTERACTIVE VISUALIZATION
# ================================================================
#
# Visualizes how a single crystal deforms under tension:
#   Panel 1 (left):   Cylinder side-view with slip planes and angles
#   Panel 2 (center): Stereographic triangle with P trajectory
#   Panel 3 (right):  Schmid factor + trig breakdown (stacked)
#
# Based on Hertzberg Figs 3.8 & 3.9 - MSE 3261 Lecture 12
# Prof. Anderson: "deck of cards" analogy, crystal rotation,
# zigzag path converging on [T12]
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
    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc, FancyArrowPatch
    return mo, np, plt, mpatches, Arc, FancyArrowPatch


# ======================= CELL 2: TITLE =======================
@app.cell
def _(mo):
    mo.md(r"""
# Crystal Slip & Schmid Factor Evolution

**The setup:** An FCC single crystal cylinder is loaded in uniaxial tension. Slip occurs on the plane with the highest resolved shear stress (Schmid factor). The crystal rotates inside the test frame as it deforms -- like shearing a deck of cards, then straightening it back.

$$\boxed{\tau_R = \sigma\cos\lambda\cos\phi = \sigma \cdot S}$$

| Symbol | Meaning |
|--------|---------|
| $\lambda$ | Angle between tensile axis **P** and slip direction **b** |
| $\phi$ | Angle between tensile axis **P** and slip plane normal **n** |
| $S = \cos\lambda\cos\phi$ | Schmid factor (higher = more resolved shear stress) |

As the crystal slips, **P** rotates toward the active slip direction ($\lambda$ decreases). When the conjugate system's $S$ exceeds the primary, it takes over. This back-and-forth produces a **zigzag path** on the stereographic triangle, converging on $[\bar{1}12]$.

---
""")
    return


# ======================= CELL 3: MATH UTILITIES =======================
@app.cell
def _(np):
    def normalize(v):
        """Normalize a vector to unit length."""
        v = np.array(v, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-15:
            return v
        return v / n

    def stereo_project(uvw):
        """Project direction [uvw] onto [001] stereographic plane.
        r = tan(theta/2), theta = angle from [001]."""
        v = normalize(uvw)
        cos_theta = np.clip(v[2], -1, 1)
        theta = np.arccos(cos_theta)
        r = np.tan(theta / 2)
        az = np.arctan2(v[1], v[0])
        return r * np.cos(az), r * np.sin(az)

    def angle_between(a, b):
        """Angle in radians between two direction vectors."""
        a_n, b_n = normalize(a), normalize(b)
        return np.arccos(np.clip(np.dot(a_n, b_n), -1, 1))

    def schmid_factor(P, slip_dir, slip_normal):
        """Compute Schmid factor and return (S, lambda_rad, phi_rad)."""
        lam = angle_between(P, slip_dir)
        phi = angle_between(P, slip_normal)
        return np.cos(lam) * np.cos(phi), lam, phi

    return normalize, stereo_project, angle_between, schmid_factor


# ======================= CELL 4: TRAJECTORY COMPUTATION =======================
@app.cell
def _(np, normalize, schmid_factor):
    # --- Slip system definitions (Hertzberg Fig 3.8) ---
    # Primary:   (111)[T01]
    # Secondary: (TT1)[0T1]
    n_I  = normalize([1, 1, 1])       # primary slip plane normal
    b_I  = normalize([-1, 0, 1])      # primary slip direction
    n_II = normalize([-1, -1, 1])     # secondary slip plane normal
    b_II = normalize([0, -1, 1])      # secondary slip direction

    # Convergence target
    P_converge = normalize([-1, 1, 2])

    # Starting tensile axis -- chosen to give 2-3 visible oscillations
    P0 = normalize([2, 5, 8])

    # --- Simulate trajectory ---
    def compute_trajectory(P_start, n_steps=400, step_size=0.010):
        trajectory = [normalize(P_start)]
        active_sys = []
        S_I_arr, S_II_arr = [], []
        lam_I_arr, phi_I_arr = [], []
        lam_II_arr, phi_II_arr = [], []

        for _ in range(n_steps):
            P = trajectory[-1]
            s1, l1, p1 = schmid_factor(P, b_I, n_I)
            s2, l2, p2 = schmid_factor(P, b_II, n_II)

            S_I_arr.append(s1)
            S_II_arr.append(s2)
            lam_I_arr.append(l1)
            phi_I_arr.append(p1)
            lam_II_arr.append(l2)
            phi_II_arr.append(p2)

            if s1 >= s2:
                active = 0
                slip_dir = b_I
            else:
                active = 1
                slip_dir = b_II
            active_sys.append(active)

            # Rotate P toward active slip direction along great circle
            proj = np.dot(P, slip_dir) * P
            tangent = slip_dir - proj
            tn = np.linalg.norm(tangent)
            if tn > 1e-10:
                tangent = tangent / tn
            P_new = normalize(P + step_size * tangent)
            trajectory.append(P_new)

        return {
            "traj": np.array(trajectory),
            "active": np.array(active_sys),
            "S_I": np.array(S_I_arr),
            "S_II": np.array(S_II_arr),
            "lam_I": np.array(lam_I_arr),
            "phi_I": np.array(phi_I_arr),
            "lam_II": np.array(lam_II_arr),
            "phi_II": np.array(phi_II_arr),
        }

    sim = compute_trajectory(P0)
    n_total = len(sim["S_I"])

    return (n_I, b_I, n_II, b_II, P0, P_converge,
            sim, n_total, compute_trajectory)


# ======================= CELL 5: SLIDER =======================
@app.cell
def _(mo, n_total):
    step_slider = mo.ui.slider(
        start=0,
        stop=n_total - 1,
        step=1,
        value=0,
        label="Deformation progress",
        full_width=True,
    )
    step_slider
    return (step_slider,)


# ======================= CELL 6: THREE-PANEL VISUALIZATION =======================
@app.cell
def _(mo, np, plt, Arc, stereo_project,
      n_I, b_I, n_II, b_II, P_converge,
      sim, step_slider, n_total):

    step = step_slider.value
    traj = sim["traj"]
    active = sim["active"]
    S_I = sim["S_I"]
    S_II = sim["S_II"]
    lam_I = sim["lam_I"]
    phi_I = sim["phi_I"]
    lam_II = sim["lam_II"]
    phi_II = sim["phi_II"]

    # Colors
    RED = "#CC2200"
    BLUE = "#0055CC"
    GREEN = "#22AA88"

    # Current angles
    cur_active = active[min(step, len(active) - 1)]
    cur_lam = lam_I[min(step, len(lam_I) - 1)] if cur_active == 0 else lam_II[min(step, len(lam_II) - 1)]
    cur_phi = phi_I[min(step, len(phi_I) - 1)] if cur_active == 0 else phi_II[min(step, len(phi_II) - 1)]

    # ============================================================
    #   BUILD THE FIGURE
    # ============================================================
    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.1, 1],
                          hspace=0.38, wspace=0.30,
                          left=0.04, right=0.97, top=0.92, bottom=0.08)
    ax_cyl = fig.add_subplot(gs[:, 0])       # left: cylinder (spans both rows)
    ax_stereo = fig.add_subplot(gs[:, 1])    # center: stereographic triangle
    ax_schmid = fig.add_subplot(gs[0, 2])    # top-right: Schmid factor
    ax_trig = fig.add_subplot(gs[1, 2])      # bottom-right: cos(lam), cos(phi)

    # ──────────────────────────────────────────────────────────────
    #  PANEL 1: CYLINDER WITH SLIP PLANES  (ax_cyl)
    # ──────────────────────────────────────────────────────────────
    ax_cyl.set_xlim(-2.5, 2.5)
    ax_cyl.set_ylim(-0.5, 6.5)
    ax_cyl.set_aspect("equal")
    for sp in ax_cyl.spines.values():
        sp.set_visible(False)
    ax_cyl.set_xticks([])
    ax_cyl.set_yticks([])

    # Progress fraction
    frac = step / max(n_total - 1, 1)

    # Cylinder dimensions
    cyl_w = 1.4
    cyl_h0 = 4.0
    # Crystal elongates as it deforms
    elongation = 1.0 + 0.40 * frac
    cyl_h = cyl_h0 * elongation
    # Thinning (constant volume approximation)
    w_factor = 1.0 / np.sqrt(elongation)
    cyl_w_cur = cyl_w * w_factor

    # Cylinder bottom-left corner (centered on x=0)
    cx = -cyl_w_cur / 2
    cy = 3.0 - cyl_h / 2  # center at y=3

    # Active slip plane angle (lambda measured from tensile axis = vertical)
    # As deformation progresses, lambda decreases
    lam_deg = np.degrees(cur_lam)
    phi_deg = np.degrees(cur_phi)
    # Slip plane angle from horizontal = 90 - phi (phi is angle from normal to tensile axis)
    slip_angle_deg = 90.0 - phi_deg  # angle the slip plane makes with horizontal

    # Shear offset: increases with deformation
    max_shear = 0.8
    shear = max_shear * frac

    # Draw cylinder outline
    rect_x = [cx, cx + cyl_w_cur, cx + cyl_w_cur, cx, cx]
    rect_y = [cy, cy, cy + cyl_h, cy + cyl_h, cy]
    ax_cyl.fill(rect_x, rect_y, color="#E8E8E8", ec="black", lw=2, zorder=2)

    # Draw slip plane lines inside the cylinder
    n_lines = 10
    slip_color = RED if cur_active == 0 else BLUE
    slip_angle_rad = np.radians(slip_angle_deg)
    for i in range(n_lines + 1):
        # Evenly spaced along height
        y_base = cy + (i / n_lines) * cyl_h
        # Shear offset increases linearly with height
        x_offset = shear * (i / n_lines - 0.5)

        # Line across the cylinder at this height, at the slip angle
        # The line extends across the cylinder width at the angle
        half_span = cyl_w_cur / 2
        dx = half_span
        dy = dx * np.tan(slip_angle_rad)

        x0 = cx + x_offset
        y0 = y_base - dy * 0.5
        x1 = cx + cyl_w_cur + x_offset
        y1 = y_base + dy * 0.5

        # Clip to cylinder bounds
        y0c = np.clip(y0, cy, cy + cyl_h)
        y1c = np.clip(y1, cy, cy + cyl_h)
        if abs(y1 - y0) > 1e-8:
            t0 = (y0c - y0) / (y1 - y0)
            t1 = (y1c - y0) / (y1 - y0)
            x0c = x0 + t0 * (x1 - x0)
            x1c = x0 + t1 * (x1 - x0)
        else:
            x0c, x1c = x0, x1

        ax_cyl.plot([x0c, x1c], [y0c, y1c], color=slip_color,
                    lw=1.2, alpha=0.5, zorder=3)

    # Redraw outline on top of slip lines
    ax_cyl.plot(rect_x, rect_y, color="black", lw=2, zorder=4)

    # --- Tension arrows ---
    arrow_len = 0.7
    # Top arrow (pulling up)
    ax_cyl.annotate("", xy=(0, cy + cyl_h + arrow_len),
                    xytext=(0, cy + cyl_h),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5),
                    zorder=5)
    ax_cyl.text(0, cy + cyl_h + arrow_len + 0.15, r"$\sigma$",
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    # Bottom arrow (pulling down)
    ax_cyl.annotate("", xy=(0, cy - arrow_len),
                    xytext=(0, cy),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5),
                    zorder=5)
    ax_cyl.text(0, cy - arrow_len - 0.15, r"$\sigma$",
                ha="center", va="top", fontsize=16, fontweight="bold")

    # --- Angle annotations ---
    # Draw lambda arc (angle between tensile axis [vertical] and slip direction)
    # Show from top of cylinder, measuring lambda from vertical
    arc_center_x = 0
    arc_center_y = cy + cyl_h * 0.5
    arc_r = 1.0

    # Lambda: angle from vertical (tensile axis) to slip direction
    # Measured as arc from 90 deg (vertical up) going toward slip direction
    lam_start = 90.0  # vertical
    lam_end = 90.0 - lam_deg  # slip direction angle from horizontal
    if lam_deg > 2:
        arc_lam = Arc((arc_center_x, arc_center_y), 2 * arc_r, 2 * arc_r,
                      angle=0, theta1=min(lam_start, lam_end),
                      theta2=max(lam_start, lam_end),
                      color=RED, lw=2.0, zorder=6)
        ax_cyl.add_patch(arc_lam)
        mid_angle = np.radians((lam_start + lam_end) / 2)
        ax_cyl.text(arc_center_x + (arc_r + 0.3) * np.cos(mid_angle),
                    arc_center_y + (arc_r + 0.3) * np.sin(mid_angle),
                    rf"$\lambda$={lam_deg:.0f}$\degree$",
                    fontsize=11, color=RED, fontweight="bold",
                    ha="center", va="center", zorder=7)

    # Phi: angle from vertical to slip plane normal
    phi_start = 90.0
    phi_end = 90.0 + phi_deg  # normal tilts to the other side
    arc_r2 = 0.7
    if phi_deg > 2:
        arc_phi = Arc((arc_center_x, arc_center_y), 2 * arc_r2, 2 * arc_r2,
                      angle=0, theta1=min(phi_start, phi_end),
                      theta2=max(phi_start, phi_end),
                      color=BLUE, lw=2.0, ls="--", zorder=6)
        ax_cyl.add_patch(arc_phi)
        mid_angle2 = np.radians((phi_start + phi_end) / 2)
        ax_cyl.text(arc_center_x + (arc_r2 + 0.35) * np.cos(mid_angle2),
                    arc_center_y + (arc_r2 + 0.35) * np.sin(mid_angle2),
                    rf"$\phi$={phi_deg:.0f}$\degree$",
                    fontsize=11, color=BLUE, fontweight="bold",
                    ha="center", va="center", zorder=7)

    # Label the active system
    sys_label = "Primary (111)[$\\bar{1}$01]" if cur_active == 0 else "Secondary ($\\bar{1}\\bar{1}$1)[0$\\bar{1}$1]"
    sys_color = RED if cur_active == 0 else BLUE
    ax_cyl.text(0, cy - 0.05, sys_label, ha="center", va="top",
                fontsize=10, color=sys_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=sys_color, alpha=0.9),
                zorder=8)

    # Title
    ax_cyl.set_title("Single Crystal in Tension", fontsize=12, fontweight="bold")

    # Text showing S value
    cur_S = S_I[min(step, len(S_I) - 1)] if cur_active == 0 else S_II[min(step, len(S_II) - 1)]
    ax_cyl.text(0, cy + cyl_h + arrow_len + 0.75,
                f"S = cos({lam_deg:.0f})cos({phi_deg:.0f}) = {cur_S:.3f}",
                ha="center", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFDD",
                          edgecolor="#999", alpha=0.9), zorder=8)

    # ──────────────────────────────────────────────────────────────
    #  PANEL 2: STEREOGRAPHIC TRIANGLE  (ax_stereo)
    # ──────────────────────────────────────────────────────────────
    def draw_arc(ax, v1, v2, n_pts=80, **kwargs):
        """Draw great circle arc on stereographic projection."""
        pts = []
        for t in np.linspace(0, 1, n_pts):
            v = (1 - t) * np.array(v1, float) + t * np.array(v2, float)
            vn = np.linalg.norm(v)
            if vn > 1e-10:
                v = v / vn
            pts.append(stereo_project(v))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], **kwargs)

    # Standard triangle boundary: 001 - 011 - 111
    draw_arc(ax_stereo, [0,0,1], [0,1,1], color="#444", lw=1.5)
    draw_arc(ax_stereo, [0,1,1], [1,1,1], color="#444", lw=1.5)
    draw_arc(ax_stereo, [1,1,1], [0,0,1], color="#444", lw=1.5)

    # Extended context arcs
    draw_arc(ax_stereo, [0,0,1], [1,0,1], color="#bbb", lw=1, ls="--")
    draw_arc(ax_stereo, [1,0,1], [1,1,1], color="#bbb", lw=1, ls="--")

    # Equal Schmid factor boundary (great circle from 001 toward T11)
    draw_arc(ax_stereo, [0,0,1], [-1,1,1], color=GREEN, lw=2, alpha=0.6,
             label="Equal $S$ boundary")

    # Label key directions
    dir_labels = {
        "001": [0, 0, 1],
        "011": [0, 1, 1],
        "111": [1, 1, 1],
        "101": [1, 0, 1],
    }
    for name, uvw in dir_labels.items():
        sx, sy = stereo_project(uvw)
        ax_stereo.plot(sx, sy, "ko", ms=4, zorder=5)
        ax_stereo.annotate(name, (sx, sy), textcoords="offset points",
                           xytext=(6, 6), fontsize=9, fontweight="bold")

    # Slip direction markers
    sx, sy = stereo_project(b_I)
    ax_stereo.plot(sx, sy, "s", color=RED, ms=10, zorder=6,
                   label=r"$\mathbf{b}_I$ [$\bar{1}$01]")
    sx, sy = stereo_project(b_II)
    ax_stereo.plot(sx, sy, "D", color=BLUE, ms=10, zorder=6,
                   label=r"$\mathbf{b}_{II}$ [0$\bar{1}$1]")

    # Convergence point [T12]
    sx, sy = stereo_project(P_converge)
    ax_stereo.plot(sx, sy, "*", color=GREEN, ms=14, zorder=6,
                   label=r"[$\bar{1}$12] target")
    ax_stereo.annotate(r"$[\bar{1}12]$", (sx, sy), textcoords="offset points",
                       xytext=(8, -10), fontsize=10, fontweight="bold", color=GREEN)

    # Draw trajectory UP TO current step (builds incrementally)
    for i in range(min(step, len(active))):
        x0, y0 = stereo_project(traj[i])
        x1, y1 = stereo_project(traj[i + 1])
        c = RED if active[i] == 0 else BLUE
        ax_stereo.plot([x0, x1], [y0, y1], "-", color=c, lw=1.8, alpha=0.7)

    # Starting P0
    px0, py0 = stereo_project(traj[0])
    ax_stereo.plot(px0, py0, "o", color="black", ms=8, zorder=9)
    ax_stereo.annotate(r"$P_0$", (px0, py0), textcoords="offset points",
                       xytext=(-14, 8), fontsize=10, color="#666")

    # Current P position
    if step < len(traj):
        px, py = stereo_project(traj[step])
        ax_stereo.plot(px, py, "o", color="black", ms=10, zorder=10)
        ax_stereo.plot(px, py, "o", color="#FFD700", ms=7, zorder=11)
        ax_stereo.annotate("P", (px, py), textcoords="offset points",
                           xytext=(-12, -12), fontsize=12, fontweight="bold",
                           color="black")

    ax_stereo.set_xlim(-0.55, 0.55)
    ax_stereo.set_ylim(-0.15, 0.65)
    ax_stereo.set_aspect("equal")
    ax_stereo.set_title("Stereographic Triangle [001]", fontsize=12, fontweight="bold")
    ax_stereo.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_stereo.grid(True, alpha=0.15)
    for sp in ax_stereo.spines.values():
        sp.set_visible(False)
    ax_stereo.set_xticks([])
    ax_stereo.set_yticks([])

    # ──────────────────────────────────────────────────────────────
    #  PANEL 3 (top): SCHMID FACTOR vs STEP  (ax_schmid)
    # ──────────────────────────────────────────────────────────────
    steps_arr = np.arange(len(S_I))

    ax_schmid.plot(steps_arr, S_I, "-", color=RED, lw=2, label=r"$S_I$ (primary)")
    ax_schmid.plot(steps_arr, S_II, "-", color=BLUE, lw=2, label=r"$S_{II}$ (secondary)")

    # Shade which system is active (very faint)
    for i in range(len(active) - 1):
        c = RED if active[i] == 0 else BLUE
        ax_schmid.axvspan(i, i + 1, alpha=0.05, color=c)

    # Current step cursor
    if step < len(S_I):
        ax_schmid.axvline(step, color="black", lw=1.5, ls="--", alpha=0.7)
        ax_schmid.plot(step, S_I[step], "o", color=RED, ms=7, zorder=10)
        ax_schmid.plot(step, S_II[step], "o", color=BLUE, ms=7, zorder=10)
        ax_schmid.annotate(f"$S_I$ = {S_I[step]:.3f}", (step, S_I[step]),
                           textcoords="offset points", xytext=(8, 6),
                           fontsize=8, color=RED, fontweight="bold")
        ax_schmid.annotate(f"$S_{{II}}$ = {S_II[step]:.3f}", (step, S_II[step]),
                           textcoords="offset points", xytext=(8, -10),
                           fontsize=8, color=BLUE, fontweight="bold")

    ax_schmid.set_ylabel(r"$S = \cos\lambda\cos\phi$", fontsize=10)
    ax_schmid.set_title("Schmid Factor", fontsize=11, fontweight="bold")
    ax_schmid.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax_schmid.grid(True, alpha=0.3)
    ax_schmid.grid(True, which="minor", alpha=0.1)
    ax_schmid.minorticks_on()
    ax_schmid.set_xticklabels([])  # share x with bottom
    for sp in ax_schmid.spines.values():
        sp.set_visible(False)

    # ──────────────────────────────────────────────────────────────
    #  PANEL 3 (bottom): cos(lambda) and cos(phi)  (ax_trig)
    # ──────────────────────────────────────────────────────────────
    ax_trig.plot(steps_arr, np.cos(lam_I), "-", color=RED, lw=2,
                 label=r"$\cos\lambda_I$")
    ax_trig.plot(steps_arr, np.cos(phi_I), "--", color=RED, lw=1.5,
                 alpha=0.7, label=r"$\cos\phi_I$")
    ax_trig.plot(steps_arr, np.cos(lam_II), "-", color=BLUE, lw=2,
                 label=r"$\cos\lambda_{II}$")
    ax_trig.plot(steps_arr, np.cos(phi_II), "--", color=BLUE, lw=1.5,
                 alpha=0.7, label=r"$\cos\phi_{II}$")

    if step < len(lam_I):
        ax_trig.axvline(step, color="black", lw=1.5, ls="--", alpha=0.7)
        # Angle readout box
        ax_trig.annotate(
            rf"$\lambda_I$={np.degrees(lam_I[step]):.1f}$\degree$  "
            rf"$\phi_I$={np.degrees(phi_I[step]):.1f}$\degree$",
            xy=(0.02, 0.96), xycoords="axes fraction", fontsize=8,
            color=RED, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
        ax_trig.annotate(
            rf"$\lambda_{{II}}$={np.degrees(lam_II[step]):.1f}$\degree$  "
            rf"$\phi_{{II}}$={np.degrees(phi_II[step]):.1f}$\degree$",
            xy=(0.02, 0.82), xycoords="axes fraction", fontsize=8,
            color=BLUE, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    ax_trig.set_xlabel("Deformation step", fontsize=10)
    ax_trig.set_ylabel(r"$\cos\lambda$, $\cos\phi$", fontsize=10)
    ax_trig.set_title(r"$\cos\lambda$ (solid) vs $\cos\phi$ (dashed)", fontsize=11,
                       fontweight="bold")
    ax_trig.legend(fontsize=7, loc="center right", framealpha=0.9)
    ax_trig.grid(True, alpha=0.3)
    ax_trig.grid(True, which="minor", alpha=0.1)
    ax_trig.minorticks_on()
    for sp in ax_trig.spines.values():
        sp.set_visible(False)

    # ──────────────────────────────────────────────────────────────
    #  SUPERTITLE
    # ──────────────────────────────────────────────────────────────
    status = "Primary (I)" if cur_active == 0 else "Secondary (II)"
    fig.suptitle(
        f"Step {step}/{n_total - 1}  |  Active: {status}  |  "
        f"S = {cur_S:.3f}",
        fontsize=13, fontweight="bold", y=0.98)

    mo.vstack([fig])
    return


# ======================= CELL 7: PHYSICS EXPLANATION =======================
@app.cell
def _(mo):
    mo.md(r"""
---

## What is happening physically?

**1. The deck-of-cards effect.** When resolved shear stress exceeds $\tau_{\text{crss}}$ on the active slip system, the crystal shears along that plane. The slip planes slide over each other like cards in a tilted deck.

**2. The test frame re-straightens.** The testing machine grips force the sample to stay vertical. This constraint means the crystal lattice *rotates* relative to the tensile axis. The angle $\lambda$ (between tensile axis and slip direction) *decreases*.

**3. The handoff.** As $\lambda$ shrinks, the primary system's Schmid factor changes. Eventually the conjugate system has a higher $S$, and slip switches. Now $P$ moves toward the *new* slip direction.

**4. Convergence on $[\bar{1}12]$.** Each switch dampens the oscillation. Both systems reach equal $S$ at the $[\bar{1}12]$ orientation, which is the stable endpoint for double slip.

---

| Observation | Explanation |
|-------------|-------------|
| $P$ moves toward active slip direction | Crystal rotation decreases $\lambda$ |
| Schmid factor changes during deformation | Both $\lambda$ and $\phi$ evolve as $P$ rotates |
| Systems alternate (zigzag on triangle) | Crossing the equal-$S$ boundary triggers handoff |
| Path converges on $\langle 112 \rangle$ | Stable double-slip orientation where $S_I = S_{II}$ |
| Texture develops in polycrystals | Many grains undergo this rotation, developing preferred orientation |

---

*Based on Hertzberg Figs 3.8 & 3.9 -- MSE 3261 Lecture 12 (Prof. Anderson, Spring 2026)*
""")
    return


if __name__ == "__main__":
    app.run()
