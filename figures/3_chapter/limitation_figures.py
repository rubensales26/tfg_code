"""Generate the Chapter 4 gradient-descent example figures, all in one visual
profile: viridis surfaces/contours, crimson gradient-descent iterates,
black-star critical points, serif fonts.

All trajectories are produced by *deterministic* full-gradient descent
    w_{k+1} = w_k - h * grad F(w_k),
no stochasticity involved.

Figures produced (PDF + PNG) in this directory:
    local_minima      -- non-convexity: sensitivity to initialization
    saddle_point      -- stalling at / escaping a saddle
    narrow_valley     -- oscillation in an ill-conditioned valley
    non_smoothness    -- discontinuous (Heaviside) vs. non-smooth (ReLU) loss
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "font.size": 10,
})

OUT = Path(__file__).resolve().parent

# shared palette ----------------------------------------------------------
CURVE = plt.cm.viridis(0.25)    # dark blue-purple: primary function curve
ACCENT = plt.cm.viridis(0.62)   # teal-green: secondary curve / trajectory
CRIMSON = "crimson"
GREY = "0.4"


def gd(w0, grad, h, n):
    """Deterministic gradient descent for a general gradient function;
    returns the array of iterates (works in any dimension)."""
    ws = [np.asarray(w0, dtype=float)]
    w = ws[0].copy()
    for _ in range(n):
        w = w - h * grad(w)
        ws.append(w.copy())
    return np.array(ws)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pgf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    print("saved", name)


# =========================================================================
# 1. Non-convexity / local minima
# =========================================================================
def F1(w):
    return 0.25 * w**4 - (1 / 3) * w**3 - w**2


def F1p(w):
    return w**3 - w**2 - 2 * w


H1, N1 = 0.05, 50
grid = np.linspace(-2.2, 3.0, 400)

traj_left = gd(-0.5, F1p, H1, N1)    # -> local minimum at -1
traj_right = gd(0.5, F1p, H1, N1)    # -> global minimum at  2

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.0))

# (a) objective with critical points
axA.plot(grid, F1(grid), color=CURVE, linewidth=1.5, alpha=0.6, zorder=1)
axA.plot(2, F1(2), "o", color="black", markersize=7, label="global minimum")
axA.plot(-1, F1(-1), "o", color=ACCENT, markersize=7, label="local minimum")
axA.set_xlabel(r"$w$")
axA.set_ylabel(r"$F(w)$")
axA.set_title("(a) Objective function")
axA.legend(loc="upper center", fontsize=8, framealpha=0.9)

# (b) two trajectories
axB.plot(grid, F1(grid), color=CURVE, linewidth=1.5, alpha=0.6, zorder=1)
axB.plot(traj_left, F1(traj_left), "-o", color=CRIMSON, markersize=3,
         linewidth=1.0, zorder=3, label=r"$w_0=-0.5")
axB.plot(traj_right, F1(traj_right), "-o", color=ACCENT, markersize=3,
         linewidth=1.0, zorder=3, label=r"$w_0=0.5")
axB.plot([-0.5, 0.5], [F1(-0.5), F1(0.5)], "o", color="black",
         markersize=5, zorder=4)
axB.annotate(r"$w_0$", (-0.5, F1(-0.5)), textcoords="offset points",
             xytext=(-4, 8), fontsize=9)
axB.annotate(r"$w_0$", (0.5, F1(0.5)), textcoords="offset points",
             xytext=(2, 8), fontsize=9)
axB.set_xlabel(r"$w$")
axB.set_ylabel(r"$F(w)$")
axB.set_title("(b) Gradient descent trajectories")
axB.legend(loc="upper center", fontsize=8, framealpha=0.9)

fig.tight_layout()
save(fig, "local_minima")
plt.close(fig)


# =========================================================================
# 2. Saddle point
# =========================================================================
def F2(w1, w2):
    return w1**2 - w2**2


def grad2(w):
    return np.array([2 * w[0], -2 * w[1]])


H2 = 0.05

traj_stall = gd([1.5, 0.0], grad2, H2, 60)    # on the line w2=0 -> stalls
traj_escape = gd([1.5, 0.01], grad2, H2, 55)  # tiny perturbation -> escapes

fig = plt.figure(figsize=(10, 4.2))

g = np.linspace(-2, 2, 220)
G1, G2 = np.meshgrid(g, g)

# (a) 3D saddle surface
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(G1, G2, F2(G1, G2), cmap="viridis",
                 linewidth=0, antialiased=True, alpha=0.9)
ax1.set_xlabel(r"$w_1$")
ax1.set_ylabel(r"$w_2$")
ax1.set_zlabel(r"$F(\mathbf{w})$")
ax1.view_init(elev=28, azim=-58)
ax1.set_title("(a) Saddle surface")

# (b) contour + trajectories
ax2 = fig.add_subplot(1, 2, 2)
levels = np.linspace(-4, 4, 17)
ax2.contourf(G1, G2, F2(G1, G2), levels=levels, cmap="viridis", alpha=0.30)
ax2.contour(G1, G2, F2(G1, G2), levels=levels, colors=GREY, linewidths=0.5)
ax2.plot(traj_stall[:, 0], traj_stall[:, 1], "-o", color=CRIMSON,
         markersize=3, linewidth=1.0, label=r"$w_0=(1.5,0)$")
ax2.plot(traj_escape[:, 0], traj_escape[:, 1], "-o", color=ACCENT,
         markersize=3, linewidth=1.0, label=r"$w_0=(1.5,0.01)$")
ax2.plot(0, 0, "*", color="black", markersize=12, label="saddle")
ax2.set_xlabel(r"$w_1$")
ax2.set_ylabel(r"$w_2$")
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_aspect("equal")
ax2.set_title("(b) Gradient descent trajectories")
ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

fig.tight_layout()
save(fig, "saddle_point")
plt.close(fig)


# =========================================================================
# 3. Non-smoothness: Heaviside (discontinuous) vs ReLU (kink)
# =========================================================================
Y = 0.3  # target label

fig, (axH, axR) = plt.subplots(1, 2, figsize=(10, 4.0))

# (a) Heaviside
wn = np.linspace(-1.0, 0.0, 100, endpoint=False)
wp = np.linspace(0.0, 1.5, 100)
# activation
axH.plot(wn, np.zeros_like(wn), "--", color=ACCENT, linewidth=1.6)
axH.plot(wp, np.ones_like(wp), "--", color=ACCENT, linewidth=1.6,
         label=r"activation $\sigma(wx)$")
# loss
axH.plot(wn, (0.0 - Y) ** 2 * np.ones_like(wn), color=CURVE, linewidth=2)
axH.plot(wp, (1.0 - Y) ** 2 * np.ones_like(wp), color=CURVE, linewidth=2,
         label=r"loss $F(w)$")
# jump markers at w=0 (filled = value attained, open = limit)
axH.plot(0, (1.0 - Y) ** 2, "o", color=CURVE, markersize=6)              # F(0)=0.49
axH.plot(0, (0.0 - Y) ** 2, "o", color="white", markeredgecolor=CURVE,
         markersize=6)                                                    # left limit 0.09
axH.plot(0, 1.0, "o", color=ACCENT, markersize=6)
axH.plot(0, 0.0, "o", color="white", markeredgecolor=ACCENT, markersize=6)
axH.axhline(Y, color=GREY, linestyle=":", linewidth=1)
axH.text(-0.95, Y + 0.03, r"$y=0.3$", color=GREY, fontsize=8)
axH.set_xlabel(r"$w$")
axH.set_title("(a) Heaviside: jump discontinuity")
axH.legend(loc="center right", fontsize=8, framealpha=0.9)

# (b) ReLU
w = np.linspace(-1.0, 1.5, 400)
relu = np.maximum(0.0, w)
axR.plot(w, relu, "--", color=ACCENT, linewidth=1.6,
         label=r"activation $\mathrm{ReLU}(wx)$")
axR.plot(w, (relu - Y) ** 2, color=CURVE, linewidth=2, label=r"loss $F(w)$")
axR.plot(0, (0.0 - Y) ** 2, "o", color=CRIMSON, markersize=6,
         label="kink at $w=0$")
axR.plot(Y, 0.0, "*", color="black", markersize=12, label="minimizer")
axR.axhline(Y, color=GREY, linestyle=":", linewidth=1)
axR.text(-0.95, Y + 0.03, r"$y=0.3$", color=GREY, fontsize=8)
axR.set_xlabel(r"$w$")
axR.set_title("(b) ReLU: non-differentiable kink")
axR.legend(loc="upper center", fontsize=8, framealpha=0.9)

fig.tight_layout()
save(fig, "non_smoothness")
plt.close(fig)


# =========================================================================
# 4. Oscillation in a narrow (ill-conditioned) valley
# =========================================================================
KAPPA = 20.0          # curvature ratio: level sets elongated by sqrt(KAPPA)
H4 = 0.09             # 1/KAPPA = 0.05 < H4 < 2/KAPPA = 0.10  -> oscillation
N4 = 40


def F4(w1, w2):
    return 0.5 * (w1**2 + KAPPA * w2**2)


def grad4(w):
    return np.array([w[0], KAPPA * w[1]])


traj_valley = gd([10.0, 4.0], grad4, H4, N4)

fig = plt.figure(figsize=(10, 4.2))

gx = np.linspace(-11, 11, 220)
gy = np.linspace(-6, 6, 220)
GX, GY = np.meshgrid(gx, gy)

# (a) 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(GX, GY, F4(GX, GY), cmap="viridis",
                 linewidth=0, antialiased=True, alpha=0.9)
ax1.set_xlabel(r"$w_1$")
ax1.set_ylabel(r"$w_2$")
ax1.set_zlabel(r"$F(\mathbf{w})$")
ax1.view_init(elev=38, azim=-58)
ax1.set_title("(a) Objective surface")

# (b) contour + trajectory
ax2 = fig.add_subplot(1, 2, 2)
levels = np.array([2, 8, 20, 50, 100, 180, 260])
ax2.contourf(GX, GY, F4(GX, GY), levels=levels, cmap="viridis", alpha=0.30)
ax2.contour(GX, GY, F4(GX, GY), levels=levels, colors=GREY, linewidths=0.6)
ax2.plot(traj_valley[:, 0], traj_valley[:, 1], "-o", color=CRIMSON,
         markersize=3.0, linewidth=1.0, label="GD iterates")
ax2.plot(traj_valley[0, 0], traj_valley[0, 1], "o", color=CRIMSON,
         markersize=6, label=r"$\mathbf{w}_0$")
ax2.plot(0, 0, "*", color="black", markersize=11,
         label=r"minimizer $\mathbf{w}^\ast$")
ax2.set_xlabel(r"$w_1$")
ax2.set_ylabel(r"$w_2$")
ax2.set_xlim(-11, 11)
ax2.set_ylim(-6, 6)
ax2.set_aspect("equal")
ax2.set_title("(b) Gradient descent trajectory")
ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.tight_layout()
save(fig, "narrow_valley")
plt.close(fig)

print("all chapter-4 figures written to", OUT)
