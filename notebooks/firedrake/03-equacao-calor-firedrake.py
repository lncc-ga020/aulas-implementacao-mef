# ---
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Equação do calor
#
# Consideremos
#
# $$
# \begin{cases}
# \dfrac{\partial u}{\partial t} - \alpha \Delta u = 0 & \text{em } \Omega, \\
# u = 0 & \text{em } \partial \Omega, \\
# u(\cdot, 0) = u_0 & \text{em } \Omega.
# \end{cases}
# $$
#
# O objetivo aqui é combinar elementos finitos no espaço com Euler implícito no
# tempo.

# %% [markdown]
# ## Forma fraca e discretização temporal
#
# Para $v \in H_0^1(\Omega)$,
#
# $$
# \int_\Omega \frac{\partial u}{\partial t} v \, dx
# +
# \alpha \int_\Omega \nabla u \cdot \nabla v \, dx
# = 0.
# $$
#
# Com Euler implícito,
#
# $$
# \int_\Omega u^{n+1} v \, dx
# +
# \Delta t \, \alpha \int_\Omega \nabla u^{n+1} \cdot \nabla v \, dx
# =
# \int_\Omega u^n v \, dx.
# $$

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from firedrake.pyplot import tripcolor

# %% [markdown]
# ## Solução exata de referência
#
# Com a escolha
#
# $$
# u_0(x, y) = \sin(\pi x)\sin(\pi y),
# $$
#
# temos a solução analítica
#
# $$
# u(x, y, t) = e^{-2\pi^2 \alpha t}\sin(\pi x)\sin(\pi y).
# $$

# %%
domain = UnitSquareMesh(24, 24)
V = FunctionSpace(domain, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)
x, y = SpatialCoordinate(domain)

alpha_value = 1.0
dt_value = 0.005
num_steps = 40
final_time = dt_value * num_steps

alpha = Constant(alpha_value)
dt = Constant(dt_value)

u_n = Function(V, name="u_n").interpolate(sin(pi * x) * sin(pi * y))
u_initial = Function(V, name="u_inicial").assign(u_n)
bc = DirichletBC(V, 0.0, "on_boundary")

a = (u * v + dt * alpha * inner(grad(u), grad(v))) * dx
L = u_n * v * dx

# %% [markdown]
# ## Evolução temporal

# %%
u_h = Function(V, name="u_h")
energies = np.empty(num_steps + 1)
energies[0] = float(assemble(u_n**2 * dx))

for step in range(1, num_steps + 1):
    solve(
        a == L,
        u_h,
        bcs=bc,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    u_n.assign(u_h)
    energies[step] = float(assemble(u_h**2 * dx))

u_exact_final = Function(V, name="u_ex_tfinal").interpolate(
    exp(-2.0 * pi**2 * alpha_value * final_time) * sin(pi * x) * sin(pi * y)
)
error_h = Function(V, name="erro").interpolate(u_n - u_exact_final)
l2_error = sqrt(assemble((u_n - u_exact_final) ** 2 * dx))

print(f"tempo final: {final_time:.3f}")
print(f"energia inicial: {energies[0]:.6e}")
print(f"energia final:   {energies[-1]:.6e}")
print(
    "energia decrescente:",
    all(b <= a + 1e-12 for a, b in zip(energies, energies[1:])),
)
print(f"Erro L2 no tempo final: {float(l2_error):.6e}")

# %% [markdown]
# ## Curva temporal da energia

# %%
times = np.linspace(0.0, final_time, num_steps + 1)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(times, energies, "o-", ms=4)
ax.set_xlabel("tempo")
ax.set_ylabel(r"$\int_\Omega u_h^2\,dx$")
ax.set_title("decaimento da energia discreta")
ax.grid(True)
plt.show()

# %% [markdown]
# ## Campos inicial, final numérico, final exato e erro
#
# Para a variável $u$, mantemos a mesma escala de cores entre tempos diferentes.

# %%
u_vmin = min(
    u_initial.dat.data_ro.min(),
    u_n.dat.data_ro.min(),
    u_exact_final.dat.data_ro.min(),
)
u_vmax = max(
    u_initial.dat.data_ro.max(),
    u_n.dat.data_ro.max(),
    u_exact_final.dat.data_ro.max(),
)
e_vmin = error_h.dat.data_ro.min()
e_vmax = error_h.dat.data_ro.max()

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
for ax, field, title, cmap, vmin, vmax in [
    (axes[0, 0], u_initial, "campo inicial", "inferno", u_vmin, u_vmax),
    (axes[0, 1], u_n, "campo final numérico", "inferno", u_vmin, u_vmax),
    (axes[1, 0], u_exact_final, "campo final exato", "inferno", u_vmin, u_vmax),
    (axes[1, 1], error_h, "erro no tempo final", "coolwarm", e_vmin, e_vmax),
]:
    collection = tripcolor(field, axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(collection, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
plt.show()

# %% [markdown]
# ## Comentário final
#
# A solução final fica visualmente escura quando usamos a mesma escala do campo
# inicial, e isso é esperado: a difusão reduziu bastante a amplitude. O ponto
# importante é que agora a comparação entre tempos diferentes é honesta na escala
# de cor.
