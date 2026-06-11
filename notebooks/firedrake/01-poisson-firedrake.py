# ---
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Problema de Poisson
#
# Consideremos
#
# $$
# \begin{cases}
# -\Delta u = f & \text{em } \Omega = (0, 1)^2, \\
# u = 0 & \text{em } \partial \Omega.
# \end{cases}
# $$
#
# Esse problema é a porta de entrada clássica para elementos finitos, pois
# expõe de forma limpa a passagem da forma forte para a forma fraca.

# %% [markdown]
# ## Solução manufaturada
#
# Escolhemos
#
# $$
# u_{\mathrm{ex}}(x, y) = \sin(\pi x)\sin(\pi y),
# $$
#
# de modo que
#
# $$
# f(x, y) = 2\pi^2 \sin(\pi x)\sin(\pi y).
# $$
#
# Assim, podemos testar a implementação comparando diretamente a solução
# numérica com um campo conhecido.

# %% [markdown]
# ## Forma fraca
#
# Para $v \in H_0^1(\Omega)$,
#
# $$
# \int_\Omega (-\Delta u) v \, dx = \int_\Omega f v \, dx.
# $$
#
# Integração por partes fornece
#
# $$
# \int_\Omega \nabla u \cdot \nabla v \, dx
# - \int_{\partial \Omega} (\nabla u \cdot n) v \, ds
# =
# \int_\Omega f v \, dx.
# $$
#
# Como $v=0$ na fronteira, obtemos:
#
# $$
# \text{encontrar } u \in H_0^1(\Omega) \text{ tal que }
# \int_\Omega \nabla u \cdot \nabla v \, dx
# =
# \int_\Omega f v \, dx
# \quad \forall v \in H_0^1(\Omega).
# $$

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor, triplot

# %% [markdown]
# ## Malha, espaço e formas variacionais

# %%
domain = UnitSquareMesh(24, 24)
V = FunctionSpace(domain, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)
x, y = SpatialCoordinate(domain)

u_exact = sin(pi * x) * sin(pi * y)
f = 2 * pi**2 * u_exact

a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# %% [markdown]
# ## Condição de contorno e solução

# %%
bc = DirichletBC(V, 0.0, "on_boundary")

u_h = Function(V, name="u_h")
solve(
    a == L,
    u_h,
    bcs=bc,
    solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
)

# %% [markdown]
# ## Diagnósticos

# %%
u_exact_h = Function(V, name="u_ex").interpolate(u_exact)
error_h = Function(V, name="erro").interpolate(u_h - u_exact_h)

l2_error = sqrt(assemble((u_h - u_exact) ** 2 * dx))
h1_semi_error = sqrt(assemble(inner(grad(u_h - u_exact), grad(u_h - u_exact)) * dx))

print(f"Erro L2:        {float(l2_error):.6e}")
print(f"Erro H1-semi:   {float(h1_semi_error):.6e}")
print(f"Graus de liberdade: {V.dim()}")

# %% [markdown]
# ## Visualização
#
# Mantemos a escala de cor compartilhada entre solução exata e solução numérica,
# porque ambas representam a mesma grandeza.

# %%
u_vmin = min(u_exact_h.dat.data_ro.min(), u_h.dat.data_ro.min())
u_vmax = max(u_exact_h.dat.data_ro.max(), u_h.dat.data_ro.max())
e_vmin = error_h.dat.data_ro.min()
e_vmax = error_h.dat.data_ro.max()

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)

triplot(domain, axes=axes[0])
axes[0].set_aspect("equal")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("malha")

for ax, field, title, cmap, vmin, vmax in [
    (axes[1], u_exact_h, "solução exata", "viridis", u_vmin, u_vmax),
    (axes[2], u_h, "solução FEM", "viridis", u_vmin, u_vmax),
    (axes[3], error_h, "erro nodal", "coolwarm", e_vmin, e_vmax),
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
# Como a solução exata é suave, este é um caso bastante favorável para
# elementos lineares. Em problemas com menor regularidade, singularidades ou
# coeficientes descontínuos, o comportamento do erro pode ser bem menos
# benigno.
