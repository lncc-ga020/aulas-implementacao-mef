# ---
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Elasticidade linear
#
# Procuramos o deslocamento vetorial $u : \Omega \to \mathbb{R}^2$ em pequenas
# deformações, governado por
#
# $$
# \begin{cases}
# -\nabla \cdot \sigma(u) = f & \text{em } \Omega, \\
# u = \bar{u} & \text{em } \Gamma_D, \\
# \sigma(u)n = \bar{t} & \text{em } \Gamma_N.
# \end{cases}
# $$
#
# Para elasticidade linear isotrópica,
#
# $$
# \varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T),
# \qquad
# \sigma(u) = \lambda \, \mathrm{tr}(\varepsilon(u)) I + 2 \mu \, \varepsilon(u).
# $$

# %% [markdown]
# ## Forma fraca
#
# Tomando $v \in [H_0^1(\Omega)]^2$,
#
# $$
# \int_\Omega \sigma(u) : \varepsilon(v) \, dx
# =
# \int_\Omega f \cdot v \, dx + \int_{\Gamma_N} \bar{t} \cdot v \, ds.
# $$
#
# Neste notebook, vamos considerar apenas força de corpo e uma face engastada.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import quiver, tripcolor

# %% [markdown]
# ## Geometria, espaço e parâmetros

# %%
Lx, Ly = 1.0, 0.2
domain = RectangleMesh(36, 8, Lx, Ly)
V = VectorFunctionSpace(domain, "CG", 1)
Q = FunctionSpace(domain, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

E = 12.0
nu = 0.30
mu_value = E / (2.0 * (1.0 + nu))
lambda_value = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

mu = Constant(mu_value)
lambda_ = Constant(lambda_value)
rho = Constant(0.08)
g = Constant(9.81)
f = as_vector((0.0, -rho * g))
I = Identity(domain.geometric_dimension())


def epsilon(w):
    return sym(grad(w))


def sigma(w):
    return lambda_ * tr(epsilon(w)) * I + 2.0 * mu * epsilon(w)


a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx

# %% [markdown]
# ## Condição de contorno e solução
#
# Engastamos a face esquerda.

# %%
bc = DirichletBC(V, Constant((0.0, 0.0)), 1)

u_h = Function(V, name="deslocamento")
solve(
    a == L,
    u_h,
    bcs=bc,
    solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
)

# %% [markdown]
# ## Quantidades derivadas

# %%
u_x = Function(Q, name="u_x").interpolate(u_h[0])
u_y = Function(Q, name="u_y").interpolate(u_h[1])
u_mag = Function(Q, name="u_mag").interpolate(sqrt(dot(u_h, u_h)))

strain_energy = 0.5 * assemble(inner(sigma(u_h), epsilon(u_h)) * dx)

print(f"u_x em [{u_x.dat.data_ro.min():.6e}, {u_x.dat.data_ro.max():.6e}]")
print(f"u_y em [{u_y.dat.data_ro.min():.6e}, {u_y.dat.data_ro.max():.6e}]")
print(f"|u| máximo: {u_mag.dat.data_ro.max():.6e}")
print(f"Energia elástica: {float(strain_energy):.6e}")

# %% [markdown]
# ## Visualizações
#
# Primeiro mostramos o campo vetorial com `quiver`. Em seguida, exibimos a
# magnitude e as duas componentes do deslocamento.

# %%
fig, ax = plt.subplots(figsize=(8, 3.0))
collection = quiver(u_h, axes=ax)
fig.colorbar(collection, ax=ax, label=r"$|u_h|$")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("campo de deslocamentos")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), constrained_layout=True)
for ax, field, title, cmap in [
    (axes[0], u_mag, r"$|u_h|$", "viridis"),
    (axes[1], u_x, r"componente $u_x$", "coolwarm"),
    (axes[2], u_y, r"componente $u_y$", "coolwarm"),
]:
    collection = tripcolor(field, axes=ax, cmap=cmap)
    fig.colorbar(collection, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
plt.show()

# %% [markdown]
# ## Comentário final
#
# Este é um modelo linear e bastante simplificado. Ele é excelente para ensinar
# a estrutura variacional do problema, mas não deve ser confundido com um modelo
# constitutivo completo para cenários com plasticidade, dano ou grandes
# deformações.
