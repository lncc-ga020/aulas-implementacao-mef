# ---
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Modelo de Gray-Scott
#
# Neste notebook, consideramos o sistema
#
# $$
# \begin{aligned}
# \frac{\partial u}{\partial t} &= \Delta u - u v^2 + A(1-u), \\
# \frac{\partial v}{\partial t} &= \delta^2 \Delta v + u v^2 - Bv.
# \end{aligned}
# $$
#
# Este é o exemplo mais completo da sequência: ele é transiente, acoplado e não
# linear.

# %% [markdown]
# ## Forma fraca
#
# Com funções-teste $p$ e $q$, a discretização de Euler implícito leva a
#
# $$
# \begin{aligned}
# \left(\frac{u-u_0}{\Delta t}, p\right)_\Omega
# + (\nabla u, \nabla p)_\Omega
# + (u v^2, p)_\Omega
# - (A(1-u), p)_\Omega
# &= 0, \\
# \left(\frac{v-v_0}{\Delta t}, q\right)_\Omega
# + (\delta^2 \nabla v, \nabla q)_\Omega
# - (u v^2, q)_\Omega
# + (Bv, q)_\Omega
# &= 0.
# \end{aligned}
# $$
#
# O par $(u_0, v_0)$ representa o estado no passo anterior.

# %% [markdown]
# ## Configuração do problema
#
# Mantemos a mesma configuração usada no material original do curso:
#
# - domínio unidimensional $[0, L]$;
# - duas incógnitas acopladas;
# - espaço de aproximação de grau mais alto para o problema não linear;
# - armazenamento da evolução temporal para gráficos finais.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from firedrake.exceptions import ConvergenceError

# %% [markdown]
# ## Geometria e espaços

# %%
num_elements = 800
L_domain = 100.0
degree = 3

domain = IntervalMesh(num_elements, 0.0, L_domain)
V = FunctionSpace(domain, "CG", degree)
W = V * V
Vref = FunctionSpace(domain, "CG", 1)

(x,) = SpatialCoordinate(domain)

# %% [markdown]
# ## Funções do problema e condições iniciais
#
# Usamos
#
# $$
# u(x, 0) = 1 - \frac{1}{2}\sin^{100}\left(\frac{\pi x}{L}\right),
# \qquad
# v(x, 0) = \frac{1}{4}\sin^{100}\left(\frac{\pi x}{L}\right).
# $$

# %%
w = Function(W, name="w")
u, v = split(w)
p, q = TestFunctions(W)

w0 = Function(W, name="w0")
u0_fun, v0_fun = w0.subfunctions
u0_fun.interpolate(1.0 - 0.5 * sin(pi * x / L_domain) ** 100)
v0_fun.interpolate(0.25 * sin(pi * x / L_domain) ** 100)
w.assign(w0)

# %% [markdown]
# ## Condições de contorno essenciais
#
# Impomos
#
# $$
# u(0, t) = u(L, t) = 1,
# \qquad
# v(0, t) = v(L, t) = 0.
# $$

# %%
u_bc = DirichletBC(W.sub(0), 1.0, "on_boundary")
v_bc = DirichletBC(W.sub(1), 0.0, "on_boundary")

# %% [markdown]
# ## Parâmetros do modelo

# %%
delta_squared = Constant(0.01)
A = Constant(0.01)
B = Constant(0.053)

total_time = 4000.0
dt_value = 1.0
Dt = Constant(dt_value)
num_steps = int(total_time / dt_value)

# %% [markdown]
# ## Residual não linear

# %%
F = inner((u - u0_fun) / Dt, p) * dx
F += inner(grad(u), grad(p)) * dx
F += u * v * v * p * dx
F -= A * (1.0 - u) * p * dx

F += inner((v - v0_fun) / Dt, q) * dx
F += inner(delta_squared * grad(v), grad(q)) * dx
F -= u * v * v * q * dx
F += B * v * q * dx

problem = NonlinearVariationalProblem(F, w, bcs=[u_bc, v_bc])
solver = NonlinearVariationalSolver(
    problem,
    solver_parameters={
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)

# %% [markdown]
# ## Estruturas para armazenamento
#
# Projetamos as soluções em um espaço `CG1` apenas para visualização e para
# construir os gráficos finais.

# %%
usol_ref = Function(Vref, name="u_ref")
vsol_ref = Function(Vref, name="v_ref")
x_ref = Function(Vref, name="x_ref").interpolate(x)

x_values = x_ref.dat.data_ro.copy()
order = np.argsort(x_values)
x_values = x_values[order]

times = np.linspace(0.0, total_time, num_steps + 1)
u_history = np.empty((num_steps + 1, x_values.size))
v_history = np.empty((num_steps + 1, x_values.size))

usol_ref.interpolate(u0_fun)
vsol_ref.interpolate(v0_fun)
u_history[0, :] = usol_ref.dat.data_ro[order]
v_history[0, :] = vsol_ref.dat.data_ro[order]

# %% [markdown]
# ## Estado inicial

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_values, u_history[0], "--", lw=2, label="u inicial")
ax.plot(x_values, v_history[0], lw=2, label="v inicial")
ax.set_xlabel("x")
ax.set_ylabel("concentração")
ax.set_title("condições iniciais")
ax.grid(True)
ax.legend()
plt.show()

# %% [markdown]
# ## Evolução temporal
#
# Para manter o passo nominal da aula e, ao mesmo tempo, tornar a execução mais
# robusta nesta versão do Firedrake, permitimos subpassos internos se um passo
# inteiro não convergir no Newton. O instante armazenado continua sendo o mesmo
# da malha temporal principal.

# %%
step_start = Function(W, name="step_start")
subdivision_schedule = [1, 2, 4]

for step in range(1, num_steps + 1):
    step_start.assign(w0)
    converged = False

    for nsub in subdivision_schedule:
        w.assign(step_start)
        w0.assign(step_start)
        Dt.assign(dt_value / nsub)

        try:
            for _ in range(nsub):
                solver.solve()
                w0.assign(w)
            converged = True
            if nsub > 1:
                print(f"passo {step:4d}: usado substepping com {nsub} subpassos")
            break
        except ConvergenceError:
            continue

    if not converged:
        raise RuntimeError(
            f"Falha de convergência no passo {step} mesmo após substepping."
        )

    Dt.assign(dt_value)

    usol, vsol = w.subfunctions
    usol_ref.interpolate(usol)
    vsol_ref.interpolate(vsol)
    u_history[step, :] = usol_ref.dat.data_ro[order]
    v_history[step, :] = vsol_ref.dat.data_ro[order]

    if step % 500 == 0:
        print(f"passo {step:4d} / {num_steps}")

print(f"u em [{u_history[-1].min():.6e}, {u_history[-1].max():.6e}]")
print(f"v em [{v_history[-1].min():.6e}, {v_history[-1].max():.6e}]")

# %% [markdown]
# ## Perfis espaciais no tempo final

# %%
selected_time = 4000.0
selected_index = int(round(selected_time / dt_value))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x_values, u_history[selected_index], "--", lw=2, label="U")
ax.plot(x_values, v_history[selected_index], lw=2, label="V")
ax.set_xlabel("x [L]")
ax.set_ylabel("concentração [adim]")
ax.set_xlim(x_values.min(), x_values.max())
ax.set_title(f"concentrações no tempo t = {times[selected_index]:.0f}")
ax.grid(True)
ax.legend()
plt.show()

# %% [markdown]
# ## Evolução espaço-tempo da componente $v$

# %%
fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(
    v_history,
    origin="lower",
    aspect="auto",
    extent=[x_values.min(), x_values.max(), times[0], times[-1]],
    cmap="jet",
    vmin=float(v_history.min()),
    vmax=float(v_history.max()),
)
fig.colorbar(image, ax=ax, label="concentração da componente v")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("evolução espaço-tempo de v")
plt.show()

# %% [markdown]
# ## Comentário final
#
# Aqui o modelo é simples do ponto de vista geométrico, mas numericamente bem
# mais rico do que os exemplos anteriores. Em problemas maiores, outras escolhas
# de pré-condicionador e de estratégia de solução não linear passam a ser
# fundamentais.
