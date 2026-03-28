import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit


Lx = 2.2
Ly = 0.41
t_max = 2
v_speed_test = 0.12


@njit(cache=True)
def navier_stokes_update_u_kernel(u, un, v, p, force_x, obstacle_mask, nx, ny, dt, dx, dy, rho, nu):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if obstacle_mask[i, j]:
                u[i, j] = 0.0
            else:
                u[i, j] = (
                    un[i, j]
                    - un[i, j] * dt / dx * (un[i, j] - un[i - 1, j])
                    - v[i, j] * dt / dy * (un[i, j] - un[i, j - 1])
                    - dt / (2.0 * dx * rho) * (p[i + 1, j] - p[i - 1, j])
                    + nu * dt / (dx * dx) * (un[i + 1, j] - 2.0 * un[i, j] + un[i - 1, j])
                    + nu * dt / (dy * dy) * (un[i, j + 1] - 2.0 * un[i, j] + un[i, j - 1])
                    + dt * force_x[i, j]
                )


@njit(cache=True)
def navier_stokes_update_v_kernel(u, v, vn, p, force_y, obstacle_mask, nx, ny, dt, dx, dy, rho, nu):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if obstacle_mask[i, j]:
                v[i, j] = 0.0
            else:
                v[i, j] = (
                    vn[i, j]
                    - u[i, j] * dt / dx * (vn[i, j] - vn[i - 1, j])
                    - v[i, j] * dt / dy * (vn[i, j] - vn[i, j - 1])
                    - dt / (2.0 * dy * rho) * (p[i, j + 1] - p[i, j - 1])
                    + nu * dt / (dx * dx) * (vn[i + 1, j] - 2.0 * vn[i, j] + vn[i - 1, j])
                    + nu * dt / (dy * dy) * (vn[i, j + 1] - 2.0 * vn[i, j] + vn[i, j - 1])
                    + dt * force_y[i, j]
                )


@njit(cache=True)
def navier_stokes_update_p_kernel(u, v, p, pn, obstacle_mask, nx, ny, dt, dx, dy, rho):
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if obstacle_mask[i, j]:
                p[i, j] = 0.0
            else:
                du_dx = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dx)
                dv_dy = (v[i, j + 1] - v[i, j - 1]) / (2.0 * dy)
                du_dy = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dy)
                dv_dx = (v[i + 1, j] - v[i - 1, j]) / (2.0 * dx)

                source = (1.0 / dt) * (du_dx + dv_dy) - du_dx * du_dx - 2.0 * du_dy * dv_dx - dv_dy * dv_dy

                p[i, j] = (
                    ((pn[i + 1, j] + pn[i - 1, j]) * dy2 + (pn[i, j + 1] + pn[i, j - 1]) * dx2) / denom
                    - source * rho * dx2 * dy2 / denom
                )


class Navier_stokes_Simulations_finite_difference_numba:
    def __init__(self, Lx, Ly, nx, ny, dt, nu, F, BC_speed=False):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.nu = nu
        self.F = F
        self.Force_x = np.ones((nx, ny), dtype=np.float64) * self.F
        self.Force_y = np.zeros((nx, ny), dtype=np.float64)
        self.rho = 1.0
        self.num_steps = int(t_max / dt)
        self.BC_speed = BC_speed

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        self.u = np.zeros((nx, ny), dtype=np.float64)
        self.v = np.zeros((nx, ny), dtype=np.float64)
        self.p = np.zeros((nx, ny), dtype=np.float64)

        self.obstacle_mask = self.obstacle_grid()

    def obstacle_grid(self):
        x = np.linspace(0.0, self.Lx, self.nx)
        y = np.linspace(0.0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        cx, cy = 0.2, 0.2
        r = 0.05

        obstacle = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
        return obstacle.astype(np.bool_)

    def boundary_conditions(self):
        if self.BC_speed:
            self.u[-1, :] = v_speed_test#self.u[-2, :]# * v_speed_test
            self.u[0, :] = v_speed_test

        else:
            self.u[-1, :] = self.u[-2, :]
            self.u[0, :] = self.u[-1, :]

        self.v[-1, :] = self.v[-2, :]
        self.p[-1, :] = self.p[-2, :]
        self.v[0, :] = self.v[-1, :]
        self.p[0, :] = self.p[-1, :]

        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]

    def run_simulation(self):
        # Warm-up compile happens on first call to each kernel.
        for _ in tqdm(range(self.num_steps)):
            un = self.u.copy()
            vn = self.v.copy()
            pn = self.p.copy()

            navier_stokes_update_u_kernel(
                self.u,
                un,
                self.v,
                self.p,
                self.Force_x,
                self.obstacle_mask,
                self.nx,
                self.ny,
                self.dt,
                self.dx,
                self.dy,
                self.rho,
                self.nu,
            )
            navier_stokes_update_v_kernel(
                self.u,
                self.v,
                vn,
                self.p,
                self.Force_y,
                self.obstacle_mask,
                self.nx,
                self.ny,
                self.dt,
                self.dx,
                self.dy,
                self.rho,
                self.nu,
            )
            navier_stokes_update_p_kernel(
                self.u,
                self.v,
                self.p,
                pn,
                self.obstacle_mask,
                self.nx,
                self.ny,
                self.dt,
                self.dx,
                self.dy,
                self.rho,
            )
            self.boundary_conditions()

    def plot_velocity_field(self):
        speed = np.sqrt(self.u * self.u + self.v * self.v)
        plt.figure(figsize=(10, 5))
        plt.imshow(speed.T, origin="lower", extent=[0, self.Lx, 0, self.Ly], cmap="viridis")
        plt.colorbar(label="Speed")
        plt.title("Velocity Field Speed (Numba JIT)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    sim = Navier_stokes_Simulations_finite_difference_numba(
        Lx,
        Ly,
        nx=int(Lx * 100),
        ny=int(Ly * 100),
        dt=0.001,
        nu=0.005,
        F=1,
        BC_speed = False
    )
    sim.run_simulation()
    sim.plot_velocity_field()
