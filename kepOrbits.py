import numpy as np
import consts as csts
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

class Body():
    def __init__(self, mass, radius, r0:np.ndarray, v0:np.ndarray, t0, rot_velocity=0, ref_body=None):
        self.mass = mass
        self.radius = radius
        self.ref_body = ref_body
        self.r0 = r0
        self.v0 = v0
        self.mu = csts.G * mass
        self.omega = rot_velocity

Earth = Body(csts.M_Earth, csts.R_Earth, np.array([0,0,0]), np.array([0,0,0]), 0, rot_velocity=(2*np.pi/csts.SID_DAY))

class Orbit():
    def __init__(self, r0:np.ndarray, v0:np.ndarray, t0, ref_body, name="Unamed Orbit"):
        self.name = name

        self.H_vec = np.cross(r0, v0)
        self.H_mag=np.linalg.norm(self.H_vec)
        self.mu = ref_body.mu
        self.ref_body = ref_body
        v0_mag = np.linalg.norm(v0)
        r0_mag = np.linalg.norm(r0)
        self.e_vec =  (1/self.mu) * ((v0_mag**2 - self.mu/r0_mag)*r0 - np.dot(r0, v0)*v0)
        self.e_mag = np.linalg.norm(self.e_vec)

        self.rp_unit = (1/self.e_mag) * self.e_vec

        self.p = (self.H_mag**2)/self.mu

        self.rp_mag = self.get_r_mag(0)
        self.ra_mag = self.get_r_mag(np.pi)

        self.rp = self.rp_mag * self.rp_unit
        self.ra = self.ra_mag * self.rp_unit * -1


        self.r0 = r0
        self.v0 = v0
        self.t0 = t0

    def get_r_mag_from_theta(self, theta):
        r_mag = self.p/(1+self.e_mag*np.cos(theta))
        return r_mag

    def get_theta_from_r_mag(self, r_mag):
        theta = np.arccos(((self.p/r_mag)-1)/self.e_mag)
        return theta

    def plot_3d(self, n_points=500, max_distance=None,
                show=True, ax=None, plot_body=True, color='royalblue', time=None):
        # THIS IS ALL AI SHENNANIGANS!!!
        # Determine orbit type
        is_elliptic = self.e_mag < 1

        # Set true anomaly range
        if is_elliptic:
            theta = np.linspace(0, 2 * np.pi, n_points)
        else:
            if max_distance is None:
                max_distance = 10 * self.rp_mag

            # Maximum true anomaly allowed by max_distance
            cos_theta_max = (
                                    self.p / max_distance - 1
                            ) / self.e_mag

            theta_inf = np.arccos(-1 / self.e_mag)
            theta_max = min(
                np.arccos(np.clip(cos_theta_max, -1, 1)),
                theta_inf - 1e-6
            )

            theta = np.linspace(-theta_max, theta_max, n_points)

        # Orbital radius
        r_mag = self.p / (1 + self.e_mag * np.cos(theta))

        # Perifocal coordinates
        r_pf = np.array([
            r_mag * np.cos(theta),
            r_mag * np.sin(theta),
            np.zeros_like(theta)
        ])

        # Build orbital coordinate system
        h_hat = self.H_vec / self.H_mag
        p_hat = self.e_vec / self.e_mag
        q_hat = np.cross(h_hat, p_hat)

        # Transform orbit into 3D coordinates
        R = np.column_stack((p_hat, q_hat, h_hat))
        r_3d = R @ r_pf

        # Create or reuse figure and axes
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # Plot orbit
        ax.plot(
            r_3d[0], r_3d[1], r_3d[2],
            label=self.name,
            color=color,
            linewidth=2
        )
        if plot_body:
            # Plot reference body as a sphere
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 20)

            radius = self.ref_body.radius

            x_sphere = radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = radius * np.outer(np.ones_like(u), np.cos(v))

            ax.plot_surface(
                x_sphere, y_sphere, z_sphere,
                color='deepskyblue',
                alpha=0.6,
                linewidth=0
            )

        # Periapsis
        rp = self.rp

        ax.scatter(
            *rp,
            color='green',
            s=20,
            depthshade=False
        )

        ax.text(
            *rp,
            ' Pe',
            color='green',
            fontsize=10,
            fontweight='bold'
        )

        # Major axis
        if is_elliptic:
            ra = self.ra

            ax.plot(
                [rp[0], ra[0]],
                [rp[1], ra[1]],
                [rp[2], ra[2]],
                linestyle=':',
                color='black',
                linewidth=1.5
            )

            # Apoapsis
            ax.scatter(
                *ra,
                color='red',
                s=20,
                depthshade=False
            )

            ax.text(
                *ra,
                ' Ap',
                color='red',
                fontsize=10,
                fontweight='bold'
            )

        else:
            # Major axis for hyperbolic orbit
            axis_length = max_distance

            ax.plot(
                [-axis_length * p_hat[0], axis_length * p_hat[0]],
                [-axis_length * p_hat[1], axis_length * p_hat[1]],
                [-axis_length * p_hat[2], axis_length * p_hat[2]],
                linestyle=':',
                color='black',
                linewidth=1.5
            )

        # Reference body center
        ax.scatter(0, 0, 0, color='black', s=15)
        radius = self.ref_body.radius

        # Set equal scaling
        if is_elliptic:
            plot_limit = max(
                np.max(np.abs(r_3d)),
                radius
            ) * 1.1
        else:
            plot_limit = max(max_distance, radius) * 1.1

        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_zlim(-plot_limit, plot_limit)
        ax.set_box_aspect((1, 1, 1))

        # Labels and title
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')



        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax