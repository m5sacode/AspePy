import numpy as np
import consts as csts
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import brentq

mpl.use('TkAgg')

class Body():
    def __init__(self, mass, radius,  rot_velocity=0, orbit=None):
        self.mass = mass
        self.radius = radius
        if orbit is not None:
            self.refbody = orbit.refbody
        self.orbit = orbit
        self.mu = csts.G * mass
        self.omega = rot_velocity



class Orbit():
    def __init__(self, refbody, name="Unamed Orbit"):
        self.refbody = refbody
        self.name = name
        self.mu = refbody.mu
    def from_r0_v0_t0(self, r0:np.ndarray, v0:np.ndarray, t0:float):

        self.H_vec = np.cross(r0, v0)
        self.H_mag=np.linalg.norm(self.H_vec)
        v0_mag = np.linalg.norm(v0)
        r0_mag = np.linalg.norm(r0)
        self.e_vec =  (1/self.mu) * ((v0_mag**2 - self.mu/r0_mag)*r0 - np.dot(r0, v0)*v0)
        self.e_mag = np.linalg.norm(self.e_vec)

        self.rp_unit = (1/self.e_mag) * self.e_vec

        self.get_orientation_from_H()

        self.p = (self.H_mag**2)/self.mu

        self.rp_mag = self.get_r_mag_from_theta(0)
        self.ra_mag = self.get_r_mag_from_theta(np.pi)

        self.a = (self.rp_mag + self.ra_mag) /2

        self.rp = self.rp_mag * self.rp_unit
        self.ra = self.ra_mag * self.rp_unit * -1

        if self.e_mag < 1 and self.e_mag > 0:  # Ellipse
            self.T = 2*np.pi*np.sqrt((self.a**3)/self.mu)
            self.n = 2*np.pi/self.T
        elif self.e_mag > 1:
            self.n_bar = np.sqrt(self.mu/(-self.a**3))
            self.T = np.inf


        self.r0 = r0
        self.v0 = v0
        self.t0 = t0

        self.theta_0 = self.get_theta_from_r_v(r0,v0)

        self.tp_first = self.get_tp_from_r_v(r0,v0)

    def from_orbital_elements(self, a, e_mag, theta_0, i, raan, argp, t0):
        self.a = a
        self.e_mag = e_mag
        self.i = i
        self.raan = raan
        self.argp = argp
        self.theta_0 = theta_0

        self.p = a* (1- e_mag**2)

        self.e_vec = self.e_mag * np.array([
            np.cos(raan) * np.cos(argp) - np.sin(raan) * np.sin(argp) * np.cos(i),

            np.sin(raan) * np.cos(argp) + np.cos(raan) * np.sin(argp) * np.cos(i),

            np.sin(argp) * np.sin(i)
        ])
        self.rp_unit = (1 / self.e_mag) * self.e_vec
        self.get_H_from_elements()

        self.r0, self.v0 = self.get_r_v_from_theta(theta_0)

        self.p = (self.H_mag**2)/self.mu

        self.rp_mag = self.get_r_mag_from_theta(0)
        self.ra_mag = self.get_r_mag_from_theta(np.pi)

        self.a = (self.rp_mag + self.ra_mag) /2

        self.rp = self.rp_mag * self.rp_unit
        self.ra = self.ra_mag * self.rp_unit * -1

        if self.e_mag < 1 and self.e_mag > 0:  # Ellipse
            self.T = 2*np.pi*np.sqrt((self.a**3)/self.mu)
            self.n = 2*np.pi/self.T
        elif self.e_mag > 1:
            self.n_bar = np.sqrt(self.mu/(-self.a**3))
            self.T = np.inf

        self.t0 = t0

        self.tp_first = self.get_tp_from_r_v(self.r0, self.v0)





    def get_orientation_from_H(self):
        H = np.asarray(self.H_vec, dtype=float)
        rp = np.asarray(self.rp_unit, dtype=float)

        H_norm = np.linalg.norm(H)
        H_hat = H / H_norm

        # Inclination
        self.i = np.arccos(np.clip(H_hat[2], -1.0, 1.0))

        # Ascending node vector
        N = np.cross([0.0, 0.0, 1.0], H)
        N_norm = np.linalg.norm(N)

        if N_norm > 1e-12:
            N_hat = N / N_norm

            # Right ascension of the ascending node
            self.raan = np.arctan2(N[1], N[0]) % (2 * np.pi)

            # Argument of periapsis
            self.argp = np.arctan2(
                np.dot(H_hat, np.cross(N_hat, rp)),
                np.dot(N_hat, rp)
            ) % (2 * np.pi)

        else:
            # Equatorial orbit: RAAN is undefined.
            self.raan = np.nan
            self.argp = np.nan


    def get_H_from_elements(self):
        a = self.a
        e = self.e_mag
        i = self.i
        mu = self.mu
        raan = self.raan

        # Angular momentum magnitude
        self.H_mag = np.sqrt(mu * a * (1.0 - e ** 2))

        # Angular momentum vector in the inertial frame
        self.H_vec = self.H_mag * np.array([
            np.sin(raan) * np.sin(i),
            -np.cos(raan) * np.sin(i),
            np.cos(i)
        ])

        return self.H_vec



    def get_r_mag_from_theta(self, theta):
        r_mag = self.p/(1+self.e_mag*np.cos(theta))
        return r_mag

    def get_theta_from_r_v(self, r, v):
        r_mag = np.linalg.norm(r)
        v_radial = np.dot(r, v)/r_mag
        theta = self.get_theta_from_r_mag(r_mag)
        if v_radial < 0:
            theta *= -1
        return theta

    def get_r_v_from_theta(self, theta):
        r_mag = self.get_r_mag_from_theta(theta)

        u = self.argp + theta

        r_vec = r_mag * np.array([
            np.cos(self.raan) * np.cos(u)
            - np.sin(self.raan) * np.sin(u) * np.cos(self.i),

            np.sin(self.raan) * np.cos(u)
            + np.cos(self.raan) * np.sin(u) * np.cos(self.i),

            np.sin(u) * np.sin(self.i)
        ])

        H_hat = self.H_vec / np.linalg.norm(self.H_vec)

        r_hat = r_vec / r_mag

        t_hat = np.cross(H_hat, r_hat)

        e = self.e_mag
        p = self.a * (1.0 - e ** 2)

        v_r = np.sqrt(self.mu / p) * e * np.sin(theta)
        v_t = np.linalg.norm(self.H_vec) / r_mag

        v_vec = v_r * r_hat + v_t * t_hat

        return r_vec, v_vec

    def get_theta_from_r_mag(self, r_mag):
        theta = np.arccos(((self.p/r_mag)-1)/self.e_mag)
        return theta

    def get_E_anom_from_theta(self, theta):
        if self.e_mag < 1 and self.e_mag>0: # Ellipse
            E = 2*np.arctan(np.tan(theta/2)/np.sqrt((1+self.e_mag)/(1-self.e_mag)))
            return E
        else:
            raise RuntimeError('E anomaly is not a thing for non elliptical orbits you dumbass... Did you mean F anomaly?')

    def get_F_anom_from_theta(self, theta):
        if self.e_mag > 1: # Hyperbola
            F = 2*np.arctanh(np.tan(theta/2)/np.sqrt((1+self.e_mag)/(1-self.e_mag)))
            return F
        else:
            raise RuntimeError('F anomaly is not a thing for elliptical orbits you dumbass... Did you mean E anomaly?')

    def get_M_anom_from_theta(self, theta):
        if self.e_mag < 1 and self.e_mag>0: # Ellipse
            E = self.get_E_anom_from_theta(theta)
            M = E - self.e_mag*np.sin(E)
            return M
        else:
            raise RuntimeError(
                'M anomaly is not a thing for non elliptical orbits you dumbass... Did you mean M_bar anomaly?')

    def get_M_bar_anom_from_theta(self, theta):
        if self.e_mag > 1: # Hyperbola
            F = self.get_F_anom_from_theta(theta)
            M_bar = self.e_mag*np.sinh(F) - F
            return M_bar
        else:
            raise RuntimeError(
                'M_bar anomaly is not a thing for elliptical orbits you dumbass... Did you mean M anomaly?')

    def get_tp_from_theta(self, theta):
        if self.e_mag < 1 and self.e_mag>0: # Ellipse
            M = self.get_M_anom_from_theta(theta)
            tp = M/self.n
        elif self.e_mag > 1:
            M_bar = self.get_M_bar_anom_from_theta(theta)
            tp = M_bar / self.n_bar
        return tp

    def get_theta_from_tp(self, tp):
        def residual(theta):
            return self.get_tp_from_theta(theta) - tp

        if 0 < self.e_mag < 1:  # Ellipse
            theta_min = -np.pi
            theta_max = np.pi

        elif self.e_mag > 1:  # Hyperbola
            theta_max = np.arccos(-1 / self.e_mag)
            theta_min = -theta_max
            eps = 1e-10
            theta_min += eps
            theta_max -= eps

        else:
            raise ValueError("Only elliptical and hyperbolic orbits are supported.")

        return brentq(residual, theta_min, theta_max)

    def get_tp_from_t(self, t):
        if 0 < self.e_mag < 1:  # Ellipse
            tp = (t - self.tp_first) % self.T

            # Center the result around periapsis: [-T/2, T/2)
            if tp > self.T / 2:
                tp -= self.T

        elif self.e_mag > 1:  # Hyperbola
            tp = t - self.tp_first

        else:
            raise ValueError("Only elliptical and hyperbolic orbits are supported.")

        return tp


    def get_tp_from_r_v(self, r, v):
        theta = self.get_theta_from_r_v(r, v)
        tp = self.get_tp_from_theta(theta)
        return tp



    def plot_3d(self, n_points=500, max_distance=None,
                show=True, ax=None, plot_body=True, color='royalblue', time=None, DrawNodes=False, Drawa=False, DrawOrbit=True, DrawSphereRadius=None):
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
        if DrawOrbit:
            ax.plot(
                r_3d[0], r_3d[1], r_3d[2],
                label=self.name,
                color=color,
                linewidth=1
            )

        if time is not None:
            tp = self.get_tp_from_t(time-self.t0)
            theta_sc = self.get_theta_from_tp(tp)
            r_sc_mag = self.get_r_mag_from_theta(theta_sc)

            r_sc = r_sc_mag * (
                np.cos(theta_sc) * p_hat
                + np.sin(theta_sc) * q_hat
            )

            ax.scatter(*r_sc, color='black', s=2)
            if DrawSphereRadius is not None:
                u = np.linspace(0, 2 * np.pi, 40)
                v = np.linspace(0, np.pi, 20)

                radius = DrawSphereRadius

                x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + r_sc[0]
                y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + r_sc[1]
                z_sphere = radius * np.outer(np.ones_like(u), np.cos(v)) + r_sc[2]

                ax.plot_surface(
                    x_sphere, y_sphere, z_sphere,
                    color='grey',
                    alpha=0.6,
                    linewidth=0
                )

        if plot_body:
            # Plot reference body as a sphere
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 20)

            radius = self.refbody.radius

            x_sphere = radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = radius * np.outer(np.ones_like(u), np.cos(v))

            ax.plot_surface(
                x_sphere, y_sphere, z_sphere,
                color='deepskyblue',
                alpha=0.6,
                linewidth=0
            )
        if DrawNodes:
            # Periapsis
            rp = self.rp

            ax.scatter(
                *rp,
                color='green',
                s=10,
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
            if Drawa:
                ra = self.ra

                ax.plot(
                    [rp[0], ra[0]],
                    [rp[1], ra[1]],
                    [rp[2], ra[2]],
                    linestyle=':',
                    color='black',
                    linewidth=1.5
                )
            if DrawNodes:
                # Apoapsis
                ax.scatter(
                    *ra,
                    color='red',
                    s=10,
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
            if Drawa:
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
        radius = self.refbody.radius

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

Earth = Body(csts.M_Earth, csts.R_Earth, rot_velocity=(2*np.pi/csts.SID_DAY))
Moon_Orbit = Orbit(Earth, name="Moon Orbit")
Moon_Orbit.from_orbital_elements(0.384401e9, 0.055, 0, 0, np.radians(5.1454), np.radians(330.12), np.radians(148.23))
Moon = Body(csts.M_Moon, csts.R_Moon, orbit=Moon_Orbit)

# Moon_Orbit.plot_3d(DrawSphereRadius=Moon.radius, time=0)

