import requests
import numpy as np
from sgp4.api import Satrec, WGS72
import kepOrbits as ko
import matplotlib.pyplot as plt

# AI testing code


n_sats = 1000
url = "https://celestrak.org/NORAD/elements/gp.php?NAME=STARLINK&FORMAT=TLE"

response = requests.get(url, timeout=60)
response.raise_for_status()
tle_text = response.text.strip()

print("HTTP status:", response.status_code)
print("First lines received:")
print("\n".join(tle_text.splitlines()[:6]))

lines = tle_text.splitlines()
satellites = []

# Parse three-line TLE records: name, line 1, line 2
i = 0
while i + 2 < len(lines):
    name = lines[i].strip()
    line1 = lines[i + 1].strip()
    line2 = lines[i + 2].strip()

    if line1.startswith("1 ") and line2.startswith("2 "):
        try:
            sat = Satrec.twoline2rv(line1, line2, WGS72)
            eccentricity = sat.ecco
            inclination = np.degrees(sat.inclo)
            period_min = 2 * np.pi / sat.no_kozai

            print(
                f"{name:25s} "
                f"e={eccentricity:.3f}, "
                f"i={inclination:.2f} deg, "
                f"T={period_min:.1f} min"
            )


            satellites.append((name, sat))

        except Exception as exc:
            print(f"Skipping {name}: {exc}")

        i += 3
    else:
        i += 1

print(f"\nFound {len(satellites)} suitable satellites.")

if len(satellites) < n_sats:
    raise RuntimeError(
        "Fewer than needed satellites passed the filters. "
        "Check the printed response and orbital elements."
    )



# Parse the TLE records
for i in range(0, len(lines) - 2, 3):
    name = lines[i].strip()
    line1 = lines[i + 1]
    line2 = lines[i + 2]

    sat = Satrec.twoline2rv(line1, line2)

    eccentricity = sat.ecco
    inclination = np.degrees(sat.inclo)
    period_min = 2 * np.pi / sat.no_kozai

    # Select Molniya-like orbits
    if (
        0.5 < eccentricity < 0.8
        and 60 < inclination < 66
        and 650 < period_min < 800
    ):
        satellites.append((name, sat))

    if len(satellites) > n_sats :
        break

if len(satellites) < n_sats:
    raise RuntimeError("Could not find enough suitable Molniya satellites.")

satellites = satellites[:n_sats]

# Create your Orbit objects
orbits = []

for name, sat in satellites:
    error, r, v = sat.sgp4(
        sat.jdsatepoch,
        sat.jdsatepochF
    )

    if error != 0:
        print(f"Skipping {name}: SGP4 error {error}")
        continue

    # Convert km and km/s to m and m/s
    r0 = np.array(r) * 1000
    v0 = np.array(v) * 1000

    # Use your own Orbit class and Earth object
    orbit = ko.Orbit(r0, v0, 0, ko.Earth)

    orbits.append((name, orbit))

    print(
        f"{name}: e={orbit.e_mag:.3f}, "
        f"i={inclination:.2f} deg, "
        f"T={period_min / 60:.2f} hours"
    )

# Plot all satellites on the same axes
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Draw Earth once and plot the first orbit
for i, (name, orbit) in enumerate(orbits):
    orbit.name = name

    orbit.plot_3d(
        show=False,
        ax=ax,
        plot_body=False, # plot_body=(i == 0) for body shown
        time=0,
        DrawOrbit=False,
        color=colors[i % len(colors)]
    )

# Use a common scale for all orbits
plot_limit = max(
    orbit.ra_mag for _, orbit in orbits
) * 1.1

ax.set_xlim(-plot_limit, plot_limit)
ax.set_ylim(-plot_limit, plot_limit)
ax.set_zlim(-plot_limit, plot_limit)
ax.set_box_aspect((1, 1, 1))

ax.set_title("Starlink Satellite Orbits")
# ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

