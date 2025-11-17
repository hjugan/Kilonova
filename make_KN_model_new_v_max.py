import numpy as np
import h5py
import matplotlib.pyplot as plt
import os  # <-- Import os for file checking

# --- 1. DEFINE PHYSICAL CONSTANTS (in CGS) ---
M_sol = 1.989e33  # g
c = 2.9979e10     # cm/s
day = 86400.0       # s

# --- 2. DEFINE KILONOVA MODEL PARAMETERS ---
M_ej = 0.03 * M_sol   # Total ejecta mass
v_k = 0.1 * c         # Characteristic "break" velocity (v_t in screenshot)
n_inner = 1.0         # Inner density power-law index (delta in screenshot)
n_outer = 10.0        # Outer density power-law index (n in screenshot)

# --- 3. DEFINE THE COMPUTATIONAL GRID ---
# We define the model at a reference time, t_exp.
t_exp = 1.0 * day
n_zones = 200         # Number of radial zones

# --- Calculate v_max ---
# v_max = v_k * (1000)^(1/n_outer) = v_k * 10^(3/n_outer) , Kasen 2017 paper
v_max = v_k * (10.0)**(3.0 / n_outer)

print("--- Grid Parameters ---")
print(f"Break velocity v_k: {v_k/c:.4f}c")
print(f"Outer slope n_outer: {n_outer}")
print(f"Calculated v_max:   {v_max/c:.4f}c (vs. old 0.4000c)")
print("-----------------------")

r_max = v_max * t_exp # Maximum radius of the grid at t_exp

# Create the radial grid cell boundaries (n_zones + 1 values)
r_outer_bnds = np.linspace(0.0, r_max, n_zones + 1)
r_inner_bnds = r_outer_bnds[:-1]
r_outer_bnds = r_outer_bnds[1:]

# Calculate cell centers and volumes
r_centers = 0.5 * (r_inner_bnds + r_outer_bnds)
zone_volumes = (4.0/3.0) * np.pi * (r_outer_bnds**3 - r_inner_bnds**3)
v_centers = r_centers / t_exp # Velocity at cell centers

# --- 4. CALCULATE THE DENSITY PROFILE ---
rho_unnormalized = np.zeros(n_zones)
r_k = v_k * t_exp

# --- Inner Profile: rho ~ r^(-n_inner) ---
inner_mask = (r_centers <= r_k)
rho_unnormalized[inner_mask] = (r_centers[inner_mask] / r_k)**(-n_inner)

# --- Outer Profile: rho ~ r^(-n_outer) ---
outer_mask = (r_centers > r_k)
rho_unnormalized[outer_mask] = (r_centers[outer_mask] / r_k)**(-n_outer)

# --- 5. NORMALIZE THE DENSITY ---
mass_unnormalized = np.sum(rho_unnormalized * zone_volumes)
normalization_factor = M_ej / mass_unnormalized

# --- Add print statements here ---
print("--- Normalization Details ---")
print(f"Target Mass (M_ej):      {M_ej:e} g ({M_ej/M_sol:.3f} M_sol)")
print(f"Unnormalized Mass:       {mass_unnormalized:e} g")
print(f"Density Norm. Factor:  {normalization_factor:e}")
print("-----------------------------")

rho_final = rho_unnormalized * normalization_factor

# --- 6. DEFINE COMPOSITION ---
elements = [
    'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al',
    'si', 'p', 's', 'cl', 'ar', 'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn',
    'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
    'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
    'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe', 'cs', 'ba'
]

abundances = {}
for el in elements:
    abundances[f'{el}_mass_frac'] = np.full(n_zones, 1.0e-20)
abundances['ge_mass_frac'] = np.full(n_zones, 1.0)

# Set r-process mass fraction to 10^-2 (using 'ge' as our tracer)
"""
rproc_mass_fraction = 1.0e-4
abundances['ge_mass_frac'] = np.full(n_zones, rproc_mass_fraction)

# Set the remaining 99% to be "other" material (e.g., Iron)
abundances['fe_mass_frac'] = np.full(n_zones, 1.0 - rproc_mass_fraction)
# Note: Sedona's radiation transport will re-normalize abundances
# in each zone, so this is equivalent to 100% Germanium.

"""
# --- 6b. DEFINE R-PROCESS HEATING PARAMETERS ---
rproc_X = np.full(n_zones, 0.001)
rproc_Ye = np.full(n_zones, 0.2)

# --- 7. WRITE THE HDF5 MODEL FILE ---
# --- Define base filenames ---
base_h5_name = 'kn_physical_powerlaw'
base_plot_name = 'kn_physical_profile'
h5_ext = '.h5'
png_ext = '.png'

# --- Find a unique filename ---
i = 1
output_filename_h5 = f"{base_h5_name}{h5_ext}"
output_filename_plot = f"{base_plot_name}{png_ext}"

while os.path.exists(output_filename_h5) or os.path.exists(output_filename_plot):
    i += 1
    suffix = f"_{i}"
    output_filename_h5 = f"{base_h5_name}{suffix}{h5_ext}"
    output_filename_plot = f"{base_plot_name}{suffix}{png_ext}"

# --- Write the file ---
with h5py.File(output_filename_h5, 'w') as f:
    print(f"Writing to HDF5 file: {output_filename_h5}")

    grid = f.create_group('grid')
    grid.create_dataset('r_inner', data=r_inner_bnds)
    grid.create_dataset('r_outer', data=r_outer_bnds)

    rproc_group = f.create_group('Xrproc')
    rproc_group.create_dataset('X', data=rproc_X)
    rproc_group.create_dataset('Y_e', data=rproc_Ye)

    f.create_dataset('rho', data=rho_final)
    f.create_dataset('t_exp', data=t_exp)
    f.create_dataset('M_ej', data=M_ej)
    f.create_dataset('v_max', data=v_max) # Save the calculated v_max

    for key, data in abundances.items():
        f.create_dataset(key, data=data)

print("HDF5 model file created successfully.")

# --- 8. PLOT THE DENSITY PROFILE ---
print("Generating density plot...")
plt.figure(figsize=(10, 7))

v_plot = v_centers / c
plt.plot(v_plot, rho_final, 'b-', label='Final Density Profile')

# --- Add lines to verify the slopes ---
rho_k = rho_final[inner_mask][-1]
v_inner_plot = v_plot[inner_mask]
plt.plot(v_inner_plot, rho_k * (v_inner_plot / (v_k/c))**(-n_inner),
         'r--', label=f'n_inner = {n_inner} slope')

v_outer_plot = v_plot[outer_mask]
plt.plot(v_outer_plot, rho_k * (v_outer_plot / (v_k/c))**(-n_outer),
         'g--', label=f'n_outer = {n_outer} slope')

plt.axvline(v_k / c, color='k', linestyle=':',
            label=f'v_k = {v_k/c:.2f}c')

# --- Add marker for the 10^-3 drop-off point ---
rho_at_vmax = rho_final[-1]
rho_at_vk = rho_final[inner_mask][-1]
plt.axhline(rho_at_vk * 1e-3, color='purple', linestyle=':',
            label=r'$\rho(v_k) \times 10^{-3}$')
plt.plot(v_max/c, rho_at_vmax, 'rx', markersize=10,
         label=f'Outer edge v_max\n(rho_drop = {rho_at_vmax/rho_at_vk:.1e})')


plt.title(f'Kilonova Physical Power-Law (M={M_ej/M_sol:.2f} M_sol)')
plt.xlabel('Velocity (units of c)')
plt.ylabel('Density (g / cm$^3$)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.savefig(output_filename_plot)

print(f"Plot saved to '{output_filename_plot}'.")