import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
"""
import mendeleev as mlv
atoms = mlv.element(np.arange(1, 93, 1).tolist())
atomic_weights = np.full(93, np.nan)
for i in range(len(atoms)):
    atomic_weights[i+1] = atoms[i].mass # offset because element 0 dne
"""
# --- DEFINE PHYSICAL CONSTANTS (in CGS) ---
M_sol = 1.989e33  # g
c = 2.9979e10     # cm/s
day = 86400.0       # s

# --- DEFINE KN MODEL PARAMETERS ---
M_ej = 0.03 * M_sol   # Total ejecta mass
v_k = 0.1 * c         # Characteristic "break" velocity
n_inner = 1.0         # Inner density power-law index
n_outer = 10.0        # Outer density power-law index
v_t = ()

# ---  DEFINE COMPUTATIONAL GRID ---
#define the model at a reference time, t_exp.
# 1 day time step seems reasonable for KN (i think)
t_exp = 1.0 * day

n_zones = 200         # num of radial zones
v_max = 0.4 * c       # Maximum velocity of the grid
r_max = v_max * t_exp # Maximum radius of the grid at t_exp

# Create the radial grid cell boundaries (n_zones + 1 values)
r_outer_bnds = np.linspace(0.0, r_max, n_zones + 1)
r_inner_bnds = r_outer_bnds[:-1]
r_outer_bnds = r_outer_bnds[1:]

# Calculate cell centers and volumes
r_centers = 0.5 * (r_inner_bnds + r_outer_bnds)
zone_volumes = (4.0/3.0) * np.pi * (r_outer_bnds**3 - r_inner_bnds**3) #Thinking of it as a spherical shell, outer volume minus inner volume
v_centers = r_centers / t_exp # Velocity at cell centers

# --- CALCULATE THE DENSITY PROFILE ---

# This is an "unnormalized" density profile. Just to get the shape
# Needs to be scaled by a normalization constant later to get the total mass right.
rho_unnormalized = np.zeros(n_zones)

# Find the radius corresponding to the break velocity
r_k = v_k * t_exp

# --- Inner Profile: rho ~ r^(-n_inner) ---
# We can also express this in velocity: (v/v_k)^(-n_inner)
inner_mask = (r_centers <= r_k)
rho_unnormalized[inner_mask] = (r_centers[inner_mask] / r_k)**(-n_inner)

# --- Outer Profile: rho ~ r^(-n_outer) ---
# (v/v_k)^(-n_outer)
outer_mask = (r_centers > r_k)
rho_unnormalized[outer_mask] = (r_centers[outer_mask] / r_k)**(-n_outer)

# --- 5. NORMALIZE THE DENSITY ---
# Calculate the total mass with our unnormalized density
mass_unnormalized = np.sum(rho_unnormalized * zone_volumes)

# Find the normalization constant
normalization_factor = M_ej / mass_unnormalized

# Apply the normalization to get the final density
rho_final = rho_unnormalized * normalization_factor

# --- 6. DEFINE COMPOSITION ---
# Create mass fraction arrays for all elements Sedona knows.
# For a Kilonova, this is a simplification. We'll just put all
# the mass in one heavy r-process element as a tracer.
# Let's use Germanium ('ge') as a stand-in for "r-process".
# The 'make_sedona_model.py' from the Ia example shows all the
# element keys Sedona expects.

# List of all elements Sedona tracks

elements = [
    'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al',
    'si', 'p', 's', 'cl', 'ar', 'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn',
    'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
    'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
    'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe', 'cs', 'ba'
]

# Create a dictionary to hold the mass fraction arrays
abundances = {}
for el in elements:
    # Initialize all mass fractions to a tiny number (or zero)
    abundances[f'{el}_mass_frac'] = np.full(n_zones, 1.0e-20)

# Set r-process mass fraction to 10^-2 (using 'ge' as our tracer)
rproc_mass_fraction = 1.0e-4
abundances['ge_mass_frac'] = np.full(n_zones, rproc_mass_fraction)

# Set the remaining 99% to be "other" material (e.g., Iron)
abundances['fe_mass_frac'] = np.full(n_zones, 1.0 - rproc_mass_fraction)
# Note: Sedona's radiation transport will re-normalize abundances
# in each zone, so this is equivalent to 100% Germanium.

# --- DEFINE R-PROCESS HEATING PARAMETERS ---
# set a representative electron fraction 
#rproc_X = np.full(n_zones, rproc_mass_fraction)
rproc_Ye = np.full(n_zones, 0.1) # Typical value would range between .03 and 0.4

# Normalization constant details because I'm curious      
print("--- Normalization Details ---")
print(f"Target Mass (M_ej):      {M_ej:e} g ({M_ej/M_sol:.3f} M_sol)")
print(f"Unnormalized Mass:       {mass_unnormalized:e} g")
print(f"Density Norm. Factor:  {normalization_factor:e}")


# --- 7. WRITE THE HDF5 MODEL FILE ---

output_filename = 'kn_bpl_m03_v01_X0001.h5'

with h5py.File(output_filename, 'w') as f:
    print(f"Writing to HDF5 file: {output_filename}")

    # --- Write grid information ---
    # Sedona expects the grid in a group named 'grid'
    grid = f.create_group('grid')
    grid.create_dataset('r_inner', data=r_inner_bnds)
    grid.create_dataset('r_outer', data=r_outer_bnds)

    # --- Write r-process heating information ---
    # This group tells Sedona what fraction of the mass
    # contributes to r-process heating.
    rproc_group = f.create_group('Xrproc')
    rproc_group.create_dataset('X', data=rproc_X)
    rproc_group.create_dataset('Y_e', data=rproc_Ye)

    # --- Write primary model data ---
    f.create_dataset('rho', data=rho_final)
    f.create_dataset('t_exp', data=t_exp)
    f.create_dataset('M_ej', data=M_ej)
    f.create_dataset('v_max', data=v_max)

    # --- Write abundances ---
    # Save each element's mass fraction array
    for key, data in abundances.items():
        f.create_dataset(key, data=data)

print("HDF5 model file created successfully.")

# --- 8. PLOT THE DENSITY PROFILE (Your Main Goal) ---
print("Generating density plot...")
plt.figure(figsize=(10, 7))

# Plot density vs. velocity in units of c
v_plot = v_centers / c
plt.plot(v_plot, rho_final, 'b-', label='Final Density Profile')

# --- Add lines to verify the slopes ---
# Find the density at the break
rho_k = rho_final[inner_mask][-1]

# Inner slope line
v_inner_plot = v_plot[inner_mask]
plt.plot(v_inner_plot, rho_k * (v_inner_plot / (v_k/c))**(-n_inner),
         'r--', label=f'n_inner = {n_inner} slope')

# Outer slope line
v_outer_plot = v_plot[outer_mask]
plt.plot(v_outer_plot, rho_k * (v_outer_plot / (v_k/c))**(-n_outer),
         'g--', label=f'n_outer = {n_outer} slope')

# Add a vertical line at the break velocity
plt.axvline(v_k / c, color='k', linestyle=':',
            label=f'v_k = {v_k/c:.2f}c')

plt.title(f'Kilonova Broken Power-Law Density (M={M_ej/M_sol:.2f} M_sol)')
plt.xlabel('Velocity (units of c)')
plt.ylabel('Density (g / cm$^3$)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Use the unique filename for the plot
plt.savefig(output_filename_plot)

print(f"Plot saved to '{output_filename_plot}'.")