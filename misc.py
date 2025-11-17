# --- WRITE THE HDF5 MODEL FILE W/ Unique Name---

# --- Define base filenames ---
base_h5_name = 'kn_broken_powerlaw'
base_plot_name = 'kn_density_profile'
h5_ext = '.h5'
png_ext = '.png'

# --- Find a unique filename ---
i = 1
# Check for the base file names first
output_filename_h5 = f"{base_h5_name}{h5_ext}"
output_filename_plot = f"{base_plot_name}{png_ext}"

# Loop while *either* file exists, finding a unique suffix
while os.path.exists(output_filename_h5) or os.path.exists(output_filename_plot):
    i += 1
    suffix = f"_{i}"
    output_filename_h5 = f"{base_h5_name}{suffix}{h5_ext}"
    output_filename_plot = f"{base_plot_name}{suffix}{png_ext}"

# --- Now, output_filename_h5 and output_filename_plot are unique ---

with h5py.File(output_filename_h5, 'w') as f:
    print(f"Writing to HDF5 file: {output_filename_h5}")

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
