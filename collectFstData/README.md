These scripts are for collecting genome_Fst data for a given timestep into one dat file. All three are needed.

The script creates and then deletes a file `readPositions.txt`. If this file already exists IT WILL BE DELETED.

The script should be called in a directory containing `parameters.txt` and the folders (each starting with `sim`) that contain the Fst data. All simulations must have the same total number of genes, tsave value, and tend value as that in the parameter file.

# Hey raph, here's what you should run in the console
To use the default timestep 3000:
```{bash}
bash collect_Fst_data.sh <path_to_dat_file>
```

To specify the timestep:
```{bash}
bash collect_Fst_data.sh <path_to_dat_file> <timestep>
```
(`./collect_Fst_data.sh` may not work, you need to to do `bash` intstead, I don't know why)

The path is for the file you want to write the Fst data to. It should be a .dat file. It will overwrite the file if it already exists.

Timestep is an integer. If no timstep is given 3000 will be used (the `tend` for all current simulations). If the given timestep was not recorded by the sim, the next lowest timestep that was recorded will be used instead. If the given timestep is too low, the first recorded timestep will be used.

The resulting .dat file will contain the genome_Fst data at the timestep from each sim folder.