#!/usr/bin/python3

# Print values to the screen
# Use this script from the command line with the
# name of the binary file you want to read

import numpy as np
import sys

with open(sys.argv[1], "rb") as binary_file:
    # Read the whole file at once
    data = binary_file.read()

print(np.frombuffer(data, np.float64))
