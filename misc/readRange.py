#!/usr/bin/python3

# Print values to the screen
# Use this script from the command line with the
# name of the binary file you want to read
# give start and pos for range (required, im not doing input checking for this ok)

import numpy as np
import sys

startPos = int(sys.argv[2])
endPos = int(sys.argv[3])

with open(sys.argv[1], "rb") as binary_file:
    # Read the whole file at once
    data = binary_file.read()

print(np.frombuffer(data, np.float64)[startPos:endPos])
