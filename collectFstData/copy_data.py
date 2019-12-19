# reads in .dat file as list of floats
# copies elements [startPos:endPos] in another .dat file
# APPENDS, not overwrites the file (bash script clears file first)
#
# arguments: file to write to (.dat),
#            file to read from (.dat)
#            start position 
#            end position

import sys
import numpy as np

# could improve checking here
# if not enough parameters, provide some defaults
# check files are .dat, check positions are ints, etc
if len(sys.argv) < 5:
    raise Exception("Not enough arguments provided.")
if len(sys.argv) > 5:
    raise Exception("Too many arguents provided.")

writeFileName = sys.argv[1]
readFileName = sys.argv[2]
startPos = int(sys.argv[3])
endPos = int(sys.argv[4])

#print("writing to", writeFileName)
#print("reading from", readFileName)

readFolderName = readFileName
with open(readFolderName, "rb") as binary_file:
    data = binary_file.read()

# how to check this won't go outside the array?
# without frombuffering the whole file first
# ... may just have to do that i guess
chosenFst = np.frombuffer(data, np.float64)[startPos:endPos]

with open(writeFileName, "ab") as f:
    f.write(chosenFst)