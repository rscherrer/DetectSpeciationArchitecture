# takes parameter file name, timestep
# works out the range of elements to take from .dat file
# to get genome at given timestep (or nearby)
# write start and end index of that range to a txt file
# for the bash script bc, uh, i don't know a better way to do that right now
# should maybe just do all the bash stuff in python (os package)

import re
import sys
import numpy as np

if len(sys.argv) < 3:
    raise Exception

parameterFile = sys.argv[1] # .txt
twrite = int(sys.argv[2]) # int, timestep to get pos for
# i should rename twrite

# Read parameter file
with open(parameterFile, "rt") as f:
    data = f.read()

# get the total number of genes
m = re.search("nvertices ", data)
if(m):
    ngenes = data[m.end():].split(' ', 3)[0:3] # get nvertices for all three traits
    ngenes = [int(i) for i in ngenes]
    ngenes = sum(ngenes) # sum nvertices
else:
    raise Exception("Couldn't find nvertices parameter.")

# get tsave (time between each write in simulation)
m = re.search("tsave ", data)
if(m):
    tsave = int(data[m.end():].split(' ', 1)[0])
else:
    raise Exception("Couldn't find tsave parameter.")

# get tend (total number of timesteps in simulation)
m = re.search("tend ", data)
if(m):
    tend = int(data[m.end():].split(' ', 1)[0])
else:
    raise Exception("Couldn't find tend parameter.")


# check twrite
# if twrite is past sim time, use last recorded time
if twrite >= tend:
    twrite = tend - (tend % tsave) - tsave
    print("Provided time is too high. Using last recorded time:", twrite)

# if twrite is before and recorded time, use the first possible
if twrite < tsave:
    twrite = tsave
    print("Provided time is too low. Using first recorded time:", twrite)

# if twrite isn't a multiple of tsave, go to the next lowest multiple
if twrite % tsave != 0:
    twrite = twrite - (twrite % tsave)
    print("Data at that time not recorded. Using next lowest time:", twrite)

# get index of twrite in the recorded times
tidx = int(twrite / tsave - 1)

# calculate start and end elements of the genome at tidx
startPos = tidx * ngenes
endPos = startPos + ngenes

# write to field for the bash script, sorry i don't know a better way to do this currently
with open("readPositions.txt", "wt") as f:
    f.write(str(startPos) + ' ' + str(endPos))