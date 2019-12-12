#!/bin/bash

# copies genome_Fst.dat for the given timestep into given file
# arguments: file to write to, timestep
# run from the folder created by build_target.sh

# assumes all simulations are using the same: 
#               total number of genes
#               tsave
#               tend

# assumes parameter file is in current dir

if [ "$1" != "" ]; then

    if [ "$2" == "" ]; then
        echo "No time provided, using default timestep 0"
        $2 = 0
    fi

    if [ -f "$1" ]; then
        echo "Overwriting file..."
        rm $1
    else
        echo "Creating file..."
    fi

    # find the range of elements to be copied
    # (python code writes these values to readPositions.txt)
    # there is probably a better wya to do that right? maybe do everything in python?
    python3 get_Fst_range.py parameters.txt $2
    # read start and end positions from txt file
    read STARTPOS ENDPOS < readPositions.txt
    echo $STARTPOS
    echo $ENDPOS

    # for each sim folder, copy Fst data for the given range in file
    # appends each bit of data into the file
    for folder in $(ls -d sim*)
    do
        python3 copy_data.py $1 $folder/genome_Fst.dat $STARTPOS $ENDPOS
    done

else
    echo "Please provide file to write to"

fi