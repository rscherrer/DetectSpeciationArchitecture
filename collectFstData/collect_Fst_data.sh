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
        # this is specific to the simulations we're currently doing
        echo "No time provided, using default timestep 3000"
        $2 = 3000
    fi

    if [ -f "$1" ]; then
        echo "Overwriting file..."
        rm $1
    else
        echo "Creating file..."
    fi

    # find the range of elements to be copied
    # (python code writes these values to readPositions.txt)
    python3 get_Fst_range.py parameters.txt $2
    # read start and end positions from txt file
    read STARTPOS ENDPOS < readPositions.txt
    echo $STARTPOS
    echo $ENDPOS

    # for each sim folder, copy Fst data for the given range in file
    # appends each bit of data into the file
    for folder in $(ls -d sim*)
    do
        if [ -f "genome_Fst.dat" ]; then
            python3 copy_data.py $1 $folder/genome_Fst.dat $STARTPOS $ENDPOS
        fi
    done
    # delete readPositions, don't need it anymore
    rm readPositions.txt

else
    echo "Please provide file to write to"

fi