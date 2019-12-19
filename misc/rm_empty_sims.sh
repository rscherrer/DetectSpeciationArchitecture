#! bin/bash

# run from folder with all the sim folders
# deletes any sim folder that doesn't contain "genome_Fst.dat"

for folder in $(ls -d sim*)
do
    cd folder
    if [ -f "genome_Fst.dat" ]; then
        cd ..
        rm -r folder
    else
        cd ..
    fi
done