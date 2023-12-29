#!/bin/bash

# Launch this script in the same folder of dblp.v12.json

# Split the bigger dataset into smaller files with 200k lines each
split -l 200000 dblp.v12.json split_ --verbose

# Copy them in a folder called "dataset"
mkdir dataset/
mv split_a* dataset/
cd dataset/

# Since the first line of the big dataset contains a "["
# and the last line contains a "]", in order to correctly reading
# the json records we need to remove those character
# They are at the start of the first file split_aa
# and in the file split_ay. Moreover the split_ay file only contains
# the character "]", so it's sufficien to eliminate it

# Removing "[" from the start of split_aa, using a temporary file
tail -c +4 split_aa > split_aa2
rm split_aa
mv split_aa2 split_aa
# Deleting split_ay because it contains only "]"
rm split_ay

# In the folder there will be 24 files named from split_aa to split_ax

# Now we have to consider the json format inside those files.
# Pandas will not be able to read them because it's in an incorrect format.
# Each line is a record, but each line also starts with a comma "," and that
# is the problem. 
# We must remove those commas to make pandas able to read the file
sed -i -e 's/^,//g' *
echo "File splitting finished"
