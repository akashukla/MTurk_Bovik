# 
# Helpful auto commit script.
#
# Functionality: eiher pass in commit objects
# such as files or directories and it will 
# commit them OR it will default to adding all
# untracked and pre-staged objects then doing a
# commit and push.
#
# Note: This script will automatically perform a git pull
# before initiating the commit. The script will die
# in the presence of merge conflicts so you can 
# resolve them.
#
# Author: MK Swaminathan
#

#!/bin/bash

COMMIT_OBJ=$@
echo "Using MK's easy commit script. Warning: if empty target, this will auto add all untracked objects and commit those along with pre-staged changes to the current branch head."

echo "Pulling latest changes from current head..."
git pull

if [ -z "$@" ]; then
    echo "Empty target."
else 
    # Adding all your files or commit objects
    echo "Adding your commit objects..."
    for co in $COMMIT_OBJ; do
        git add "$co"
    done
    git commit -a -m "Testing MK's auto commit script."
   
fi
git push
