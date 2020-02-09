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
