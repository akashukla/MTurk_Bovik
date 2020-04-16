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
MSG=""
echo "Using MK's easy commit script. Warning: if empty target, this will auto add all untracked objects and commit those along with pre-staged changes to the current branch head."

while getopts ":c:m:" opt; do
    case ${opt} in
        c )
            echo "Adding your commit objects..."
            # Adding all your files or commit objects
            for co in ${OPTARG}; do
                git add "$co"
                echo "$co"
            done
            ;;
        m )
            echo "Adding your commit message ..."
            MSG=${OPTARG}
            echo "$MSG"
            ;;
        \? )
            echo "Invalid option -${OPTARG}" >&2
            ;;
        : )
            echo "Option -${OPTARG} requires an argument." >&2
            exit 1
            ;;
    esac
done

echo "Performing commit"
git commit -a -m "$MSG: Using MK's auto commit tool."

#if [ -z "$@" ]; then
#    echo "Empty target."
#else 
#fi

echo "Pulling latest changes from current head..."
git pull

git push
