#!/bin/bash

# define the name of the folder for the environment
# and the name of the dependencies file
VENV="venv"
REQUIREMENTS="requirements.txt"

# Check if the virtual environment directory already exists
if [ -d "$VENV" ]; then
    rm -r $VENV
    echo -e '\n** Older environment found and deleted.'
fi

#Declare python version in pyenv
pyenv local 3.10.10

# Create a new virtual environment
python -m venv $VENV

# Activate the virtual environment
source venv/bin/activate
echo -e '\n** New environment created.\n'

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r $REQUIREMENTS
echo -e '\n** Requirements installed.\n'