# Verify if OS is Windows
if ($env:OS -notlike "Windows*") {
    Write-Host "Este script está diseñado para ejecutarse en sistemas Windows."
    exit
}

# define the name of the folder for the environment
# and the name of the dependencies file
$VENV="venv"
$REQUIREMENTS="requirements.txt"

# Check if the virtual environment directory already exists
if (Test-Path $VENV) {
    Remove-Item -Path $VENV -Recurse -Force
    Write-Host "Older environment found and deleted."
}

#Declare python version in pyenv
pyenv local 3.10.10

# Create a new virtual environment
python -m venv $VENV

#  Activate the virtual environment
$activateScript = ".\$VENV\Scripts\activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Python virtual environment activated."
} else {
    Write-Host "The activation script was not found. Please, verify if the virtual environment was created correctly."
}

# Upgrade pip and install requirements
python.exe -m pip install --upgrade pip
pip install -r $REQUIREMENTS
Write-Host 'Requirements installed.'