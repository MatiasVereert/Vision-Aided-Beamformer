# Unofficial installation script for the project, designed to be run in a clean Python environment.

# Stop execution immediately if any command fails
set -e

# Install SRMRpy first to ensure the base dependency is available in the environment
pip install git+https://github.com/jfsantos/SRMRpy.git

# Install academic packages sequentially to bypass pip's strict dependency graph resolver
pip install git+https://github.com/schmiph2/pysepm.git
pip install git+https://github.com/fgnt/pb_bss.git
pip install git+https://github.com/fgnt/nara_wpe.git

# Install the local project and all remaining official dependencies from pyproject.toml
pip install -e .

# Print a success message when finished
echo "Environment setup completed successfully!"