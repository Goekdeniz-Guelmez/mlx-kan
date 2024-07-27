import sys
from pathlib import Path
from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Add the package directory to the Python path
package_dir = root_dir / "kan"

# Read the requirements from the requirements.txt file
with open(package_dir / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

# Import the version from the package
version = {}
with open(str(package_dir / "version.py")) as f:
    exec(f.read(), version)

# Setup configuration
setup(
    name="mlx-kan",
    version=version['__version__'],
    description="KAN: Kolmogorov–Arnold Networks on Apple silicon with MLX.",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/mlx-kan",
    license="Apache-2.0",
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS :: MacOS X"
    ],
)