import os
from setuptools import find_packages, setup


install_requires = [
    "argparse",
    "matplotlib>=3.2.1",
    "numpy>=1.18.4",
    "POT>=0.7.0",
    "scanpy",
    "scikit-learn>=0.23.1",
    "scipy>=1.4.1",
    "torch>=1.5.0",
    "torchdiffeq==0.0.1",
]

version_py = os.path.join(os.path.dirname(__file__), "TrajectoryNet", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.rst").read()

setup(
    name="TrajectoryNet",
    packages=find_packages(),
    install_requires=install_requires,
    version=version,
    description="A neural ode solution for imputing trajectories between pointclouds.",
    author="Alexander Tong",
    author_email="alexandertongdev@gmail.com",
    license="MIT",
    long_description=readme,
    url="https://github.com/KrishnaswamyLab/TrajectoryNet",
)
