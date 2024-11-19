from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

version = "0.0.1"

setup(
    name="src",
    version=version,
    description="GROOT: Effective Design of Biological Sequences with Limited Experimental Data.",
    long_description=long_description,
    packages=find_packages(),
)
