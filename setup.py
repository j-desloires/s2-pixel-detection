import os
from setuptools import setup, find_packages


def parse_requirements(file):
    return sorted(
        (
            {
                line.partition("#")[0].strip()
                for line in open(os.path.join(os.path.dirname(__file__), file))
            }
            - set("")
        )
    )


setup(
    name="cloudmask",
    python_requires=">=3.7",
    version="1.0.0",
    description="Cloud mask using Sentinel-2 data",
    author="Johann Desloires",
    author_email="johann.desloires@gmail.com",
    packages=find_packages(),
    package_data={"cloudmask": ["environment.yml"]},
)
