"""Setup script."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='AI Procs',
    version='0.1.0',
    description='Automated I/O-V preparation for NN',
    author='Christian Fröhlingsdorf',
    author_email='chris@5cf.de',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
    ],
    extras_require={},
    packages=find_packages(exclude=('tests',))
)