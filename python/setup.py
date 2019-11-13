from glob import glob
from os.path import splitext, basename
from setuptools import find_packages, setup


requirements = [
    'numpy',
    'numba',
    'icc_rt',
    'pm4py',
]


setup(
    name='hmmconf',
    version='2.0.0',
    description='HMM-based conformance checker',
    author='Wai Lam Jonathan Lee',
    author_email='walee@uc.cl',
    install_requires=requirements,
    packages=find_packages('.')
)
