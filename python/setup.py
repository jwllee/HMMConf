from glob import glob
from os.path import splitext, basename
from setuptools import find_packages, setup


requirements = [

]


setup(
    name='hmmconf',
    version='0.1.0',
    description='HMM-based conformance checker',
    author='Wai Lam Jonathan Lee',
    author_email='walee@uc.cl',
    install_requires=requirements,
    packages=find_packages('.')
)
