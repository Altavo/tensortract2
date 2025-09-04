from setuptools import setup
from setuptools import find_packages

install_requires = [
    'einops>=0.8.1',
    'numpy',
    'pytorch-tcn==1.2.3',
    'requests>=2.32.3',
    'target-approximation>=0.0.5',
    'torch>=2.6.0',
    'torchaudio>=2.6.0',
    'transformers>=4.50.3',
    'tqdm',
    'soundfile>=0.13.1',
    ]

setup(
    name='tensortract2',
    version='0.0.4',
    description='A PyTorch implementation of TensorTract2',
    author='Paul Krug',
    url='https://github.com/Altavo/tensortract2',
    license='Commons Clause License Condition v1.0',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=find_packages(include=["tensortract2", "tensortract2.*"]),
    package_data={'tensortract2': ['cfg/tensortract2_version_uc81_am100.yaml']},
    install_requires=install_requires,
)