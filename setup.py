from setuptools import setup, find_packages

setup(
    name='GaussianAdaptiveAttention',
    version='0.1.0',
    author='Georgios Ioannides',
    author_email='gioannid@alumni.cmu.edu',
    packages=find_packages(),
    description='A Python library implementing Gaussian Adaptive Attention in PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'torch==2.0.0+cu117',
    ],
    url='https://github.com/gioannides/Gaussian-Adaptive-Attention',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

