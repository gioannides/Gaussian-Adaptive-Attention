from setuptools import setup, find_packages

setup(
    name='gaussian_adaptive_attention',
    version='0.1.1',
    author='Georgios Ioannides',
    author_email='gioannid@alumni.cmu.edu',
    description='A Gaussian Adaptive Attention module for PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/GaussianAdaptiveAttention',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    license='Apache 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

