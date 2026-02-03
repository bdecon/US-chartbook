from setuptools import setup, find_packages

setup(
    name='uschartbook',
    version='0.1.7',
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'pandas',
        'numpy',
        'requests',
        'matplotlib',
        'statsmodels',
        'basemap',
    ],
)
