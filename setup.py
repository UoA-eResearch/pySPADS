from setuptools import setup, find_packages

setup(
    name="pySPADS",
    version="0.0.1",
    py_modules=find_packages(),
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            "pySPADS = cli:cli"
        ]
    }
)
