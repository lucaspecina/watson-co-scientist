"""
Setup script for mini-RAUL.
"""
from setuptools import setup, find_packages

setup(
    name="mini-raul",
    version="0.1.0",
    description="A minimalistic research co-scientist system",
    author="RAUL Team",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "mini-raul=src.cli.main:main",
        ],
    },
    python_requires=">=3.8",
) 