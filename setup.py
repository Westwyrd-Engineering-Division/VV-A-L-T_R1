"""
V.V.A.L.T Setup Script
"""

from setuptools import setup, find_packages
import os


def read_file(filename):
    """Read file contents."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


setup(
    name="vvalt",
    version="0.1.0",
    author="V.V.A.L.T Contributors",
    description="Vantage-Vector Autonomous Logic Transformer - A deterministic, bounded logic reasoning system",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/VValtDisney/V.V.A.L.T",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
        "pytorch": [
            "torch>=1.9.0",
        ],
        "transformers": [
            "torch>=1.9.0",
            "transformers>=4.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vvalt=vvalt.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
