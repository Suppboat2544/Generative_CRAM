"""
CRAM Generator - Setup Configuration
=====================================
Installation: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="generative-cram",
    version="1.0.0",
    author="Klabklaydee Supaporn",
    description="Deep Learning Generator for Carboxyl-Rich Alicyclic Molecules (CRAM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Suppboat2544/Generative_CRAM",
    project_urls={
        "Bug Tracker": "https://github.com/Suppboat2544/Generative_CRAM/issues",
        "Documentation": "https://github.com/Suppboat2544/Generative_CRAM#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            # DGL GPU version needs manual installation
        ],
    },
    entry_points={
        "console_scripts": [
            "cram-generate=main:main",
            "cram-validate=src.validator:main",
            "cram-ui=ui.app:launch_app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    keywords=[
        "cram",
        "chemistry",
        "molecular-generation",
        "deep-learning",
        "marine-dom",
        "dissolved-organic-matter",
        "vae",
        "transformer",
        "graph-neural-network",
    ],
)
