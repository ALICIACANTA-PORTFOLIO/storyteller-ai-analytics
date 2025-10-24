"""
============================================================================
AUDIOMIND - Setup Configuration
============================================================================
Purpose: Legacy setup.py for backward compatibility
Recommendation: Use pyproject.toml for modern Python packaging (PEP 621)
============================================================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

# Read requirements
REQUIREMENTS = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS.exists():
    with open(REQUIREMENTS, encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="audiomind",
    version="1.0.0",
    author="Alicia Canta",
    author_email="your.email@example.com",
    description="AI-Powered Audio Knowledge Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audiomind",
    project_urls={
        "Documentation": "https://audiomind.readthedocs.io",
        "Source": "https://github.com/yourusername/audiomind",
        "Tracker": "https://github.com/yourusername/audiomind/issues",
    },
    packages=find_packages(exclude=["tests*", "docs*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "mkdocstrings[python]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audiomind=app.cli:main",
            "audiomind-api=app.main:run_api",
            "audiomind-worker=app.worker:run_worker",
            "audiomind-dashboard=app.dashboard.main:run",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# ============================================================================
# NOTES:
# 
# Installation:
#   pip install -e .              # Editable/development install
#   pip install -e ".[dev]"       # With dev dependencies
#   python setup.py install       # Standard install (not recommended)
# 
# Modern Alternative:
#   Use pyproject.toml (PEP 621) instead of setup.py
#   This file is kept for backward compatibility only
# ============================================================================
