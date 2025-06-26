"""
Setup script for GraphRAG package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graphrag",
    version="0.1.0",
    author="GraphRAG Team",
    author_email="contact@graphrag.com",
    description="A comprehensive system for building knowledge graphs from documents using LLM extraction, vector databases, and NetworkX graph processing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/graphrag/graphrag",
    project_urls={
        "Bug Tracker": "https://github.com/graphrag/graphrag/issues",
        "Documentation": "https://graphrag.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "torchvision>=0.10.0+cu111",
            "torchaudio>=0.9.0+cu111",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphrag=graphrag.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="graphrag, knowledge-graph, llm, vector-database, networkx, rag",
) 