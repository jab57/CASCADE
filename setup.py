from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="cascade",
    version="0.1.0",
    author="Jose Bird",
    author_email="jbird@birdaisolutions.com",
    description="MCP server for in silico gene perturbation analysis in cancer and immuno-oncology research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jab57/CASCADE",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "env", "env.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "scipy",
        "fastmcp",
        "requests",
        "langgraph",
        "gremln @ git+https://github.com/czi-ai/GREmLN.git",
    ],
    extras_require={
        "llm": ["ollama>=0.1.0"],
        "test": ["pytest", "pytest-cov", "pytest-asyncio"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
