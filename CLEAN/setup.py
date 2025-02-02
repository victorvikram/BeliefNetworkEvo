from setuptools import setup, find_packages

# Install all required packages from requirements.txt
# pip install -r requirements.txt

setup(
    name="clean-gss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "networkx>=2.8.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scipy>=1.9.0",
        "sas7bdat>=2.2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
    python_requires=">=3.9",
) 