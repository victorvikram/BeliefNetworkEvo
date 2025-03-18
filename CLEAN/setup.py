from setuptools import setup, find_packages

# Install all required packages from requirements.txt
# pip install -r requirements.txt

setup(
    name="clean-gss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "networkx>=3.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.3.0",  # Required for partial correlations
        "sas7bdat>=2.2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.6",
        ],
    },
    python_requires=">=3.9",
) 