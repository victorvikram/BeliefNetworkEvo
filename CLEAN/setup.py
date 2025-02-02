from setuptools import setup, find_packages

# Install all required packages from requirements.txt
# pip install -r requirements.txt

setup(
    name="gss_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'networkx>=3.1',
        'scikit-learn>=1.3.0'
    ]
) 