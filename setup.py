from setuptools import setup, find_packages

setup(
    name="trend_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "shap",
        "scikit-learn",
        "statsmodels",
        "patsy",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "trend-analysis=trend_analysis.cli:main"
        ]
    },
    author="Your Name",
    description="Trend and feature importance analysis tool",
    keywords="data-analysis shap sklearn anova",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
