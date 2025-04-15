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
        "pyyaml",
        "tabulate",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "trend-analysis=trend_analysis.cli:main"
        ]
    },
    author="Your Name",
    description="Trend and feature importance analysis tool",
    keywords="data-analysis shap sklearn anova regression feature-importance",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    python_requires='>=3.7',
)