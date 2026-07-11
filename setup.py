from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="biofilm_admm",
    version="0.1.0",
    author="Runtime-Slayers",
    author_email="contact@runtimeslayers.org",
    description="Biofilm Segmentation and Classification using ADMM and Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Runtime-Slayers/biofilm-segmentation-admm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "opencv-python",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "tqdm",
    ],
)
