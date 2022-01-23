import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyutai",
    version="0.0.2",
    author="UTAI Group",
    description="A Python implementation of Potentials Tree.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google/glome",
    packages=["pyglome"],
    install_requires=[
        "pgpmpy",
        "numpy",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.8',
)
