import setuptools

setuptools.setup(
    name="SRWF",
    version="0.1",
    author="saavn02",
    description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "srwf"},
    packages=setuptools.find_packages(where="srwf"),
    python_requires=">=3.12",
)