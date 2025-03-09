from setuptools import setup, find_packages

setup(
    name="dicom_utils",  # Package name as it will be installed.
    version="0.1.0",     # Initial version.
    author="Usman Bashir",
    author_email="your.email@example.com",
    description="A package for handling DICOM files and utilities.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dicom_utils",  # Repository URL.
    packages=find_packages(),  # This finds the inner dicom_utils package automatically.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
