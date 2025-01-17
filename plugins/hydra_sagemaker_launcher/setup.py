# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="hydra-sagemaker-launcher",
    version=read_version("hydra_plugins/hydra_sagemaker_launcher", "__init__.py"),
    author="Dennis Bappert",
    author_email="bappert@amazon.de",
    description="Amazon SageMaker Launcher for Hydra apps",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://aws.amazon.com/sagemaker/",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "hydra-core>=1.1.0.dev7",
        "sagemaker>=2.74.0",
        "cloudpickle==1.6.0",
        "pickle5==0.0.11",
        "python-dotenv==0.19.2",
    ],
    include_package_data=True,
)
