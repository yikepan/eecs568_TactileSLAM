# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="neuralfeels",
    version="0.0.1",
    author="Meta Research",
    description="Neural Feels.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/neuralfeels",
    packages=["neuralfeels"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyserial==3.5",
        "betterproto==2.0.0b5",
        "cobs==1.2.0",
        "google-api-python-client==2.97.0",
        "google-auth-httplib2==0.1.0",
        "google-auth-oauthlib==0.5.0",
    ],
)
