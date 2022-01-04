import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reqs:
    requirements = reqs.read()

setuptools.setup(
    name="distrakt",
    version="0.0.1",
    author="Robert Yenokyan",
    description="Distrakt Python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robert-yenokyan/distrakt.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[requirements]
)
