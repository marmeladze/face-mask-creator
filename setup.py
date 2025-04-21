from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-mask-creator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple library for creating face masks from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/face-mask-creator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "dlib>=19.22.0",
    ],
    package_data={
        "face_mask_creator": ["models/*"],
    },
    py_modules=["face_mask_creator.utils"],
) 