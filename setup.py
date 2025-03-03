from setuptools import find_packages, setup

setup(
    name="Fatigue Detection",
    version="1.0",
    author="Yuchen Zhou",
    author_email="zyc200187@gmail.com",
    install_requires=[],
    packages=find_packages(include=["fatigue.*"]),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "dlib"],
    },
)
