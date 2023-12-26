from setuptools import setup, find_packages

setup(
    name = "alan",
    version = "0.0.1",
    keywords = ("test", "xxx"),
    license = "MIT Licence",
    packages=['alan'],
    package_dir={'':'src'},
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_requires=[
        "numpy",
    ],
    include_package_data = True,
    platforms = "any",
)
