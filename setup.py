from setuptools import setup, find_packages

setup(
    name = "alan",
    version = "0.0.1",
    keywords = ("test", "xxx"),
    license = "MIT Licence",
    packages=['alan'],
    package_dir={'':'src'},
    python_requires='>=3.9.0',
    install_requires=[
        "torch>=2.0.0",
        "opt_einsum",
        "pytest",
    ],
    extras_requires=[
    ],
    include_package_data = True,
    platforms = "any",
)
