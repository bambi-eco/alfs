from setuptools import setup, find_namespace_packages
from alfr.globals import __version__

setup(
    name="alfr",
    version=__version__,
    description="A cross-platform light field renderer written in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/schedldave/alfr",
    author="David C. Schedl",
    author_email="david.schedl@fh-hagenberg.at",
    packages=find_namespace_packages(include=["alfr", "alfr.*"]),
    include_package_data=True,
    # keywords=["moderngl", "window", "context"],
    # license="MIT", # Todo: license
    platforms=["any"],
    python_requires=">=3.6",
    classifiers=[
        "Operating System :: OS Independent",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3 :: Only",
    ],
    install_requires=[
        "moderngl>=5.6",
        "numpy>=1.16,<2",
        "pyrr>=0.10.3,<1",
        "opencv-python>=4.5",
    ],
    extras_require={
        "PySide6": ["PySide6>=6.2"],
    },
    project_urls={
        # "Documentation": "https://moderngl-window.readthedocs.io",
        "ALFR": "https://github.com/schedldave/alfr",
    },
)
