from distutils.core import setup

setup(
    name='image_tools',
    python_requires='>=3.7',
    author='Martin Privat',
    version='0.9.7',
    packages=['image_tools','image_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='image processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "qtpy",
        "pyqtgraph",
        "opencv-python-headless",
        "scikit-image",
        "geometry @ git+https://github.com/ElTinmar/geometry.git@v0.3.1",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@v0.5.4"
    ]
)
