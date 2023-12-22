from distutils.core import setup

setup(
    name='image_tools',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.2.24',
    packages=['image_tools','image_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='image processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "opencv-python",
        "scikit-image",
        "PyQt5",
        "geometry @ git+https://github.com/ElTinmar/geometry.git@main",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main"
    ],
    extras_require = {
        'gpu': ["cupy", "cucim"]
    }
)
