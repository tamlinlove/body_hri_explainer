from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['body_hri_explainer'],
    package_dir={'': 'src'}
)

setup(**d)