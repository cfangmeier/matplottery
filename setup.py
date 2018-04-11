from setuptools import setup, find_packages

with open('requirements.txt') as req:
    install_requires = [l.strip() for l in req.readlines()]

setup(
    name='matplottery',
    packages=find_packages(),
    version='0.1.1',
    description='Histogram plotting with matplotlib',
    author='Nick Amin',
    url='https://github.com/aminnj/matplottery',
    install_requires=install_requires,
    keywords=['matplotlib', 'hep', 'histogram', 'plotting'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
