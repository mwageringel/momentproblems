import setuptools

VERSION = open("VERSION").read().strip()

setuptools.setup(
    name="momentproblems",
    version=VERSION,
    packages=['momentproblems',
              'momentproblems.examples'],
    )
