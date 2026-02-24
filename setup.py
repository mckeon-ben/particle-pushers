import setuptools

with open('README.md', 'r') as fh:
    description = fh.read()

setuptools.setup(
    name='particle-pushers',
    version='0.0.1',
    author='Ben McKeon',
    author_email='mckeon.ben@ul.ie',
    packages=['particle_pushers'],
    description='A simple Python package containing various relativistic particle pushers.',
    long_description=description,
    long_description_content_type='text/markdown',
    url='https://github.com/mckeon-ben/particle-pushers.git',
    license='MIT',
    python_requires='>=3.10',
    install_requires=['numpy', 'scipy']
)
