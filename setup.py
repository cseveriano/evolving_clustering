from setuptools import setup, find_packages

setup(
    name='evolclustering',
    package_dir={'':'src/models'},
    packages=find_packages("src/models"),
    version='0.1',
    description='Evolving Clustering',
    author='Carlos Severiano',
    author_email='carlossjr@gmail.com',
    url='https://github.com/cseveriano/evolving_clustering',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)