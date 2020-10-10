from setuptools import setup, find_packages

setup(
    name='neurora',
    version='1.1.4.0',
    description=(
        'A Python Toolbox for Multimode Neural Data Representation Analysis'
    ),
    long_description=open('README.md').read(),
    author='Zitong Lu',
    author_email='zitonglu1996@gmail.com',
    maintainer='Zitong Lu',
    maintainer_email='zitonglu1996@gmail.com',
    license='MIT License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/neurora/NeuroRA',
    install_requires=[
        'numpy',
        'scipy>=1.4',
        'mne',
        'nibabel',
        'matplotlib',
        'nilearn'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    package_data={'neurora': ['template/*.gz']},
)
