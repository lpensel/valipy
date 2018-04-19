
from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(
    name='ValiPy',
    version='0.1.0dev1',
    description='A machine learning model validation toolkit for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lpensel/valipy',
    author='Lukas Pensel',
    author_email='lukas@familiepensel.de',


    classifiers=[ 
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        #'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    #keywords='?? ?? ??',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'dist']),

    install_requires=['h5py','numpy','scikit-learn','Keras','six','joblib'],
    python_requires='>=3',
    #py_modules=["six"],
)