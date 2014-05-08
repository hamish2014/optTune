from distutils.core import setup 

setup(name='optTune',
      description='Various algorithms for tuning an optimization algorithm under multiple objective function evaluation budgets.',
      long_description='The optimization tuning (optTune) Python package contains various tuning algorithms. Many of these algorithms are designed for tuning an optimization algorithm under multiple objective function evaluation budgets.',
      version='0.1.0',
      packages=[
        'optTune',
        'optTune.tMOPSO_code',
        'optTune.paretoArchives',
        'optTune.MOTA_code'],
      author="Antoine Dymond",
      author_email="antoine.dymond@gmail.com",
#      url="https://pythonhosted.org/optTune/",
      url="https://github.com/hamish2014/optTune/",
      license="GPLv3",
      #extra meta data
      classifiers = [  #taken from http://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 3 - Alpha", 
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering" 
        ],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib'
      ],

)
