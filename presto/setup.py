#!/usr/bin/env python3

from setuptools import setup

# NOTE(ycho): we'll override `ext_modules` and `cmdclass`
# later when we implement pybind11 cxx sub-modules.
# for eigen:
ext_modules = [ ]
cmdclass = { }


if __name__ == '__main__':
    setup(name='presto',
          use_scm_version=dict(
              root='../',
              relative_to=__file__,
              version_scheme='no-guess-dev'
          ),
          ext_modules=ext_modules,
          cmdclass=cmdclass,
          scripts=[]
          )
