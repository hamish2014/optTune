#! /usr/bin/env bash

make -B html
cd _build/html
echo zip html_docs.zip -rq \*
zip html_docs.zip -rq *
cd -
mv _build/html/html_docs.zip ./

echo
echo now upload html_docs.zip on https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=optTune