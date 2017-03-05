#!/bin/sh

pep8 ./**/*.py
flake8 ./**/*.py
nosetests --with-coverage --nologcapture
