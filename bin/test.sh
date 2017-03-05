#!/bin/sh

pep8 ./**/*.py && nosetests --with-coverage --nologcapture
