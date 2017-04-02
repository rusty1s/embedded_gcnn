# Embedded Graph Convolutional Neural Networks

[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Requirements Status][requirements-image]][requirements-url]
[![Code Climate][code-climate-image]][code-climate-url]
[![Code Climate Issues][code-climate-issues-image]][code-climate-issues-url]

[build-image]: https://travis-ci.org/rusty1s/embedded_gcnn.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/embedded_gcnn
[coverage-image]: https://img.shields.io/codecov/c/github/rusty1s/embedded_gcnn.svg
[coverage-url]: https://codecov.io/github/rusty1s/embedded_gcnn?branch=master
[requirements-image]: https://requires.io/github/rusty1s/embedded_gcnn/requirements.svg?branch=master
[requirements-url]: https://requires.io/github/rusty1s/embedded_gcnn/requirements/?branch=master
[code-climate-image]: https://codeclimate.com/github/rusty1s/embedded_gcnn/badges/gpa.svg
[code-climate-url]: https://codeclimate.com/github/rusty1s/embedded_gcnn
[code-climate-issues-image]: https://codeclimate.com/github/rusty1s/embedded_gcnn/badges/issue_count.svg
[code-climate-issues-url]: https://codeclimate.com/github/rusty1s/embedded_gcnn/issues

## PascalVOC

In order to run the `PascalVOC` dataset you need to [download the test data from THE VOC2012 challenge](http://host.robots.ox.ac.uk:8080/) separately.
You need to create an account to do so.
Then pass in the path to the `VOC2012` folder in the `test_dir` parameter of the `PascalVOC` constructor.
