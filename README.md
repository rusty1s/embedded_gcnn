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

![SlIC and Quickshift Segmentation](image.jpg)

**TensorFlow implementation of my [Graph-based Image
Classification](https://github.com/rusty1s/deep-learning/blob/master/master/main.pdf)
master thesis *(german)***

Embedded graph convolutional neural networks aim to make significant improvements to learning on graphs where nodes are positioned on a twodimensional euclidean plane.
As proof, we implemented an image classification on embedded graphs by first segmenting the image into superpixels with the use of [SLIC](https://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf) and [Quickshift](http://vision.cs.ucla.edu/papers/vedaldiS08quick.pdf), converting this representation into a graph and inputting these to the neural network.
Graphs are trained on four different datasets and are automatically downloaded by running the corresponding train scripts:
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [TinyImageNet](https://tiny-imagenet.herokuapp.com/)
* [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/)

This repository also includes layer implementations of alternative approaches such as the [SGCNN](https://arxiv.org/abs/1312.6203) and the [GCN](https://arxiv.org/abs/1609.02907) to validate the results.

## Requirements

To install the required python packages, run:

```bash
pip install -r requirements.txt
```

## Running tests

Install the test requirements

```bash
pip install -r requirements_test.txt
```

and run the test suite:

```bash
nosetests --nologcapture
```

## Cite

Please cite my master thesis if you use this code in your own work:

```
@mastersthesis{fey2017egcn,
  title={Convolutional Neural Networks auf Graphrepr{\"a}sentationen von Bildern},
  author={Matthias Fey},
  school={Technische Universit{\"a}t Dortmund},
  year={2017},
}
```
