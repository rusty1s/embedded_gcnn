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

![Neural Network Approach](https://user-images.githubusercontent.com/6945922/28238129-45422ebe-694d-11e7-9fc5-aee9e651a334.png)
![SlIC and Quickshift Segmentation](https://user-images.githubusercontent.com/6945922/27761633-61569a56-5e60-11e7-96d6-5a0507d26cf8.jpg)

This is a TensorFlow implementation of my master thesis on [Graph-based Image
Classification](https://github.com/rusty1s/deep-learning/blob/master/master/main.pdf)
*(german)*.

**Embedded graph convolutional neural networks (EGCNN)** aim to make significant improvements to learning on graphs where nodes are positioned on a twodimensional euclidean plane and thus possess an orientation (like up, down, right and left).
As proof, we implemented an image classification on embedded graphs by first segmenting the image into superpixels with the use of [SLIC](https://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf) or [Quickshift](http://vision.cs.ucla.edu/papers/vedaldiS08quick.pdf), converting this representation into a graph and inputting these to the neural network.
Graphs are trained on three different datasets and are automatically downloaded by running the corresponding train scripts:

* [MNIST](http://yann.lecun.com/exdb/mnist/) (run `python mnist_embedded.py`)
* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) (run `python cifar_embedded.py`)
* [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/) (run `python pascal_embedded.py` and `python pascal_squeeze.py` for validation on 2d images)

This repository also includes layer implementations of alternative approaches such as [SGCNN](https://arxiv.org/abs/1312.6203) and [GCN](https://arxiv.org/abs/1609.02907) for graphs and the Fire module of [SqueezeNet](https://arxiv.org/abs/1602.07360) for images to validate the results.

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
@mastersthesis{fey2017egcnn,
  title={Convolutional Neural Networks auf Graphrepr{\"a}sentationen von Bildern},
  author={Matthias Fey},
  school={Technische Universit{\"a}t Dortmund},
  year={2017},
}
```
