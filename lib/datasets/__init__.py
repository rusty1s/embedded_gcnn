from .mnist import MNIST
from .cifar_10 import Cifar10
from .tiny_image_net import TinyImageNet
from .pascal_voc import PascalVOC
from .queue import PreprocessQueue

__all__ = ['MNIST', 'Cifar10', 'TinyImageNet', 'PascalVOC', 'PreprocessQueue']
