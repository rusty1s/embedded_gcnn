from lib.datasets.pascal_voc import PascalVOC

bla = PascalVOC(
    train_dir='data/pascal_voc/input',
    test_dir='data/pascal_voc/input/VOCtestkit')

print(len(bla.train._data))
print(bla.train._labels.shape)
print(len(bla.validation._data))
print(bla.validation._labels.shape)
print(len(bla.test._data))
print(bla.test._labels.shape)
