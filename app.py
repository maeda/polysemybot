import sys

from model import Model
from preprocessing import PreProcessing, DatasetReader


if __name__ == '__main__':
    model = Model()

    model.train(PreProcessing(DatasetReader(open(sys.argv[1], 'r'))))
    model.evaluate_randomly(10)
