import sys

from model import Model
from dataset import process
from pre_processing import PreProcessing

if __name__ == '__main__':
    model = Model()

    dataset = process(PreProcessing(open(sys.argv[1], 'r')))

    model.train(dataset)
    model.evaluate_randomly(dataset, 10)
