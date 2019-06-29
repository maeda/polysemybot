import argparse

from model import Model
from dataset import process, load
from pre_processing import PreProcessing


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    model = Model()

    if args.train:
        file_train = args.train.split('/')[-1].split('.')[0]
        try:
            dataset = load(file_train)
        except Exception as e:
            dataset = process(PreProcessing(open(args.train, 'r'), file_train))
        model.train(dataset)

    if args.test:
        dataset = load(args.corpus.split('/')[-1].split('.')[0])
        while True:
            decoded_words = model.evaluate(dataset.vocabulary, str(input("> ")))
            print(' '.join(decoded_words))

