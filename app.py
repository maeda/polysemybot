import argparse

from model import Model
from dataset import process, load, DatasetStorage
from pre_processing import PreProcessing


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-it', '--iteration', type=int, default=1000, help='Train the model with it iterations')
    parser.add_argument('-s', '--save', type=int, default=100, help='Save every s iterations')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    if args.train:
        file_train = args.train.split('/')[-1].split('.')[0]

        dataset = process(PreProcessing(open(args.train, 'r'), file_train))

        model = Model(dataset.vocab_size(), dataset.vocab_size())
        model.train(dataset, n_iter=args.iteration, save_every=args.save)

    if args.test:
        dataset = load(args.corpus.split('/')[-1].split('.')[0])
        model = Model(dataset.vocab_size(), dataset.vocab_size())
        while True:
            decoded_words = model.evaluate(dataset.vocabulary, str(input("> ")))
            print(' '.join(decoded_words))

