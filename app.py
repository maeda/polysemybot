import argparse

from model import Model
from dataset import process, load, WordEmbedding
from pre_processing import PreProcessing


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-it', '--iteration', type=int, default=50000, help='Train the model with it iterations')
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    parser.add_argument('-s', '--save', type=int, default=1000, help='Save every s iterations')
    parser.add_argument('-d', '--dropout', type=float, default=0.0, help='Dropout probability for rnn and dropout layers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    # parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    if args.train:
        dataset_id = args.train.split('/')[-1].split('.')[0]

        pre_processing = PreProcessing(open(args.train, 'r'), dataset_id)
        dataset = process(pre_processing)

        word_embeddings = WordEmbedding.load_from_file('./embedding/starwars/fasttext_cbow_300d.bin')

        model = Model(
            word_embeddings,
            hidden_size=args.hidden,
            dropout_p=args.dropout,
            learning_rate=args.learning_rate,
            n_layers=args.layer
        )
        model.summary()
        model.train(dataset, n_iter=args.iteration, save_every=args.save)

    if args.test:

        dataset = load(args.test)

        model = Model.load(args.test)

        while True:
            decoded_words = model.evaluate(str(input("> ")), dataset)
            print(' '.join(decoded_words))

