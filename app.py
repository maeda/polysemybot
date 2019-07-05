import argparse

import settings
from embeddings import WordEmbeddingBasic
from model import Model, EncoderRNN, DecoderRNN
from dataset import process, load
from pre_processing import PreProcessing


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-it', '--iteration', type=int, default=50000, help='Train the model with it iterations')
    parser.add_argument('-hi', '--hidden', type=int, default=300, help='Hidden size in encoder and decoder')
    parser.add_argument('-s', '--save', type=int, default=1000, help='Save every s iterations')
    parser.add_argument('-d', '--dropout', type=float, default=0.0, help='Dropout probability for rnn and dropout layers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    # parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')

    args = parser.parse_args()
    return args


def run(hidden, layer, dropout, learning_rate, iteration, save, train=None, test=None):
    if train:
        dataset_id = train.split('/')[-1].split('.')[0]

        pre_processing = PreProcessing(open(train, 'r'), dataset_id)
        dataset = process(pre_processing)

        encoder_embeddings = WordEmbeddingBasic(pairs=dataset.pairs)
        decoder_embeddings = WordEmbeddingBasic(pairs=dataset.pairs)

        encoder = EncoderRNN(encoder_embeddings, hidden, layer).to(settings.device)
        decoder = DecoderRNN(hidden, decoder_embeddings, dropout, layer).to(settings.device)

        model = Model(
            encoder=encoder,
            decoder=decoder,
            learning_rate=learning_rate,
        )
        model.summary()
        model.train(dataset, n_iter=iteration, save_every=save)

    if test:

        dataset = load(test)

        model = Model.load(test)

        while True:
            decoded_words = model.evaluate(str(input("> ")), dataset)
            print(' '.join(decoded_words))


if __name__ == '__main__':
    args = parse()
    run(args.hidden, args.layer, args.dropout, args.learning_rate, args.iteration, args.save, args.train, args.test)


