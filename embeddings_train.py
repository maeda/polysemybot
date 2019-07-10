from deprecated import deprecated

from dataset import process
from embeddings import WordEmbedding
from pre_processing import PreProcessing


@deprecated(reason="not tested")
def retrain():
    ds = process(PreProcessing('./data/starwars.txt'))

    word_embedding = WordEmbedding(source='./embedding/FT/fasttext_cbow_300d.bin')

    word_embedding.train(ds.pairs)
    word_embedding.save('./embedding/starwars', 'starwars.bin')


@deprecated(reason="not tested")
def train():
    ds = process(PreProcessing(open('./data/starwars.txt', 'r')))

    word_embedding = WordEmbedding(source=ds.pairs)

    word_embedding.train(ds.pairs)

    word_embedding.save(target_folder='./embedding/starwars', filename='starwars.bin')


if __name__ == '__main__':
    train()
