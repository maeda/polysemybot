from __future__ import unicode_literals, print_function, division

import os
import random
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dataset import Dataset
import settings
from embeddings import WordEmbedding
from pre_processing import SOS, EOS, control_words
from utils import time_since


class EncoderRNN(nn.Module):
    def __init__(self, word_embedding: WordEmbedding, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.word_embedding = word_embedding
        self.embedding = word_embedding.embedding_layer
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=settings.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, word_embedding: WordEmbedding, dropout_p, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.word_embedding = word_embedding
        self.embedding = word_embedding.embedding_layer
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, word_embedding.n_words())
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=settings.device)


class Model:

    def __init__(self,
                 encoder: EncoderRNN,
                 decoder: DecoderRNN,
                 teacher_forcing_ratio: float = 1,
                 max_lenght: int = 20,
                 learning_rate: float = 0.01
                 ):

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_length = max_lenght
        self.learning_rate = learning_rate

        self.plot_losses = []
        self.encoder: EncoderRNN = encoder
        self.decoder: DecoderRNN = decoder
        self.encoder_optimizer, self.decoder_optimizer = self._optimizers(learning_rate)

    def _optimizers(self, learning_rate):
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        return encoder_optimizer, decoder_optimizer

    def _optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

    def summary(self):
        print(self.encoder)
        print(self.decoder)

    @classmethod
    def load(cls, dataset_id):
        with open(os.path.join(settings.BASE_DIR, dataset_id, settings.SAVE_DATA_DIR, '.metadata'), 'r') as f:
            model_name = f.read()

        directory = os.path.join(settings.BASE_DIR, dataset_id, settings.SAVE_DATA_DIR, model_name)
        model = torch.load(directory)
        return model['model']

    def train(self, dataset: Dataset, n_iter=50, print_every=10, save_every=10, plot_every=10):

        start = time.time()
        self.plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        training_pairs = dataset.training_pairs(n_iter, self.encoder.word_embedding)
        criterion = nn.NLLLoss()

        sos_id = self.encoder.word_embedding.word2index(SOS)
        eos_id = self.decoder.word_embedding.word2index(EOS)

        for iteration in tqdm(range(1, n_iter + 1)):
            training_pair = training_pairs[iteration - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self._train(sos_id, eos_id, input_tensor, target_tensor, self.encoder, self.decoder, self.encoder_optimizer,
                               self.decoder_optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if iteration % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (time_since(start, iteration / n_iter), iteration, iteration / n_iter * 100, print_loss_avg))

            if iteration % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                self.plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if iteration % save_every == 0:
                model_dir = '{}-{}_{}'.format(str(self.encoder.n_layers), str(self.decoder.n_layers), str(iteration))
                model_name = '{}.torch'.format('backup_bidir_model')

                directory = os.path.join(settings.BASE_DIR, dataset.idx, settings.SAVE_DATA_DIR,
                                         model_dir)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                torch.save({
                    'iteration': iteration,
                    'enc': self.encoder.state_dict(),
                    'dec': self.decoder.state_dict(),
                    'enc_opt': self.encoder_optimizer.state_dict(),
                    'dec_opt': self.decoder_optimizer.state_dict(),
                    'model': self,
                    'loss': loss
                }, os.path.join(directory, model_name))

                with open(os.path.join(settings.BASE_DIR, dataset.idx, settings.SAVE_DATA_DIR, '.metadata'), 'w') as f:
                    f.write(os.path.join(model_dir, model_name))

    def _train(self, sos_id, eos_id, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
               criterion: nn.NLLLoss):
        encoder_hidden = encoder.init_hidden()

        self._optimizers_zero_grad()

        input_length = input_tensor.to(settings.device).size(0)
        target_length = target_tensor.to(settings.device).size(0)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[sos_id]], device=settings.device).to(settings.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == eos_id:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def evaluate_randomly(self, dataset: Dataset, n=10):
        for i in range(n):
            pair = random.choice(dataset.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words = self.evaluate(pair[0], dataset)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluate(self, sentence, dataset: Dataset):
        with torch.no_grad():
            input_tensor = dataset.tensor_from_sentence(self.encoder.word_embedding, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=settings.device)

            sos_id = self.encoder.word_embedding.word2index(SOS)
            eos_id = self.encoder.word_embedding.word2index(EOS)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[sos_id]], device=settings.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []

            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)

                if topi.item() == eos_id:
                    break

                decoded_word = self.decoder.word_embedding.index2word(topi.item())

                if len(decoded_words) > 1 and decoded_word == decoded_words[-1]:
                    break

                decoded_words.append(decoded_word)

                decoder_input = topi.squeeze().detach()

            return [item for item in decoded_words if item not in control_words()]
