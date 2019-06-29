from __future__ import unicode_literals, print_function, division

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dataset import EOS_token, SOS_token, Dataset, control_words
import settings
from utils import time_since, TensorHelper


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=settings.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size, n_layers)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
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
                 input_size: int,
                 output_size: int,
                 hidden_size: int = 300,
                 teacher_forcing_ratio: float = 0.5,
                 max_lenght: int = 20,
                 dropout_p: float = 0.01,
                 learning_rate: float = 0.01,
                 tensor_helper: TensorHelper = TensorHelper(settings.device, EOS_token)):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.max_length = max_lenght
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate

        self.tensor_helper = tensor_helper
        self.plot_losses = []
        self.encoder, self.decoder = self.build(input_size, output_size, dropout_p)
        self.encoder_optimizer, self.decoder_optimizer = self._optimizers(learning_rate)

    def build(self, input_size: int, output_size: int, dropout_p: float = 0.1):
        encoder = EncoderRNN(input_size, self.hidden_size).to(settings.device)
        decoder = DecoderRNN(self.hidden_size, output_size, dropout_p=dropout_p).to(settings.device)

        return encoder, decoder

    def _optimizers(self, learning_rate):
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        return encoder_optimizer, decoder_optimizer

    def _optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

    def train(self, dataset: Dataset, n_iter=50, print_every=10, save_every=10, plot_every=10):

        start = time.time()
        self.plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        training_pairs = dataset.training_pairs(n_iter)
        criterion = nn.NLLLoss()

        for iteration in range(1, n_iter + 1):
            training_pair = training_pairs[iteration - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self._train(input_tensor, target_tensor, self.encoder, self.decoder, self.encoder_optimizer,
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
                directory = os.path.join(settings.SAVE_DATA_DIR, dataset.idx,
                                         '{}-{}_{}'.format(
                                             str(self.encoder.n_layers),
                                             str(self.decoder.n_layers),
                                             str(iteration)))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                torch.save({
                    'iteration': iteration,
                    'enc': self.encoder.state_dict(),
                    'dec': self.decoder.state_dict(),
                    'enc_opt': self.encoder_optimizer.state_dict(),
                    'dec_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss
                }, os.path.join(directory, '{}_{}.torch'.format(iteration, 'backup_bidir_model')))

    def _train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
               criterion: nn.NLLLoss):
        encoder_hidden = encoder.init_hidden()

        self._optimizers_zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=settings.device)

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
                if decoder_input.item() == EOS_token:
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
            output_words = self.evaluate(dataset.vocabulary, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluate(self, vocabulary, sentence):
        with torch.no_grad():
            input_tensor = self.tensor_helper.tensor_from_sentence(vocabulary, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=settings.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=settings.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []

            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)

                if topi.item() == EOS_token:
                    break

                decoded_words.append(vocabulary.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return [item for item in decoded_words if item not in control_words()]
