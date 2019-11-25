#
# Clarified model for seq2seq+attn
#

# -*- coding: utf-8 -*-
"""

Adapted from tutorial code at https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
from __future__ import unicode_literals, print_function, division
from io import open
import string
import re
import os
import time
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import os
from ars2seq2seq.util.model_util import save_checkpoint, load_checkpoint, checkpoint_exists
from ars2seq2seq.util.tensorboard import TBLogger
from ars2seq2seq.models.eval import evaluate_pairs
from ars2seq2seq.util.vocab import read_paired, length_filter_pair, UNK_token, SOS_token, EOS_token, normalize_string
from ars2seq2seq.util.vocab import process_sentence

MAX_LENGTH=100


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, bidirectional=False,
                 custom_embedding=False, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.bidirectional, self.num_layers = bidirectional, num_layers
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.hidden_size = hidden_size
        self.device = device
        # Alllow ability for backwards embedder to use the decoder
        if custom_embedding:
            self.embedding = custom_embedding
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size * self.num_directions, device=self.device)
        #return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device,
                 dropout_p=0.1,
                 bidirectional=False, num_layers=1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        self.bidirectional, self.num_layers = bidirectional, num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)



class Seq2Seq:
    def __init__(self, hidden_size,
                 input_lang, output_lang,
                 bidirectional=False,
                 num_layers=1,
                 initial_learning_rate=0.001,
                 output_root="./output",
                 device=None):
        self.input_size = input_lang.n_words
        self.output_size = output_lang.n_words
        self.hidden_size = hidden_size
        self.input_lang, self.output_lang = input_lang, output_lang
        self.bidirectional, self.num_layers = bidirectional, num_layers
        self.encoder = EncoderRNN(self.input_size, hidden_size, device, bidirectional=bidirectional, num_layers=num_layers)
        self.decoder = AttnDecoderRNN(hidden_size, self.output_size, device, bidirectional=bidirectional, num_layers=num_layers)
        self.device = device
        if device:
            self.encoder.to(device)
            self.decoder.to(device)
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.initial_learning_rate = initial_learning_rate
        self.teacher_forcing_ratio = 0.5
        self.output_root = os.path.join(output_root, self.exp_name())
        self.checkpoint_dir = os.path.join(self.output_root, "checkpoints")
        self.log_dir = os.path.join(self.output_root, "logs")

    def exp_name(self):
        return "h={}_nl={}_lr={}".format(self.hidden_size,  self.num_layers, self.initial_learning_rate)

    def load_checkpoint(self):
        print("Loading checkpoint from {}".format(self.checkpoint_dir))
        if checkpoint_exists("encoder", self.checkpoint_dir) and checkpoint_exists("decoder", self.checkpoint_dir):
            print("Checkpoints exist at {}, loading".format(self.checkpoint_dir))
            load_checkpoint(self.encoder, "encoder", self.encoder_optimizer, self.checkpoint_dir)
            load_checkpoint(self.decoder, "decoder", self.decoder_optimizer, self.checkpoint_dir)
        else:
            print("Checkpoints do not exist, no action taken")

    def save_checkpoint(self, epoch):
        save_checkpoint("encoder", self.encoder, self.encoder_optimizer, self.checkpoint_dir, epoch)
        save_checkpoint("decoder", self.decoder, self.decoder_optimizer, self.checkpoint_dir, epoch)

    def generate(self, sentence, max_length=MAX_LENGTH, include_attentions=False):
        sentence = normalize_string(sentence)
        with torch.no_grad():
            input_tensor = tensor_from_sentence(self.input_lang, sentence, self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            if include_attentions:
                decoder_attentions = torch.zeros(max_length, max_length)
            else:
                decoder_attentions = None

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                if include_attentions:
                    decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            if include_attentions:
                return decoded_words, decoder_attentions[:di + 1]
            else:
                return decoded_words, None

    def evaluate(self, eval_pairs, n=30, randomize=True):
        if randomize:
            to_eval = [random.choice(eval_pairs) for i in range(n)]
        else:
            to_eval = eval_pairs[0:n]

        def generate_fn(sentence):
            return self.generate(sentence)

        return evaluate_pairs(generate_fn, to_eval)

    def _train_step(self, input_tensor, target_tensor, criterion, max_length=MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / target_length

    def train(self,
              n_iters,
              train_pairs,
              val_pairs,
              save_every=1000,
              sample_every=100,
              eval_every=1000,
              ngpu=2,
              load_checkpoint_if_exist=True):
        print("Starting training")
        os.makedirs("output", exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        tb_logger = TBLogger(self.log_dir)

        print("Beginning training, total # items={}".format(len(train_pairs)))
        epoch_size = len(train_pairs)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.initial_learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.initial_learning_rate)

        if load_checkpoint_if_exist:
            self.load_checkpoint()

        #if (device.type == 'cuda') and (ngpu > 1):
        #    encoder = nn.DataParallel(encoder, list(range(ngpu)))
        #    decoder = nn.DataParallel(decoder, list(range(ngpu)))

        training_pairs = [tensors_from_pair(random.choice(train_pairs), self.input_lang, self.output_lang, self.device)
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()
        progbar = tqdm(range(1, n_iters + 1))
        accum_losses = []  # For computing mean loss
        with open(os.path.join(self.log_dir, "guessed_trials.txt"), "w") as val_log_f:
            for step in progbar:
                training_pair = training_pairs[step - 1]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]

                loss = self._train_step(input_tensor, target_tensor, criterion)
                accum_losses.append(loss)
                progbar.set_description("Loss={:.5f}".format(loss))

                if step % sample_every == 0:
                    mean_loss = np.mean(accum_losses)
                    tb_logger.scalar_summary("Mean Loss", mean_loss, step)
                    accum_losses = []

                if step % save_every == 0:
                    epoch = (step // epoch_size) + 1
                    self.save_checkpoint(epoch)

                if step % eval_every == 0:
                    print("---------------------\nValidate check")
                    acc, res_str = self.evaluate(val_pairs)
                    tb_logger.scalar_summary("Val Acc", acc, step)
                    val_log_f.write("\n==========================================\nStep={}\n".format(step))
                    val_log_f.write(res_str)
                    val_log_f.write("\n")

            epoch = (step // epoch_size) + 1
            save_checkpoint(epoch)
            acc, res_str = self.evaluate(val_pairs)
            tb_logger.scalar_summary("Val Acc", acc, step)
            val_log_f.write("\n==========================================\nStep={}\n".format(step))
            val_log_f.write(res_str)
            val_log_f.write("\n")

    def evaluate_and_show_attention(self, input_sentence):
        output_words, attentions = self.generate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        show_attention(input_sentence, output_words, attentions)


def indexes_from_sentence(lang, sentence):
    return [lang.word2index.get(word, UNK_token) for word in process_sentence(sentence)]


def tensor_from_sentence(lang, sentence, device):
    """
    Converts the target sentence into a tensor.  Automatically appends the special <EOS> to the end.
    :param lang:
    :param sentence:
    :param device:
    :return:
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair, input_lang, output_lang, device):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)



def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + process_sentence(input_sentence) +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()








