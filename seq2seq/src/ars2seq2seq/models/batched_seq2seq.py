"""
Adapted and fixed implementation from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
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
import math
import os
from ars2seq2seq.util.model_util import save_checkpoint, load_checkpoint, checkpoint_exists
from ars2seq2seq.util.tensorboard import TBLogger
from ars2seq2seq.models.eval import evaluate_pairs
from ars2seq2seq.util.vocab import read_paired, length_filter_pair, UNK_token, SOS_token, EOS_token, PAD_token
from ars2seq2seq.util.dataset import form_batch_from_pairs, custom_collate_fn

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import numpy as np

import io
import torchvision
from PIL import Image

from ars2seq2seq.models.masked_cross_entropy import compute_loss

MAX_LENGTH = 1000
BATCH_SIZE = 1

def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq

"""
OK, issue is with the stepping, after putting checks into the batch formed and stepping portions.
It may very well be the 

"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # TODO: Confirm we can send the GRU a hidden=None
        embedded = self.embedding(input_seqs)
        # Best explanation for why packed sequences observed in https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # Basically it's a convenience structure for RNN units that allow us to save computation over
        # multi-length sequences.
        # So you'd pad first, then pack it.
        # Closest analogy is how multiple channels are packed into a contiguous byte array for an image.
        # We want to be able to do convolution operations over these in a natural way, so specialized
        # storage structures allow us to access then via spatial coordinates.
        # In this case, the GRU knows how to operate over the packed sequence, and the inverse is obtained by the pad_packed_sequence() call.
        # Note the output here is the entire encoding sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # See https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence, appear to be a way of storing multiple sequences together
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, device="cpu"):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            # self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))  # Original method had psychotically big values
            self.v = nn.Parameter(torch.empty(1, hidden_size).uniform_(-1, 1))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
#         print('[attn] seq len', seq_len)
#         print('[attn] encoder_outputs', encoder_outputs.size()) # S x B x N
#         print('[attn] hidden', hidden.size()) # S=1 x B x N

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, seq_len)) # B x S
#         print('[attn] attn_energies', attn_energies.size())

        attn_energies = attn_energies.to(self.device)

        # For each batch of encoder outputs
        # NOTE: This is currently done incrementally; there may be a way to do this more efficiently using 3D tensors
        # NOTE: If this is to be done incrementally, might as well perform masking here to invalidate
        # specific attention ranges (?)
        # Should check here, https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq, where they claim
        # to be able to generate a batchwise attention
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
#         print('[attn] attn_energies', attn_energies.size())
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)


    def score(self, hidden, encoder_output):
        """
        Z-scores each of the logits for the softmax computation.  Z-scores must be scalars.
        :param hidden:
        :param encoder_output:
        :return:
        """
        if self.method == 'dot':
            # energy = hidden.dot(encoder_output)
            energy = torch.tensordot(hidden, encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # energy = hidden.dot(energy)
            energy = torch.tensordot(hidden, energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            #energy = self.v.dot(energy)
            energy = torch.tensordot(self.v, energy)
            return energy



class LuongAttnDecoderRNN(nn.Module):
    # TODO: Save out the attention model, or ensure that it saves out correctly (Torch figures out
    # connections well?).
    def __init__(self, attn_model, hidden_size, output_size, num_layers=1, dropout=0.1, device="cpu"):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, device=device)

    def forward(self, input_seq, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
#         print('[decoder] input_seq', input_seq.size()) # batch_size x 1
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
#         print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state
#         print('[decoder] last_hidden', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
#         print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
#         print('[decoder] attn_weights', attn_weights.size())
#         print('[decoder] encoder_outputs', encoder_outputs.size())
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
#         print('[decoder] context', context.size())

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
#         print('[decoder] rnn_output', rnn_output.size())
#         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))  # TODO: Why are we concating with one object here?

        # Finally predict next token (Luong eq. 6)
#         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)  # Emitting own hypothesis as logits

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights




class BatchedSeq2Seq:
    def __init__(self, hidden_size,
                 input_lang, output_lang,
                 batch_size=BATCH_SIZE,
                 max_length=MAX_LENGTH,
                 bidirectional=False,
                 num_layers=1,
                 initial_learning_rate=0.001,
                 output_root="./output",
                 device=None):
        self.input_size = input_lang.n_words
        self.output_size = output_lang.n_words
        self.hidden_size = hidden_size
        self.max_length, self.batch_size = max_length, batch_size
        self.input_lang, self.output_lang = input_lang, output_lang
        self.bidirectional, self.num_layers = bidirectional, num_layers
        self.encoder = EncoderRNN(self.input_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        self.decoder = LuongAttnDecoderRNN('concat', hidden_size, self.output_size, device=device, num_layers=num_layers)
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

    def generate(self, input_X, input_lens, max_length=MAX_LENGTH):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_X, input_lens)
            batchsize = input_X.size(1)
            all_decoder_outputs = Variable(torch.zeros(self.max_length, self.batch_size, self.decoder.output_size,
                                                       device=self.device))
            decoder_input = Variable(torch.LongTensor([[SOS_token] * batchsize])).transpose(0, 1).to(self.device) #
            decoder_context = encoder_outputs[-1].to(self.device)
            all_decoder_outputs = all_decoder_outputs.to(self.device)
            decoder_hidden = encoder_hidden
            decoded_words = [[] for x in range(batchsize)]
            decoded_idxes = [[] for x in range(batchsize)]
            for dc_idx in range(max_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_context, decoder_hidden, encoder_outputs
                )
                all_decoder_outputs[dc_idx] = decoder_output
                top_vals, top_idxes = decoder_output.topk(1, dim=-1)
                decoder_input = top_idxes.detach()  # Detach from history as input
                for b_idx in range(batchsize):
                    top_idx = top_idxes[b_idx, 0].cpu().item()
                    # Get the decoded word
                    if top_idx == EOS_token:
                        decoded_idxes[b_idx].append(EOS_token)
                    elif top_idx == PAD_token:
                        pass
                    else:
                        decoded_words[b_idx].append(self.output_lang.index2word.get(top_idx))
                        decoded_idxes[b_idx].append(top_idx)
        return decoded_words, decoded_idxes

    def evaluate(self, val_dataset):
        # TODO: Get iteration over Val dataset
        #val_input_X, val_input_lens, val_target_X, val_target_lens = val_dataset.form_batch_from_pairs([1])
        #decoded_words, decoded_idxes = self.generate(val_input_X, val_input_lens)
        #strict_matches = 0
        return 1, "" # dud for now

    def _train_step(self, input_X, input_lengths, target_X, target_lengths, criterion):
        max_target_length = max(target_lengths)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        #print("Starting encoding")
        encoder_outputs, encoder_hidden = self.encoder(input_X, input_lengths)
        #print("... encoded")
        batchsize = encoder_outputs.size(1)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batchsize, self.decoder.output_size, device=self.device))

        # Prepare input and output variables
        #print("Starting decode")
        decoder_input = Variable(torch.LongTensor([[SOS_token] * batchsize])).transpose(0, 1).to(self.device)
        decoder_context = encoder_outputs[-1].to(self.device)
        all_decoder_outputs = all_decoder_outputs.to(self.device)
        decoder_hidden = encoder_hidden
        #print("... decode variables set")

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for dc_idx in range(max_target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_context, decoder_hidden, encoder_outputs
                )
                all_decoder_outputs[dc_idx] = decoder_output
                decoder_input = target_X[dc_idx]
        else:
            # TODO: Get decoder inputs to line up properly
            for dc_idx in range(max_target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_context, decoder_hidden, encoder_outputs
                )
                all_decoder_outputs[dc_idx] = decoder_output
                top_vals, top_idxes = decoder_output.topk(1, dim=-1)
                decoder_input = top_idxes.detach()  # Detach from history as input

        #print("Decode finished")
        loss = self.batched_loss(all_decoder_outputs, target_X, target_lengths, criterion)
        #print("Batched loss finished")
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        #print("Backward finished")
        return loss.item()

    def batched_loss(self, guess_Y, gold_Y, target_lengths, criterion):
        """
        Formats the decoded guesses and the gold, masked against the target lengths, using
        the batched criterion function.
        :param X:
        :param Y:
        :param target_lengths:
        :param criterion:
        :return:
        """
        return criterion(guess_Y.transpose(0, 1).contiguous(),
                  gold_Y.transpose(0, 1).contiguous(),
                  target_lengths)

    def train(self,
              n_iters,
              train_dataset,
              val_dataset,
              save_every=1000,
              sample_every=100,
              eval_every=1000,
              ngpu=2,
              profile=False,
              load_checkpoint_if_exist=True):
        print("Starting training")
        os.makedirs("output", exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        tb_logger = TBLogger(self.log_dir)

        print("Beginning training, total # items={}".format(len(train_dataset)))
        epoch_size = len(train_dataset)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.initial_learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.initial_learning_rate)

        if load_checkpoint_if_exist:
            self.load_checkpoint()

        #if (device.type == 'cuda') and (ngpu > 1):
        #    encoder = nn.DataParallel(encoder, list(range(ngpu)))
        #    decoder = nn.DataParallel(decoder, list(range(ngpu)))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn)
        criterion = compute_loss
        accum_losses = []  # For computing mean loss
        with open(os.path.join(self.log_dir, "guessed_trials.txt"), "w") as val_log_f:
            num_epochs = 10
            step = 0
            for epoch in range(num_epochs):
                print("Starting epoch={}".format(epoch))
                progbar = tqdm(train_dataloader)
                #for idx in range(10):
                for paired_idxes in progbar:
                    # to grab data ahead of time.  We may be getting most amoritzed in this next step.
                    input_X, input_lengths, target_X, target_lengths = form_batch_from_pairs(paired_idxes, self.device)
                    #print("Batch formed, stepping")
                    loss = self._train_step(input_X, input_lengths, target_X, target_lengths, criterion)
                    #print("step")
                    accum_losses.append(loss)
                    progbar.set_description("Loss={:.5f}".format(loss))

                    if step % sample_every == 0:
                        mean_loss = np.mean(accum_losses)
                        tb_logger.scalar_summary("Mean Loss", mean_loss, step)
                        accum_losses = []

                    if step % save_every == 0 and step > 0:
                        epoch = (step // epoch_size) + 1
                        self.save_checkpoint(epoch)

                    if step % eval_every == 0 and step > 0:
                        print("---------------------\nValidate check")
                        acc, res_str = self.evaluate(val_dataset)
                        tb_logger.scalar_summary("Val Acc", acc, step)
                        val_log_f.write("\n==========================================\nStep={}\n".format(step))
                        val_log_f.write(res_str)
                        val_log_f.write("\n")
                    step += 1
                    if profile and step == 5:
                        return # Force exit

            epoch = (step // epoch_size) + 1
            save_checkpoint(epoch)
            acc, res_str = self.evaluate(val_dataset)
            tb_logger.scalar_summary("Val Acc", acc, step)
            val_log_f.write("\n==========================================\nStep={}\n".format(step))
            val_log_f.write(res_str)
            val_log_f.write("\n")

    def evaluate_and_show_attention(self, input_sentence):
        output_words, attentions = self.generate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        #show_attention(input_sentence, output_words, attentions)



