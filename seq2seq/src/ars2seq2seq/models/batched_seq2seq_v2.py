"""
Adapted and fixed implementation from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

Incorporates batched attention from https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
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

def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq

"""
OK, issue is with the stepping, after putting checks into the batch formed and stepping portions.
It may very well be the 

"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size,embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=False)

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class DynamicEncoder(nn.Module):
    """
    Claim is this one does not need to have packed sequences
    """
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx))
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1))  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]



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


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1, device="cpu"):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.device=device
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden



class BatchedSeq2Seq_v2:
    def __init__(self, hidden_size,
                 input_lang, output_lang,
                 batch_size=10,
                 max_length=1000,
                 bidirectional=False,
                 num_layers=1,
                 initial_learning_rate=1e-4,
                 output_root="./output",
                 device=None):
        self.input_size = input_lang.n_words
        self.output_size = output_lang.n_words
        self.hidden_size = hidden_size
        self.max_length, self.batch_size = max_length, batch_size
        self.input_lang, self.output_lang = input_lang, output_lang
        self.bidirectional, self.num_layers = bidirectional, num_layers
        self.encoder = EncoderRNN(self.input_size, hidden_size, hidden_size, n_layers=num_layers)
        #self.encoder = EncoderRNN(self.input_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        self.decoder = BahdanauAttnDecoderRNN(hidden_size, hidden_size, self.output_size, device=device, n_layers=num_layers)
        #self.decoder = LuongAttnDecoderRNN('concat', hidden_size, self.output_size, device=device, num_layers=num_layers)
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

    def generate(self, input_X, input_lens):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_X, input_lens)
            batchsize = input_X.size(1)
            all_decoder_outputs = Variable(torch.zeros(self.max_length, self.batch_size, self.decoder.output_size,
                                                       device=self.device))
            decoder_input = Variable(torch.LongTensor([[SOS_token] * batchsize])).transpose(0, 1).to(self.device)
            decoder_context = encoder_outputs[-1].to(self.device)
            all_decoder_outputs = all_decoder_outputs.to(self.device)
            decoder_hidden = encoder_hidden
            decoded_words = [[] for x in range(batchsize)]
            decoded_idxes = [[] for x in range(batchsize)]
            for dc_idx in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
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

    def evaluate(self, val_dataloader):
        val_iter = tqdm(val_dataloader)
        decoded_sentences = []
        num = 1
        for val_pair_idxes in val_iter:
            val_input_X, val_input_lens, val_target_X, val_target_lens = form_batch_from_pairs(val_pair_idxes, self.device)
            decoded_word_batches, decoded_idxes = self.generate(val_input_X, val_input_lens)
            for decoded_words in decoded_word_batches:
                decoded_sentences.append("#{}:\t{}".format(num, " ".join(decoded_words)))
                num += 1
        res_str = "\n".join(decoded_sentences)
        return 1, res_str # dud for now

    def _train_step(self, input_X, input_lengths, target_X, target_lengths, criterion):
        max_target_length = max(target_lengths)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.encoder(input_X, input_lengths)
        batchsize = encoder_outputs.size(1)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batchsize, self.decoder.output_size, device=self.device))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[SOS_token] * batchsize])).transpose(0, 1).to(self.device)
        decoder_context = encoder_outputs[-1].to(self.device)
        all_decoder_outputs = all_decoder_outputs.to(self.device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for dc_idx in range(max_target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                all_decoder_outputs[dc_idx] = decoder_output
                decoder_input = target_X[dc_idx]
        else:
            # TODO: Get decoder inputs to line up properly
            for dc_idx in range(max_target_length):
                decoder_output, decoder_hidden  = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                all_decoder_outputs[dc_idx] = decoder_output
                top_vals, top_idxes = decoder_output.topk(1, dim=-1)
                decoder_input = top_idxes.detach()  # Detach from history as input

        loss = self.batched_loss(all_decoder_outputs, target_X, target_lengths, criterion)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
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

    def _validate(self, val_log_f, val_dataloader, epoch, step):
        print("---------------------\nValidate check\nEpoch={} Step={}".format(epoch, step))
        acc, res_str = self.evaluate(val_dataloader)
        val_log_f.write("\n==========================================\nEpoch={}\tStep={}\n".format(epoch, step))
        val_log_f.write(res_str)
        val_log_f.write("\n")
        val_log_f.flush()
        return acc

    def train(self,
              n_iters,
              train_dataset,
              val_dataset,
              save_every=1000,
              sample_every=100,
              eval_every=200,
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
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn, shuffle=False)
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
                    loss = self._train_step(input_X, input_lengths, target_X, target_lengths, criterion)
                    accum_losses.append(loss)
                    progbar.set_description("Loss={:.5f}".format(loss))

                    if step % sample_every == 0:
                        mean_loss = np.mean(accum_losses)
                        tb_logger.scalar_summary("Mean Loss", mean_loss, step)
                        accum_losses = []

                    if step % eval_every == 0 and step > 0:
                        acc = self._validate(val_log_f, val_dataloader, epoch, step)
                        tb_logger.scalar_summary("Val Acc", acc, step)

                    if step % save_every == 0 and step > 0:
                        epoch = (step // epoch_size) + 1
                        self.save_checkpoint(epoch)

                    step += 1
                    if profile and step == 5:
                        return # Force exit
                acc = self._validate(val_log_f, val_dataloader, epoch, step)
                tb_logger.scalar_summary("Val Acc", acc, step)
                self.save_checkpoint(epoch)

    def evaluate_and_show_attention(self, input_sentence):
        output_words, attentions = self.generate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        #show_attention(input_sentence, output_words, attentions)



