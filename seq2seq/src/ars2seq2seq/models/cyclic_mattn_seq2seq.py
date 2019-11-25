import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ars2seq2seq.models.seq2seq import *

from ars2seq2seq.models.multi_attn_seq2seq import MultiAttnDecoderRNN

class CyclicMultiHeadSeq2Seq():
    def __init__(self, hidden_size,
                 input_lang, output_lang,
                 bidirectional=False,
                 num_layers=1,
                 num_attn = 2,
                 initial_learning_rate=0.001,
                 output_root="./output/cycle",
                 device=None):
        self.input_size = input_lang.n_words
        self.output_size = output_lang.n_words
        self.hidden_size = hidden_size
        self.num_attn = num_attn
        self.input_lang, self.output_lang = input_lang, output_lang
        self.bidirectional, self.num_layers = bidirectional, num_layers
        self.fwd_encoder = EncoderRNN(self.input_size, hidden_size, device, bidirectional=bidirectional, num_layers=num_layers)
        self.fwd_decoder = MultiAttnDecoderRNN(hidden_size, self.output_size, device,
                                               bidirectional=bidirectional, num_layers=num_layers,
                                               num_attn=num_attn)
        self.bck_encoder = EncoderRNN(self.input_size, hidden_size, device, bidirectional=bidirectional, num_layers=num_layers)
        self.bck_decoder = MultiAttnDecoderRNN(hidden_size, self.output_size, device,
                                               bidirectional=bidirectional, num_layers=num_layers,
                                               num_attn=num_attn)
        self.device = device
        if device:
            self.fwd_encoder.to(device)
            self.fwd_decoder.to(device)
            self.bck_encoder.to(device)
            self.bck_decoder.to(device)
        self.fwd_encoder_optimizer = None
        self.fwd_decoder_optimizer = None
        self.bck_encoder_optimizer = None
        self.bck_decoder_optimizer = None
        self.initial_learning_rate = initial_learning_rate
        self.teacher_forcing_ratio = 0.5
        self.output_root = os.path.join(output_root, self.exp_name())
        self.checkpoint_dir = os.path.join(self.output_root, "checkpoints")
        self.log_dir = os.path.join(self.output_root, "logs")

    def exp_name(self):
        return "h={}_nl={}_nattn={}_lr={}".format(self.hidden_size, self.num_layers, self.num_attn, self.initial_learning_rate)

    def load_checkpoint(self):
        if checkpoint_exists("encoder", self.checkpoint_dir) and checkpoint_exists("decoder", self.checkpoint_dir):
            print("Checkpoints exist at {}, loading".format(self.checkpoint_dir))
            load_checkpoint(self.fwd_encoder, "fwd_encoder", self.fwd_encoder_optimizer, self.checkpoint_dir)
            load_checkpoint(self.fwd_decoder, "fwd_decoder", self.fwd_decoder_optimizer, self.checkpoint_dir)
            load_checkpoint(self.bck_encoder, "bck_encoder", self.fwd_encoder_optimizer, self.checkpoint_dir)
            load_checkpoint(self.bck_decoder, "bck_decoder", self.fwd_decoder_optimizer, self.checkpoint_dir)
        else:
            print("Checkpoints do not exist, no action taken")

    def save_checkpoint(self, epoch):
        save_checkpoint("fwd_encoder", self.fwd_encoder, self.fwd_encoder_optimizer, self.checkpoint_dir, epoch)
        save_checkpoint("fwd_decoder", self.fwd_decoder, self.bck_decoder_optimizer, self.checkpoint_dir, epoch)
        save_checkpoint("bck_encoder", self.bck_encoder, self.fwd_encoder_optimizer, self.checkpoint_dir, epoch)
        save_checkpoint("bck_decoder", self.bck_decoder, self.fwd_decoder_optimizer, self.checkpoint_dir, epoch)

    def generate(self, sentence, max_length=MAX_LENGTH, include_attentions=False, custom_max_gen_len=None):
        sentence = normalize_string(sentence)
        with torch.no_grad():
            input_tensor = tensor_from_sentence(self.input_lang, sentence, self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.fwd_encoder.initHidden()

            encoder_outputs = torch.zeros(max_length, self.fwd_encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.fwd_encoder(input_tensor[ei],
                                                                  encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            if include_attentions:
                decoder_attentions = torch.zeros(max_length, max_length)
            else:
                decoder_attentions = None

            # Adjust so we remove all but the last token.  This is to give us the opportunity to slip on a
            # <EOS> at the end of need be.
            max_gen_length = max_length - 1 if custom_max_gen_len is None else custom_max_gen_len
            for di in range(max_gen_length):
                decoder_output, decoder_hidden, decoder_attention = self.fwd_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                if include_attentions:
                    decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    if not(topi.item() in self.output_lang.index2word):
                        decoded_words.append('UNK')
                    else:
                        decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            # Safety valve: If the
            if di == (max_length - 1) and decoded_words[-1] != '<EOS>':
                decoded_words.append('<EOS>')

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

    def _fwd(self, input_tensor, target_tensor, encoder, decoder, loss, criterion, max_length=MAX_LENGTH):
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        # Consider granular teacher forcing, drawing at each point
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        num_generated = 0
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
                num_generated += 1
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, target_tensor[di])
                num_generated += 1
                if decoder_input.item() == EOS_token:
                    break
        return loss

    def _train_step(self, input_sentence, input_tensor, target_tensor, criterion, max_length=MAX_LENGTH):
        target_length = target_tensor.size(0)
        # Optimization hack: try setting max length to be the length of the target tensor + 2
        custopm_gen_max_length = target_length + 2
        fwd_decoded_sentence = " ".join(self.generate(input_sentence, max_length=max_length,
                                                      custom_max_gen_len=custopm_gen_max_length)[0])  # Nothing says we can't generate it twice (inefficient, but easy)
        fwd_decoded_tensor = tensor_from_sentence(self.output_lang, fwd_decoded_sentence, self.device).detach()

        self.fwd_encoder_optimizer.zero_grad()
        self.fwd_decoder_optimizer.zero_grad()
        self.bck_decoder_optimizer.zero_grad()
        self.bck_decoder_optimizer.zero_grad()
        loss = 0
        loss = self._fwd(input_tensor, target_tensor, self.fwd_encoder, self.fwd_decoder, loss, criterion, max_length=max_length)

        # Problem isn't with the bck_encoder nor bck_decoder, as using them for forward inferenc eworked.
        # Problem doesn't appear to be with detaching the input tensor, nor detaching the fwd_decoded tensor
        # Interesting: If we don't do a forward pass, and then simply target the input_tensor from the fwd_decoded_tensor,
        # then it bombs
        #input_tensor2 = input_tensor.detach()

        # OK, eating its own tail seems to work.  There may be some issue with using the input tensor as the
        # target.  It may not be related to detaching it, as that doing that earlier did not remove the CuDNN execution error.
        #loss = self._fwd(input_tensor, fwd_decoded_tensor, self.bck_encoder, self.bck_decoder, loss, criterion,
        #          max_length=max_length)
        loss = self._fwd(fwd_decoded_tensor, input_tensor, self.bck_encoder, self.bck_decoder, loss, criterion,
                         max_length=max_length)
        loss.backward()
        self.fwd_encoder_optimizer.step()
        self.fwd_decoder_optimizer.step()
        self.bck_encoder_optimizer.step()
        self.bck_decoder_optimizer.step()

        return loss.item() / target_length

    def train(self,
              n_iters,
              train_pairs,
              val_pairs,
              special_pairs,
              save_every=1000,
              sample_every=100,
              eval_every=1000,
              ngpu=2,
              load_checkpoint_if_exist=True):
        """
        Uses special pairs that are sampled into the batch, and are validated separately
        :param n_iters:
        :param train_pairs:
        :param val_pairs:
        :param special_pairs:
        :param save_every:
        :param sample_every:
        :param eval_every:
        :param ngpu:
        :param load_checkpoint_if_exist:
        :return:
        """
        print("Starting training")
        os.makedirs("output", exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        tb_logger = TBLogger(self.log_dir)

        print("Beginning training, total # items={}".format(len(train_pairs)))
        epoch_size = len(train_pairs)
        self.fwd_encoder_optimizer = optim.SGD(self.fwd_encoder.parameters(), lr=self.initial_learning_rate)
        self.fwd_decoder_optimizer = optim.SGD(self.fwd_decoder.parameters(), lr=self.initial_learning_rate)
        self.bck_encoder_optimizer = optim.SGD(self.fwd_encoder.parameters(), lr=self.initial_learning_rate)
        self.bck_decoder_optimizer = optim.SGD(self.fwd_decoder.parameters(), lr=self.initial_learning_rate)

        if load_checkpoint_if_exist:
            self.load_checkpoint()

        #if (device.type == 'cuda') and (ngpu > 1):
        #    encoder = nn.DataParallel(encoder, list(range(ngpu)))
        #    decoder = nn.DataParallel(decoder, list(range(ngpu)))

        selected_train_pair = random.choice(train_pairs)
        training_tensor_pairs = [(selected_train_pair[0], tensors_from_pair(selected_train_pair, self.input_lang, self.output_lang, self.device))
                          for i in range(n_iters)]
        selected_special_pair = random.choice(special_pairs)
        special_tensor_pairs = [(selected_special_pair[0], tensors_from_pair(selected_special_pair, self.input_lang, self.output_lang, self.device))
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()
        progbar = tqdm(range(1, n_iters + 1))
        accum_losses = []  # For computing mean loss
        with open(os.path.join(self.log_dir, "guessed_trials.txt"), "w") as val_log_f:
            for step in progbar:
                training_sentence, training_pair = training_tensor_pairs[step - 1]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]
                loss = self._train_step(training_sentence, input_tensor, target_tensor, criterion)
                accum_losses.append(loss)
                special_sentence, special_pair = special_tensor_pairs[step - 1]
                sinput_tensor = special_pair[0]
                starget_tensor = special_pair[1]
                sloss = self._train_step(special_sentence, sinput_tensor, starget_tensor, criterion)
                accum_losses.append(sloss)
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
                    print("---------------------\nSpecial check")
                    sp_acc, sp_res_str = self.evaluate(special_pairs, n=len(special_pairs), randomize=False)
                    tb_logger.scalar_summary("Special Acc", sp_acc, step)
                    val_log_f.write("\n==========================================\nStep={}\n".format(step))
                    val_log_f.write(res_str)
                    val_log_f.write("\n")
                    val_log_f.write("\n\t* * * * * * * * * *\nSpecials\n* * * * * * * * * *\n".format(step))
                    val_log_f.write(sp_res_str)
                    val_log_f.write("\n")

            epoch = (step // epoch_size) + 1
            self.save_checkpoint(epoch)
            acc, res_str = self.evaluate(val_pairs)
            tb_logger.scalar_summary("Val Acc", acc, step)
            sp_acc, sp_res_str = self.evaluate(special_pairs, n=len(special_pairs), randomize=False)
            tb_logger.scalar_summary("Special Acc", sp_acc, step)
            val_log_f.write("\n==========================================\nStep={}\n".format(step))
            val_log_f.write(res_str)
            val_log_f.write("\n")
            val_log_f.write("\n\t* * * * * * * * * *\nSpecials\n* * * * * * * * * *\n".format(step))
            val_log_f.write(sp_res_str)
            val_log_f.write("\n")

    def evaluate_and_show_attention(self, input_sentence):
        output_words, attentions = self.generate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        show_attention(input_sentence, output_words, attentions)
