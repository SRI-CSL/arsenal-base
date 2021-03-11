import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ars2seq2seq.models.seq2seq import *
from ars2seq2seq.util.polish import *

# from IPython.core.debugger import set_trace

DEFAULT_MAX_LENGTH=1000

class MultiAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device,
                 num_attn=2,  # Number of attention units
                 dropout_p=0.1,
                 bidirectional=False, num_layers=1,
                 max_length=DEFAULT_MAX_LENGTH):
        super(MultiAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        self.bidirectional, self.num_layers = bidirectional, num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.num_attn = num_attn
        #self.attns = [nn.Linear(self.hidden_size * 2, self.max_length) for i in range(num_attn)]
        self.attn0 = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn1 = nn.Linear(self.hidden_size * 2, self.max_length)

        #self.attn_combine = [nn.Linear(self.hidden_size * 2, self.hidden_size) for i in range(num_attn)]
        self.attn_combine0 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine1 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights0 = F.softmax(self.attn0(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_weights1 = F.softmax(self.attn1(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied0 = torch.bmm(attn_weights0.unsqueeze(0), encoder_outputs.unsqueeze(0))
        attn_applied1 = torch.bmm(attn_weights1.unsqueeze(0), encoder_outputs.unsqueeze(0))
        loutput0 = torch.cat((embedded[0], attn_applied0[0]), 1)
        loutput0 = self.attn_combine0(loutput0).unsqueeze(0)
        loutput1 = torch.cat((embedded[0], attn_applied1[0]), 1)
        loutput1 = self.attn_combine1(loutput1).unsqueeze(0)
        output = torch.stack((loutput0, loutput1), 0)
        output = torch.mean(output, dim=0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, None

    def initHidden(self):
        # Todo: consider better initialization?
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)


def get_wordtype(word):
    tuples = word.split("#")
    if len(tuples) == 1:
        return word
    return tuples[1]


class MultiHeadSeq2Seq():
    def __init__(self, hidden_size,
                 input_lang, output_lang,
                 bidirectional=False,
                 max_length=DEFAULT_MAX_LENGTH,
                 num_layers=1,
                 num_attn = 2,
                 initial_learning_rate=0.001,
                 dropout=0.1,
                 output_root="./output",
                 verbose=True,
                 device=None,
                 match_parens=False,
                 init_type=""):
        self.input_size = input_lang.n_words
        self.output_size = output_lang.n_words
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_attn = num_attn
        self.input_lang, self.output_lang = input_lang, output_lang
        self.bidirectional, self.num_layers = bidirectional, num_layers
        self.verbose = verbose
        self.encoder = EncoderRNN(self.input_size, hidden_size, device, bidirectional=bidirectional, num_layers=num_layers)
        self.decoder = MultiAttnDecoderRNN(hidden_size, self.output_size, device,
                                           bidirectional=bidirectional, num_layers=num_layers,
                                           max_length=self.max_length, dropout_p=dropout,
                                           num_attn=num_attn)
        self.device = device
        self.match_parens = match_parens
        self.init_type = init_type

        if (self.init_type == ''):
            self.type_filtering=False
        else:
            self.type_filtering=True
            # Check against vocabulary to see if the specified hole type is in the vocabulary.
            # Raise an exception if not.
            all_types = set([get_wordtype(word) for word in self.output_lang.word2index.keys()])
            if not(self.init_type in all_types):
                raise Exception("Error, specified hole type={} must be in vocabulary!".format(self.init_type))

        if device:
            self.encoder.to(device)
            self.decoder.to(device)
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.initial_learning_rate = initial_learning_rate
        self.teacher_forcing_ratio = 0.5
        self.root_dir = output_root
        self.output_root = os.path.join(output_root, self.exp_name())
        self.checkpoint_dir = os.path.join(self.output_root, "checkpoints")
        self.log_dir = os.path.join(self.output_root, "logs")

    def exp_name(self):
        return "h={}_nl={}_nattn={}_lr={}".format(self.hidden_size, self.num_layers, self.num_attn, self.initial_learning_rate)

    def __str__(self):
        res = "\n".join([self.exp_name(), "in={}".format(self.input_lang), "out={}".format(self.output_lang),
                         "input_size={}".format(self.input_size),
                         "output_size={}".format(self.output_size)])

        return res


    def select_token(self, decoder_output, hole_type):
        if self.type_filtering and (self.output_lang.name == 'pn'):
            # We are performing type filtering, and our targets are in polish notation (pn).  Instead of selecting just
            # the top-1 element, we select the entire vocabulary, and sort through them to filter on the valid
            # type.
            vals,indices = decoder_output.data.topk(decoder_output.data.size()[1], sorted=True)
            indices = indices.detach()[0]
            # Go through each of the indices (in order), and stop at the first one that matches the
            # specified hole_type.
            print("New token of type {} to produce".format(hole_type))
            for i in indices:
                token = i.item()
                word = self.output_lang.index2word[token]
                print("We test token {}, i.e. word {}".format(i,word))
                word_type = get_wordtype(word)
                if word_type == hole_type:
                    return i
            # If we've bottomed out, throw error
            raise Exception("None of possible vocabulary items matches hole type!")
        else:
            topv, topi = decoder_output.data.topk(1)
            # Check for ties
            if len(decoder_output.data == topv) > 1:
                print("Several ties, topv={}, topi={}".format(topv, topi))
            return topi

    def load_checkpoint(self, override_checkpoint_dir=None):
        checkpoint_dir = self.checkpoint_dir
        if override_checkpoint_dir is not None:
            checkpoint_dir = override_checkpoint_dir
        if checkpoint_exists("encoder", checkpoint_dir) and checkpoint_exists("decoder", checkpoint_dir):
            if self.verbose:
                print("Checkpoints exist at {}, loading".format(checkpoint_dir))
            load_checkpoint(self.encoder, "encoder", self.encoder_optimizer, checkpoint_dir, device=self.device)
            load_checkpoint(self.decoder, "decoder", self.decoder_optimizer, checkpoint_dir, device=self.device)
            # Return iterations [DE]
            with open(os.path.join(checkpoint_dir, "setup.json"), "r") as f:
                args = json.load(f)
                iters=args.get('iterations')
                return iters
        else:
            if self.verbose:
                print("Checkpoints do not exist, no action taken")
            return 0

    def save_checkpoint(self, epoch, iters):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        setup_json = {
            'input_lang' : self.input_lang.name,
            'output_lang' : self.output_lang.name,
            'max_length' : self.max_length,
            'num_layers' : self.num_layers,
            'num_attn' : self.num_attn,
            'hidden_size' : self.hidden_size,
            'match_parens' : self.match_parens,
            'init_type' : self.init_type,
            'dropout': self.decoder.dropout_p,
            'iterations': iters
        }
        with open(os.path.join(self.checkpoint_dir, "setup.json"), "w") as f_out:
            f_out.write(json.dumps(setup_json, indent=4, sort_keys=True))

        with open(os.path.join(self.checkpoint_dir, "notes.txt"), "w") as f:
            f.write("input_lang={}\n".format(self.input_lang))
            f.write("output_lang={}\n".format(self.output_lang))
            f.write("max_length={}\n".format(self.max_length))
            f.write("num_layers={}\n".format(self.num_layers))
            f.write("num_attn={}\n".format(self.num_attn))
            f.write("hidden_size={}\n".format(self.hidden_size))
            f.write("match_parens={}\n".format(self.match_parens))            
            f.write("init_type={}\n".format(self.init_type))            
            f.write("dropout={}\n".format(self.decoder.dropout_p))            
            f.write("iterations={}\n".format(iters))            
        save_checkpoint("encoder", self.encoder, self.encoder_optimizer, self.checkpoint_dir, epoch)
        save_checkpoint("decoder", self.decoder, self.decoder_optimizer, self.checkpoint_dir, epoch)
        self.input_lang.save(override_fpath=os.path.join(self.checkpoint_dir,
                                                         "{}.vocab".format(self.input_lang.name)))
        self.output_lang.save(override_fpath=os.path.join(self.checkpoint_dir,
                                                          "{}.vocab".format(self.output_lang.name)))

    def generate(self, sentence, include_attentions=False):
        #print("multi_attn generate, max length {}, output lang {}, mp {}".format(self.max_length, 
        #    self.output_lang.name, self.match_parens, self.type_filtering))
        max_length = self.max_length
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

            # In the decoder loop below, count the number of unmatched '['
            # and stop decoding when we reach 0 after a ']'.
            # Note that we can still get unmatched parentheses if we reach max_length.
            # We could add the missing right brackets in that case.
            unmatched_brackets = 0
            expected_args = 1
            hole_types = [self.init_type]
            
            for di in range(max_length):
                #print("Decoded words: {}, expected args: {}".format(decoded_words, expected_args))
                
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                if include_attentions:
                    decoder_attentions[di] = decoder_attention.data

                # TODO: Identify if we are getting ties here. Could they be the source of ambiguities?
                topi = self.select_token(decoder_output, hole_types[0])
                token = topi.item()
                    
                if self.type_filtering and (self.output_lang.name == 'pn'):
                    word = self.output_lang.index2word[token]
                    op_arity = re.split('#',word)
                    op = op_arity[0]
                    args = op_arity[2:]
                    arity = len(args)
                    decoded_words.append("{}#{}#{}".format(op, arity, op_arity[1]))
                    hole_types = args + hole_types[1:]
                    if len(hole_types) == 0:
                        break  # Nothing left to match!
                else:

                    if self.match_parens and (self.output_lang.name == 'sexp' or self.output_lang.name == 'json'):
                        decoded_words.append(self.output_lang.index2word[token])
                        if self.output_lang.is_opening_bracket(token):
                            unmatched_brackets += 1
                        elif self.output_lang.is_closing_bracket(token):
                            unmatched_brackets -= 1
                            if unmatched_brackets < 0:
                                print("Output starting with right bracket")
                                break
                            elif unmatched_brackets == 0:
                                # No more unmatched opening brackets, i.e. we're done!
                                break
                    if self.match_parens and (self.output_lang.name == 'pn'):
                        word = self.output_lang.index2word[token]
                        decoded_words.append(word)
                        op_arity = get_pn_op_arity(word)
                        if op_arity:  # it is a non-terminal
                            [op,arity] = op_arity
                            expected_args += arity-1  # -1 because the new nonterminal fills an arg position one level up
                        else: #it is a terminal
                            expected_args -= 1
                            if expected_args <= 0:
                                break  # Nothing left to match!
                    
                    else:
                        if token == EOS_token:
                            decoded_words.append('<EOS>')
                            break
                        else:
                            decoded_words.append(self.output_lang.index2word[token])
                    
                decoder_input = topi.squeeze().detach()

            if include_attentions:
                return decoded_words, decoder_attentions[:di + 1]
            else:
                return decoded_words, None

    def evaluate(self, eval_pairs, n=30, randomize=True):
        if n is None:
            n = len(eval_pairs)
        if randomize:
            to_eval = [random.choice(eval_pairs) for i in range(n)]
        else:
            to_eval = eval_pairs[0:n]

        def generate_fn(sentence):
            return self.generate(sentence)

        return evaluate_pairs(generate_fn, to_eval)

    def _train_step(self, input_tensor, target_tensor, criterion):
        max_length = self.max_length
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        if input_length > max_length:
            print("WARNING: Requested input_length={} > max_length={}".format(input_length, max_length))

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        # Consider granular teacher forcing, drawing at each point
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
            unmatched_brackets = 0
            expected_args = 1
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                token = decoder_input.item()
                
                if self.match_parens and (self.output_lang.name == 'sexp' or self.output_lang.name == 'json'):
                    if self.output_lang.is_opening_bracket(token):
                        unmatched_brackets += 1
                    elif self.output_lang.is_closing_bracket(token):
                        unmatched_brackets -= 1
                        if unmatched_brackets < 0:
                            print("Output starting with right bracket")
                            break
                        elif unmatched_brackets == 0:
                            # No more unmatched opening brackets, i.e. we're done!
                            break
                if self.match_parens and (self.output_lang.name == 'pn'):
                    word = self.output_lang.index2word[token]
                    if self.type_filtering:
                        op_arity = get_typedpn_op_arity(word)
                    else:
                        op_arity = get_pn_op_arity(word)
                    if op_arity:  # it is a non-terminal
                        [op,arity] = op_arity
                        expected_args += arity-1  # -1 because the new nonterminal fills an arg position one level up
                    else: #it is a terminal
                        expected_args -= 1
                    if expected_args <= 0:
                        break;  # Nothing left to match!
                else:
                    if token == EOS_token:
                        break
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / target_length

    def train(self,
              n_iters,
              train_pairs,
              val_pairs,
              special_pairs=None,
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
        :param start_iter:
        :param special_pairs:
        :param save_every:
        :param sample_every:
        :param eval_every:
        :param ngpu:
        :param load_checkpoint_if_exist:
        :return:
        """
        print("Starting training")
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        tb_logger = TBLogger(self.log_dir)

        print("Beginning training, total # items={}".format(len(train_pairs)))
        print("# train_pairs={}, # val_pairs={}".format(len(train_pairs), len(val_pairs)))
        epoch_size = len(train_pairs)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.initial_learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.initial_learning_rate)

        start_iter = 0
        if load_checkpoint_if_exist:
            start_iter = self.load_checkpoint()

        #if (device.type == 'cuda') and (ngpu > 1):
        #    encoder = nn.DataParallel(encoder, list(range(ngpu)))
        #    decoder = nn.DataParallel(decoder, list(range(ngpu)))

        # Note: May be doing a lot of redundant work here, as pairs may already have tensor versions.
        print("Generating Training pairs")
        tqdm_n_iters = tqdm(range(n_iters))
        training_tensor_pairs = [tensors_from_pair(random.choice(train_pairs), self.input_lang, self.output_lang, self.device)
                          for i in tqdm_n_iters]
        if special_pairs is not None:
            print("Vectorizing specials")
            special_tensor_pairs = [tensors_from_pair(random.choice(special_pairs), self.input_lang, self.output_lang, self.device)
                              for i in range(n_iters)]
        criterion = nn.NLLLoss()
        progbar = tqdm(range(start_iter+1, n_iters + 1))
        accum_losses = []  # For computing mean loss
        orig_type_filtering = self.type_filtering
        with open(os.path.join(self.log_dir, "guessed_trials.txt"), "w") as val_log_f:
            step = 0
            for step in progbar:
                training_pair = training_tensor_pairs[step - 1]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]
                loss = self._train_step(input_tensor, target_tensor, criterion)
                accum_losses.append(loss)
                if special_pairs is not None:
                    special_pair = special_tensor_pairs[step - 1]
                    sinput_tensor = special_pair[0]
                    starget_tensor = special_pair[1]
                    sloss = self._train_step(sinput_tensor, starget_tensor, criterion)
                    accum_losses.append(sloss)
                progbar.set_description("Loss={:.5f}".format(loss))

                if step % sample_every == 0:
                    mean_loss = np.mean(accum_losses)
                    tb_logger.scalar_summary("Mean Loss", mean_loss, step)
                    accum_losses = []

                if step % save_every == 0:
                    epoch = (step // epoch_size) + 1
                    self.save_checkpoint(epoch, step)

                if step % eval_every == 0:
                    print("---------------------\nTrain check w/o type-filtering")
                    self.type_filtering = False
                    train_accNTF, train_res_strNTF = self.evaluate(train_pairs, n=10, randomize=False)
                    tb_logger.scalar_summary("Train Acc NTF", train_accNTF, step)
                    val_log_f.write("\n==========================================\nStep={}\n".format(step))
                    val_log_f.write("Training check\n")
                    val_log_f.write(train_res_strNTF)
                    val_log_f.write("\n")

                    print("---------------------\nTrain check w type-filtering")
                    self.type_filtering = True
                    train_acc, train_res_str = self.evaluate(train_pairs, n=10, randomize=False)
                    tb_logger.scalar_summary("Train Acc", train_acc, step)
                    val_log_f.write("\n==========================================\nStep={}\n".format(step))
                    val_log_f.write("Training check\n")
                    val_log_f.write(train_res_str)
                    val_log_f.write("\n")

                    print("---------------------\nValidate check w/o type-filtering")
                    self.type_filtering = False
                    accNTF, res_strNTF = self.evaluate(val_pairs, randomize=False)
                    tb_logger.scalar_summary("Val Acc NTF", accNTF, step)
                    if special_pairs is not None:
                        print("---------------------\nSpecial check w/o type-filtering")
                        sp_accNTF, sp_res_strNTF = self.evaluate(special_pairs, n=len(special_pairs), randomize=False)
                        tb_logger.scalar_summary("Special Acc NTF", sp_accNTF, step)
                    #val_log_f.write("\n==========================================\nStep={}\n".format(step))
                    val_log_f.write("\nValidation check\n")
                    val_log_f.write(res_strNTF)
                    val_log_f.write("\n")
                    if special_pairs is not None:
                        val_log_f.write("\n\t* * * * * * * * * *\nSpecials\n* * * * * * * * * *\n".format(step))
                        val_log_f.write(sp_res_strNTF)
                        val_log_f.write("\n")

                    print("---------------------\nValidate check w type-filtering")
                    self.type_filtering = True
                    acc, res_str = self.evaluate(val_pairs, randomize=False)
                    tb_logger.scalar_summary("Val Acc", acc, step)
                    if special_pairs is not None:
                        print("---------------------\nSpecial check w type-filtering")
                        sp_acc, sp_res_str = self.evaluate(special_pairs, n=len(special_pairs), randomize=False)
                        tb_logger.scalar_summary("Special Acc", sp_acc, step)
                    #val_log_f.write("\n==========================================\nStep={}\n".format(step))
                    val_log_f.write("\nValidation check\n")
                    val_log_f.write(res_str)
                    val_log_f.write("\n")
                    if special_pairs is not None:
                        val_log_f.write("\n\t* * * * * * * * * *\nSpecials\n* * * * * * * * * *\n".format(step))
                        val_log_f.write(sp_res_str)
                        val_log_f.write("\n")
                        
                    self.type_filtering = orig_type_filtering

            epoch = (step // epoch_size) + 1
            self.save_checkpoint(epoch, step)

            self.type_filtering = False
            train_accNTF, train_res_strNTF = self.evaluate(train_pairs, n=10, randomize=False)
            tb_logger.scalar_summary("Train Acc NTF", train_accNTF, step)

            accNTF, res_strNTF = self.evaluate(val_pairs, randomize=False)
            tb_logger.scalar_summary("Val Acc NTF", accNTF, step)
            if special_pairs is not None:
                sp_accNTF, sp_res_strNTF = self.evaluate(special_pairs, n=len(special_pairs), randomize=False)
                tb_logger.scalar_summary("Special Acc NTF", sp_accNTF, step)

            self.type_filtering = True
            train_acc, train_res_str = self.evaluate(train_pairs, n=10, randomize=False)
            tb_logger.scalar_summary("Train Acc", train_acc, step)

            acc, res_str = self.evaluate(val_pairs, randomize=False)
            tb_logger.scalar_summary("Val Acc", acc, step)
            if special_pairs is not None:
                sp_acc, sp_res_str = self.evaluate(special_pairs, n=len(special_pairs), randomize=False)
                tb_logger.scalar_summary("Special Acc", sp_acc, step)

            val_log_f.write("\n==========================================\nStep={}\n".format(step))
            val_log_f.write("Training pair check w/o type-filtering\n")
            val_log_f.write(train_res_strNTF)
            val_log_f.write("\nValidation pair check w/o type-filtering\n")
            val_log_f.write(res_strNTF)
            val_log_f.write("\n")
            if special_pairs is not None:
                val_log_f.write("\n\t* * * * * * * * * *\nSpecials\n* * * * * * * * * *\n".format(step))
                val_log_f.write(sp_res_strNTF)
                val_log_f.write("\n")

            val_log_f.write("Training pair check w type-filtering\n")
            val_log_f.write(train_res_str)
            val_log_f.write("\nValidation pair check w type-filtering\n")
            val_log_f.write(res_str)
            val_log_f.write("\n")
            if special_pairs is not None:
                val_log_f.write("\n\t* * * * * * * * * *\nSpecials\n* * * * * * * * * *\n".format(step))
                val_log_f.write(sp_res_str)
                val_log_f.write("\n")

    def evaluate_and_show_attention(self, input_sentence):
        output_words, attentions = self.generate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        show_attention(input_sentence, output_words, attentions)
