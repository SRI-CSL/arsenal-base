from ars2seq2seq.util.vocab import process_sentence
from tqdm import tqdm

EOS_TOK = "<EOS>"

def exact_match(gold_words, guessed_words):
    """
    Computes the exact match accuracy between a gold and guessed word sequence.
    <EOS> at the end of the guessed word sequence is removed.
    Matches are conducted on the string form of the tokens.
    :param gold_seq:
    :param guess_seq:
    :return:
    """
    if len(guessed_words) > 0 and guessed_words[-1] == EOS_TOK:
        guessed_words = guessed_words[:-1]
    gold_str = " ".join(gold_words)
    guessed_str = " ".join(guessed_words)

    # print("Match: Gold={}, Guess={}".format(gold_str, guessed_str))

    return gold_str == guessed_str


def exact_acc(gold_guess_pairs):
    """For a sequence of (gold, guess) word sequence pairs, returns the exact accuracy."""
    if len(gold_guess_pairs) == 0:
        return 0, []
    num_correct = 0
    for (gold, guess) in gold_guess_pairs:
        if exact_match(gold, guess):
           num_correct += 1
    return num_correct


def evaluate_pairs(evaluate_fn, eval_pairs):
    num_correct = 0
    res_str = ""

    def emit(str):
        nonlocal res_str
        res_str += str
        res_str += "\n"
        return res_str

    i_iter = tqdm(range(len(eval_pairs)))
    print("Evaluating pairs")
    for i in i_iter:
        pair = eval_pairs[i]
        lang1_str, lang2_str = pair
        emit("# {}".format(i))
        lang1_toks = process_sentence(lang1_str)
        lang2_toks = process_sentence(lang2_str)
        num_lang1_toks = len(lang1_toks)
        num_lang2_toks = len(lang2_toks)
        guessed_words, _ = evaluate_fn(lang1_str)
        guessed_sentence = ' '.join(guessed_words)
        num_guess_toks = len(guessed_words)
        is_correct = exact_match(lang2_toks, guessed_words)
        if is_correct:
            num_correct += 1
        emit('Lang (#={}):\t{}'.format(num_lang1_toks, pair[0]))
        emit('Gold (#={}):\t{}'.format(num_lang2_toks, pair[1]))
        if is_correct:
            emit('Guess (#={}):\t{}'.format(num_guess_toks, guessed_sentence))
        else:
            emit('* Guess (#={}):\t{}'.format(num_guess_toks, guessed_sentence))
        emit('')
    acc = num_correct / len(eval_pairs)
    emit("Exact acc = {:.5f}".format(acc))
    return acc, res_str
