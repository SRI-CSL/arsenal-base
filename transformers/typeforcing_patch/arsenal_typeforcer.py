from .generation_logits_process import LogitsProcessor
import torch

class ArsenalTypeLogitsProcessor(LogitsProcessor):

    def __init__(self, id2vocab: dict):
        self.id2vocab = id2vocab
        for id, token in id2vocab.items():
            if token == "[PAD]":
                self.pad_id = id
            elif token == "[SEP]":
                self.eos_id = id


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # don't attempt type forcing for the first token (special token to signal begin of sequence)
        if(input_ids.shape[-1] == 1):
            return scores

        penalties = []

        # iterate over instances in batch
        for i in range(input_ids.shape[0]):
            inst = input_ids[i,:]

            hole_types = []

            # iterate over tokens in instance
            for j in inst:
                word = self.id2vocab[(int(j))]
                op_arity = word.split("#")
                op = op_arity[0]
                args = op_arity[2:]
                arity = len(args)
                hole_types = args + hole_types[1:]

            # only allow padding tokens if we are already at or beyond the sequence's content
            # (just to handle all cases consistently - we could probably use zeros here just
            # as well b/c the generation should stop after reaching the EOS token)
            if int(inst[-1]) == self.eos_id or int(inst[-1]) == self.pad_id:
                penalty_vec = list(map(lambda x:
                         0 if x == self.pad_id
                         else -float("inf"),
                         range(len(self.id2vocab))))

            # if hole_types is empty, we are done and should only allow for the eos token
            elif len(hole_types) == 0:
                penalty_vec = [-float("inf")] * len(self.id2vocab)
                penalty_vec[self.eos_id] = 0

            # do the actual typeforcing
            else:
                penalty_vec = list(map(lambda x:
                       -float("inf") if self.id2vocab[x].startswith("[") or
                                        self.id2vocab[x].split("#")[1] != hole_types[0]
                        else 0,
                        range(len(self.id2vocab))))

            penalties.append(penalty_vec)

            ## for debugging only
            print(f"********\ninput ids: {inst.tolist()}")
            # for t in inst.tolist():
            #     print(f"   {int(t):>3}: {self.id2vocab[int(t)]}")
            print(f"---\nhole types: {hole_types}")
            allowed_tokens = [t for t, x in enumerate(penalty_vec) if x > -float("inf")]

            inst_scores = scores[i,:]
            penalty_vec = torch.tensor(penalty_vec, device=input_ids.device)
            next_cand = int(torch.argmax(inst_scores))

            print(f"current token: {inst[-1]} ({self.id2vocab[int(inst[-1])]})")
            print(f"allowed tokens: {allowed_tokens} (predicted: {next_cand})")
            for t in allowed_tokens:
                print(f"   {int(t):>3}: {self.id2vocab[int(t)]}")
            print("\n")

            mod = inst_scores + penalty_vec
            forced_next_cand = int(torch.argmax(mod))
            if next_cand != forced_next_cand:
                if not (int(inst[-1]) == self.eos_id or int(inst[-1]) == self.pad_id):
                    print(f"forced alternative prediction! Change from {next_cand} ({self.id2vocab[next_cand]}) to {forced_next_cand}")

        print("...... batch end ......")
        ## end for debugging only

        penalties = torch.tensor(penalties, device=input_ids.device)
        scores = scores + penalties

        return scores