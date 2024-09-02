import torch
import numpy as np
from backend.utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# SOS_token = 0
# EOS_token = 1
# hidden_size = 128
# batch_size = 8
# MAX_LENGTH = 64


class Loader:
    def __init__(self, input_lang, output_lang, pairs, eos_token, max_length):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.EOS_token = eos_token
        self.MAX_LENGTH = max_length

    def tokenize(self, sentence):
        return [word for word in sentence.split(" ")]

    def _indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(" ")]

    def _tensorFromSentence(self, lang, sentence):
        indexes = self._indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(1, -1)

    def _tensorsFromPair(self, pair):
        input_tensor = self._tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self._tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    def get_dataloader(self, batch_size):
        # input_lang, output_lang, pairs = Data.prepare()

        n = len(self.pairs)
        input_ids = np.zeros((n, self.MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, self.MAX_LENGTH), dtype=np.int32)

        # tokenzing the sentences
        # last token is EOS : 1
        for idx, (inp, tgt) in enumerate(self.pairs):
            inp_ids = self._indexesFromSentence(self.input_lang, inp)
            tgt_ids = self._indexesFromSentence(self.output_lang, tgt)
            inp_ids.append(self.EOS_token)
            tgt_ids.append(self.EOS_token)
            input_ids[idx, : len(inp_ids)] = inp_ids
            target_ids[idx, : len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(
            torch.LongTensor(input_ids).to(self.device),
            torch.LongTensor(target_ids).to(self.device),
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        return self.input_lang, self.output_lang, train_dataloader
