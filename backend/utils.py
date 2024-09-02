import re
import unicodedata


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Data:
    def __init__(self, filepath, lang1, lang2, reverse=False) -> None:
        self.filepath = filepath
        self.lang1 = lang1
        self.lang2 = lang2
        self.reverse = reverse

    def _unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    # Lowercase, trim, and remove non-letter characters
    def _normalizeString(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        return s

    def _readLangs(self):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open(self.filepath, encoding="utf-8").read().strip().split("\n")

        # Split every line into pairs and normalize
        pairs = [[self._normalizeString(s) for s in l.split("\t")[:2]] for l in lines]

        # Reverse pairs, make Lang instances
        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.lang2)
            output_lang = Lang(self.lang1)
        else:
            input_lang = Lang(self.lang1)
            output_lang = Lang(self.lang2)

        return input_lang, output_lang, pairs

    def prepare(self):
        input_lang, output_lang, pairs = self._readLangs()
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        # print(input_lang.name, input_lang.n_words)
        # print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs
