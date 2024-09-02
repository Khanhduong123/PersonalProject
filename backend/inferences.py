import random
from backend.utils import *
from backend.loader import *
from backend.model import *
from backend.training import *
from backend.evaluate import *
from backend.configs import *
import warnings

warnings.filterwarnings("ignore")


class Inferecences:
    def __init__(self, file_path, lang1, lang2, encoder_path, decoder_path) -> None:
        self.file_path = file_path
        self.lang1 = lang1
        self.lang2 = lang2
        self.encoder = encoder_path
        self.decoder = decoder_path
        self.SOS_token = 0
        self.EOS_token = 1
        self.hidden_size = 128
        self.batch_size = 8
        self.MAX_LENGTH = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def user_input(self, sentences):
        data = Data(
            filepath="data\eng-vie.txt", lang1="eng", lang2="viet", reverse=True
        )
        input_lang, output_lang, pairs = data.prepare()

        loader = Loader(input_lang, output_lang, pairs, self.EOS_token, self.MAX_LENGTH)
        input_lang, output_lang, train_dataloader = loader.get_dataloader(
            self.batch_size
        )

        # Load model
        encoder = Encoder(input_lang.n_words, self.hidden_size).to(self.device)
        decoder = Decoder(self.hidden_size, output_lang.n_words).to(self.device)
        encoder.load_state_dict(torch.load(self.encoder))
        decoder.load_state_dict(torch.load(self.decoder))
        encoder.eval()
        decoder.eval()

        result = evaluate(loader, encoder, decoder, sentences, input_lang, output_lang)[
            :-1
        ]

        return result
