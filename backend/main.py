import random
from backend.utils import *
from backend.loader import *
from backend.model import *
from backend.training import *
from backend.evaluate import *
import warnings

warnings.filterwarnings("ignore")
SOS_token = 0
EOS_token = 1
hidden_size = 128
batch_size = 8
MAX_LENGTH = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    data = Data(filepath="eng-vie.txt", lang1="eng", lang2="viet", reverse=True)
    input_lang, output_lang, pairs = data.prepare()

    loader = Loader(input_lang, output_lang, pairs, EOS_token, MAX_LENGTH)
    input_lang, output_lang, train_dataloader = loader.get_dataloader(batch_size)

    encoder = Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = Decoder(hidden_size, output_lang.n_words).to(device)
    trainer = Trainer(
        trainloader=train_dataloader, encoder=encoder, decoder=decoder, n_epochs=1
    )
    trainer.fit()
    trainer.save_model("encoder.pth", "decoder.pth")


if __name__ == "__main__":
    main()
