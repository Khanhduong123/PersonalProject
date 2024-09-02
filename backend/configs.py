import torch


class Parameters:
    def __init__(self) -> None:
        self.SOS_token = 0
        self.EOS_token = 1
        self.hidden_size = 128
        self.batch_size = 8
        self.MAX_LENGTH = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
