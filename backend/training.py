from datetime import timedelta
import math
import sys
import time
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch import optim
import tqdm
from backend.loader import *

plt.switch_backend("agg")


class Trainer:
    def __init__(
        self, trainloader, encoder, decoder, n_epochs, learning_rate=0.001
    ) -> None:

        self.train_loader = trainloader
        self.encoder = encoder
        self.decoder = decoder
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss()
        self.cache = {
            "Train_loss": [],
        }
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

    def forward(self):

        self.encoder.train()
        self.decoder.train()

        cache = {"loss": []}
        total_loss = 0

        for data in self.train_loader:
            input_tensor, target_tensor = data

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, _, _ = self.decoder(
                encoder_outputs, encoder_hidden, target_tensor
            )

            loss = self.criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1),
            )
            loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            total_loss += loss.item()

            cache["loss"].append(loss.item())
        loss = sum(cache["loss"]) / len(cache["loss"])
        self.cache[f"Train_loss"].append(loss)

    def _timeSince(self, start, percent):
        now = time.time()
        s = now - start
        es = s / percent
        rs = es - s
        return f"{timedelta(seconds=int(s))} (remaining: {timedelta(seconds=int(rs))})"

    def fit(self):
        if torch.cuda.is_available():
            print(
                f"Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}"
            )
        else:
            print("Running on: CPU")

        # Calculate the total number of update steps
        total_update_steps = len(self.train_loader) * self.n_epochs
        print(f"Total update step: {total_update_steps}")

        # Initialize logs list to store training logs
        logs = []

        # Training loop for the specified number of epochs
        for epoch in tqdm.tqdm(range(1, self.n_epochs + 1)):
            # print(f"Epoch: {epoch}")

            try:
                # Perform the forward pass and calculate training loss
                self.forward()
                train_loss = round(self.cache["Train_loss"][-1], 5)
                logs.append(f"\t=> Train epoch: loss: {train_loss}")
            except KeyboardInterrupt:
                sys.exit(0)

            # Print the training loss for the current epoch
            print(f"\tTrain Loss: {train_loss}")

    def save_model(self, encoder_path, decoder_path):
        """Save the encoder and decoder models to the specified paths."""
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        print(f"Encoder saved to {encoder_path}")
        print(f"Decoder saved to {decoder_path}")