import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

from emg_hero.datasets import FeatureDataset


class Trainer:
    def __init__(self, obs, actions, val_obs, val_actions, batch_size, epochs, lr, device):
        self.epochs = epochs
        self.lr = lr
        self.device = device

        train_dataset = FeatureDataset(obs, actions)
        val_dataset = FeatureDataset(val_obs, val_actions)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = torch.nn.BCELoss()

    def train(self, policy):
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        policy.to(
            self.device)
        policy.train()

        for epoch in range(self.epochs):
            for batch in self.train_loader:
                observations, actions = batch
                observations, actions = observations.to(
                    self.device), actions.to(
                    self.device)

                predictions = policy(observations)
                loss = self.criterion(predictions, actions)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
            if epoch % 10 == 0:
                val_loss, val_acc = self.evaluate(policy)
                print(f'Epoch {epoch} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.3f}')

        policy.eval()

    def evaluate(self, policy):
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for batch in self.val_loader:
                observations, actions = batch
                observations, actions = observations.to(
                    self.device), actions.to(
                    self.device)
                predictions = policy(observations)
                pred_actions = (predictions > 0.5).float()
                loss = self.criterion(predictions, actions)
                acc = (pred_actions == actions).all(dim=1).float().mean()
                total_loss += loss
                total_acc += acc

        val_loss = total_loss / len(self.val_loader)
        val_acc = total_acc / len(self.val_loader)
        return val_loss, val_acc
