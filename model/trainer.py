import os
import numpy as np

import torch

from datetime import datetime

class Trainer:

    def __init__(self, optimizer, criterion, output_path, device):

        self.optimizer = optimizer
        self.criterion = criterion
        self.output_path = output_path
        self.device = device

    def train(self, model, train_loader, epoch):

        train_loss, correct = 0.0, 0
        model = model.float().to(self.device)

        model.train()

        print(f"\nEpoch {epoch+1}")

        for idx, (images,labels) in enumerate(train_loader):

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = model(images.float())

            loss = self.criterion(outputs,labels)
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(labels == predicted).item()

            loss.backward()
            self.optimizer.step()

        avg_loss = train_loss / (idx + 1)
        accuracy = 100. * correct / len(train_loader.dataset)

        return avg_loss, accuracy

    def validate(self, model, val_loader):

        val_loss, correct = 0.0, 0
        model = model.float().to(self.device)
        model.eval()

        with torch.no_grad():
            for idx, (images, labels) in enumerate(val_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images.float())
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.sum(labels == predicted).item()

        avg_loss = val_loss / (idx + 1)
        accuracy = 100. * correct / len(val_loader.dataset)

        return avg_loss, accuracy

