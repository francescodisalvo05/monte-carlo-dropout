import os
import numpy as np

import torch

from datetime import datetime

class Trainer:

    def __init__(self, optimizer, criterion, output_path):

        self.optimizer = optimizer
        self.criterion = criterion
        self.output_path = output_path

        pass

    def train(self, model, train_loader, epoch):

        train_loss, correct = 0.0, 0
        model = model.float()

        model.train()

        print(f"\nEpoch {epoch+1}")

        for idx, (images,labels) in enumerate(train_loader):

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
        model = model.float()
        model.eval()

        with torch.no_grad():
            for idx, (images, labels) in enumerate(val_loader):
                outputs = model(images.float())
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.sum(labels == predicted).item()

        avg_loss = val_loss / (idx + 1)
        accuracy = 100. * correct / len(val_loader.dataset)

        return avg_loss, accuracy

    def save_ckpt(self, model, epoch, ckpt_path):

        ckpt_filepath = os.path.join(ckpt_path,'ckpt.pth')

        torch.save({
            'num_epochs': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_filepath)