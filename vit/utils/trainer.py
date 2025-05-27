import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, epoch, scaler):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (total / len(self.train_loader.dataset)),
                'acc': correct / total
            })

        train_loss = running_loss / len(self.train_loader)
        train_acc = correct / total
        return train_loss, train_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = correct / total
        return val_loss, val_acc