import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms, datasets
from tqdm import tqdm
from model.symbol_alexnet import AlexNet
from model.symbol_alexnet_CBAM import AlexNet_CBAM
from model.symbol_alexnet_CA import AlexNet_CA
from model.symbol_alexnet_FSA import AlexNet_FSA
from model.symbol_alexnet_GCSA import AlexNet_GCSA
from model.symbol_alexnet_SA import AlexNet_SA
from model.symbol_alexnet_LA import AlexNet_LA
from model.symbol_alexnet_PSCA import AlexNet_PSCA
from model.symbol_resnet_CBAM import resnet34_CBAM
from model.symbol_resnet import resnet34
from model.symbol_resnet_PSCA import resnet34_PSCA
from model.symbol_resnet_SA import resnet34_SA
from model.symbol_resnet_CA import resnet34_CA
from model.symbol_resnet_FSA import resnet34_FSA
from model.symbol_resnet_GCSA import resnet34_GCSA
from model.symbol_resnet_LA import resnet34_LA
from model.symbol_zfnet_CBAM import zfnet_CBAM
from model.symbol_zfnet_PSCA import zfnet_PSCA
from model.symbol_zfnet import zfnet
from model.symbol_zfnet_CA import zfnet_CA
from model.symbol_zfnet_FSA import zfnet_FSA
from model.symbol_zfnet_GCSA import zfnet_GCSA
from model.symbol_zfnet_SA import zfnet_SA
from model.symbol_zfnet_LA import zfnet_LA
from model.symbol_MobileNet import MobileNet
from model.symbol_MobileNet_SA import MobileNet_SA
from model.symbol_MobileNet_CA import MobileNet_CA
from model.symbol_MobileNet_FSA import MobileNet_FSA
from model.symbol_MobileNet_GCSA import MobileNet_GCSA
from model.symbol_MobileNet_CBAM import MobileNet_CBAM
from model.symbol_MobileNet_PSCA import MobileNet_PSCA
from model.symbol_MobileNet_LA import MobileNet_LA



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    return device

#
def create_data_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# def create_data_transforms():
#     return {
#         "train": transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#             transforms.RandomRotation(15),
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#             transforms.RandomPerspective(distortion_scale=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
#         ]),
#         "val": transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     }


def load_datasets(data_root, data_transforms):
    image_path = os.path.join(data_root, "select-first")
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transforms["train"]
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transforms["val"]
    )

    return train_dataset, val_dataset


def save_class_indices(dataset, save_path='class_indices.json'):
    class_dict = {v: k for k, v in dataset.class_to_idx.items()}
    with open(save_path, 'w') as f:
        json.dump(class_dict, f, indent=4)


def create_dataloaders(train_dataset, val_dataset, batch_size=128):
    num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    print(f"Using {num_workers} dataloader workers")

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )
    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader


def initialize_model(model_name='MobileNet_CBAM', num_classes=4, device=None):
    model_classes = {
        'AlexNet': AlexNet,
        'AlexNet_CBAM': AlexNet_CBAM,
        'AlexNet_CA': AlexNet_CA,
        'AlexNet_FSA': AlexNet_FSA,
        'AlexNet_GCSA': AlexNet_GCSA,
        'AlexNet_SA': AlexNet_SA,
        'AlexNet_LA': AlexNet_LA,
        'AlexNet_PSCA': AlexNet_PSCA,
        'resnet34': resnet34,
        'resnet34_CBAM': resnet34_CBAM,
        'resnet34_PSCA': resnet34_PSCA,
        'resnet34_SA': resnet34_SA,
        'resnet34_CA': resnet34_CA,
        'resnet34_LA': resnet34_LA,
        'resnet34_FSA': resnet34_FSA,
        'resnet34_GCSA': resnet34_GCSA,
        'zfnet': zfnet,
        'zfnet_CBAM': zfnet_CBAM,
        'zfnet_PSCA': zfnet_PSCA,
        'zfnet_CA': zfnet_CA,
        'zfnet_FSA': zfnet_FSA,
        'zfnet_GCSA': zfnet_GCSA,
        'zfnet_SA': zfnet_SA,
        'zfnet_LA': zfnet_LA,
        'MobileNet': MobileNet,
        'MobileNet_CBAM': MobileNet_CBAM,
        'MobileNet_SA': MobileNet_SA,
        'MobileNet_CA': MobileNet_CA,
        'MobileNet_FSA': MobileNet_FSA,
        'MobileNet_GCSA': MobileNet_GCSA,
        'MobileNet_LA': MobileNet_LA,
        'MobileNet_PSCA': MobileNet_PSCA
    }

    model_class = model_classes.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model_class(num_classes=num_classes)

    # Modify last layer if needed
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    return model


def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0

    with tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{total_epochs}", file=sys.stdout) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            pbar.set_postfix(loss=running_loss / total_samples)

    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, device, epoch, total_epochs):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{total_epochs}", file=sys.stdout) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix(acc=100 * correct / total)

    return correct / total


def train_model(config):
    # Setup
    device = setup_device()
    data_transforms = create_data_transforms()

    # Data loading
    data_root = r"C:\Users\Administrator\Desktop\38044"
    train_dataset, val_dataset = load_datasets(data_root, data_transforms)
    save_class_indices(train_dataset)

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config['batch_size']
    )

    # Model initialization
    model = initialize_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        device=device
    )

    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    total_time = 0

    for epoch in range(config['epochs']):
        start_time = time.time()

        # Train and validate
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer,
            device, epoch, config['epochs']
        )
        val_acc = validate(model, val_loader, device, epoch, config['epochs'])

        # Update learning rate
        scheduler.step(val_acc)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])
            print(f"New best model saved with accuracy: {best_acc:.2%}")

        # Log epoch info
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train loss: {train_loss:.4f} - "
              f"Val acc: {val_acc:.2%} - "
              f"Time: {epoch_time:.2f}s")

    # Final summary
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"\nTraining complete in {h:.0f}h {m:.0f}m {s:.0f}s")
    print(f"Best validation accuracy: {best_acc:.2%}")

    return model, history


if __name__ == '__main__':
    config = {
        'model_name': 'AlexNet_LA',
        'num_classes': 4,
        'batch_size': 128,
        'epochs': 200,
        'learning_rate': 0.00008,
        'save_path': r'C:\Users\Administrator\Desktop\classification\weights\select\消融实验大数据集\下采样\max'
    }

    train_model(config)