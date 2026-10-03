import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from cnn import CNN

def main():
    data = CIFAR10("/data/data/cifar10", train=False, transform=ToTensor())
    model = CNN().to(CNN.device)
    model.load_state_dict(torch.load("model.pth", map_location=model.device))
    model.eval()

    total, correct = 0, 0
    test_loss = 0.0
    loader = DataLoader(data, 64, num_workers = 4, pin_memory=True, 
                        prefetch_factor=64, persistent_workers=True)

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            loss = model.loss(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct * 100 / total}%")
    print(f"Loss: {test_loss / len(loader)}")


if __name__ == "__main__":
    main()