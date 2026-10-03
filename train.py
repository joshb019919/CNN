from cnn import CNN
import torch


def main():
    cnn = CNN()
    cnn.train()
    for epoch in range(cnn.epochs):
        for images, labels in cnn.dataloader:
            images, labels = images.to(cnn.device), labels.to(cnn.device)
            outputs = cnn(images)
            loss = cnn.loss(outputs, labels)
            loss.backward()
            cnn.opt.step()
            cnn.opt.zero_grad()

    torch.save(cnn.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
