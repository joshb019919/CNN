from cnn import CNN
import torch
import numpy as np
import pandas as pd


def main():
    cnn = CNN()
    cnn.train()
    df = pd.DataFrame(columns=["epoch", "accuracy", "loss"])

    total, correct = 0, 0
    epoch_total, epoch_correct = 0, 0
    train_loss = 0.0
    epoch_loss = 0.0

    for epoch in range(cnn.epochs):
        for images, labels in cnn.dataloader:
            images, labels = images.to(cnn.device), labels.to(cnn.device)
            outputs = cnn(images)
            loss = cnn.loss(outputs, labels)
            train_loss += loss.item()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            loss.backward()
            cnn.opt.step()
            cnn.opt.zero_grad()

            total += labels.size(0)
            epoch_total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_correct += (predicted == labels).sum().item()

        epoch_accuracy = float(epoch_correct * 100 / epoch_total)
        epoch_loss = float(epoch_loss / len(cnn.dataloader))

        df_size = len(df) if len(df) > 0 else 0

        df.loc[df_size] = [epoch+1, epoch_accuracy, epoch_loss]
        print(f"Epoch {int(epoch+1)} Accuracy:\t{epoch_accuracy}")
        print(f"Epoch {int(epoch+1)} Loss:\t\t{epoch_loss}")

        epoch_correct, epoch_loss, epoch_total = 0, 0, 0

    torch.save(cnn.state_dict(), "model.pth")
    df = df.astype({"epoch": np.int32, "accuracy": np.float64, "loss": np.float64})
    df.to_csv("metrics.csv")


if __name__ == "__main__":
    main()
