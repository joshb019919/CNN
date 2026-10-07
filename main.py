from src.train import main as trainer
from src.test import main as tester


def main():
    print("Training Now...\n")
    trainer()
    print("Training Complete.")
    print("Testing Now...\n")
    tester()
    print("Testing Complete.")


if __name__ == "__main__":
    main()