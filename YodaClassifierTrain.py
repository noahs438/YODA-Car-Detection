import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt

from YodaDataset import YodaDataset
from YodaModel import YodaModel


def train(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs, device):
    train_losses = []
    test_losses = []
    train_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss = 0.0

        # Implement early stopping
        best_loss = float('inf')
        epochs_no_improve = 0
        early_stop_patience = 5

        # Set the model to training mode
        model.train()

        # Iterate over the training data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero out the gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(images)
            loss = criterion(output, labels)
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()

            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)
        train_losses.append(average_train_loss)
        scheduler.step(average_train_loss)

        # Set the model to evaluation mode to validate the classifier's predictions
        model.eval()
        with torch.no_grad():
            # Keep track of the number of correct predictions
            num_correct = 0
            total_predictions = 0
            test_loss = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                # For checking the accuracy
                _, prediction = torch.max(output, 1)
                total_predictions += labels.size(0)
                # Number of correct predictions
                num_correct += (prediction == labels).sum().item()

                loss = criterion(output, labels)
                test_loss += loss.item()

            average_test_loss = test_loss / len(test_loader)
            test_losses.append(average_test_loss)

            # Calculate the accuracy
            accuracy = num_correct / total_predictions

            epoch_end = time.time()
            # Print the time in minutes
            print('Epoch: {} |\tTraining Loss: {:.6f} |\tTesting Loss: {:.6f} |\tAccuracy: {:.6f} |\tTime: {:.3}'.format(
                epoch + 1, average_train_loss, average_test_loss, accuracy, (epoch_end - epoch_start)/60.0))

            if test_loss < best_loss:
                best_loss = test_loss
                epochs_no_improve = 0
                # Save the model if it's the best so far
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stop_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

    train_end = time.time()
    print('Total training time: {:.3} minutes'.format((train_end - train_start)/60.0))

    return train_losses, test_losses


def evaluate(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        # Keep track of the number of correct predictions
        num_correct = 0
        total_predictions = 0
        test_loss = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            # For checking the accuracy
            _, prediction = torch.max(output, 1)
            total_predictions += labels.size(0)
            # Number of correct predictions
            num_correct += (prediction == labels).sum().item()

            loss = criterion(output, labels)
            test_loss += loss.item()

        average_test_loss = test_loss / len(test_loader)

        # Calculate the accuracy
        accuracy = num_correct / total_predictions

        print('Testing Loss: {:.6f} |\tAccuracy: {:.6f}'.format(average_test_loss, accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YodaNet')
    parser.add_argument('-b', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
    # Defaulting to 40 epochs given the information in the lab handout
    parser.add_argument('-e', type=int, default=40, help='number of epochs')
    parser.add_argument('-s', type=str, required=True, help='path to save model')
    parser.add_argument('-p', type=str, required=True, help='path to save loss plot')
    parser.add_argument('-m', type=str, required=True, help='mode (train/test)')
    args = parser.parse_args()

    # Define transforms
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Resize((150, 150)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load in the training dataset
    train_dataset = YodaDataset(labels_path='./data/Kitti8_ROIs/train/labels.txt', root_dir='./data/Kitti8_ROIs/train',
                                transform=img_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=4)

    # Load in the testing dataset
    test_dataset = YodaDataset(labels_path='./data/Kitti8_ROIs/test/labels.txt', root_dir='./data/Kitti8_ROIs/test',
                               transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=True, num_workers=4)

    print("Datasets loaded successfully")

    # Either car or no car - so 2 classes
    num_classes = 2
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model
    model = YodaModel(num_classes=num_classes, weights=ResNet18_Weights.DEFAULT).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    if args.m == 'train':
        train_losses, test_losses = train(model, criterion, optimizer, scheduler, train_loader, test_loader, args.e,
                                          device)
        torch.save(model.state_dict(), args.s)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.p)
        plt.show()

    elif args.m == 'test':
        model.load_state_dict(torch.load(args.s))
        evaluate(model, test_loader, criterion, device)