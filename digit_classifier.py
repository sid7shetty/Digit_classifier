import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

# Define the CNN architecture
class DigitClassifier(nn.Module):
    def __init__(self, use_dropout=True, use_batch_norm=True):
        super(DigitClassifier, self).__init__()
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        # Fully connected layers
        self.fc1 = nn.Linear(7*7*64, 128)
        self.bn3 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten for fully connected layers
        x = x.view(-1, 7*7*64)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Function to load the MNIST dataset
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download and load test data
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train the model
def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return train_loss / len(train_loader), correct / total

# Function to test the model
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2%})\n')
    
    return test_loss, accuracy

# Function to visualize some predictions
def visualize_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    
    # Get a batch of test data
    data, targets = next(iter(test_loader))
    data, targets = data[:num_samples].to(device), targets[:num_samples].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.argmax(dim=1)
    
    # Plot images with predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        # Display image
        axes[i].imshow(data[i].squeeze().cpu().numpy(), cmap='gray')
        
        # Set title with prediction and true label
        color = 'green' if predictions[i] == targets[i] else 'red'
        axes[i].set_title(f'Pred: {predictions[i].item()}\nTrue: {targets[i].item()}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Digit Classifier')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-dropout', action='store_true', default=False, help='disables dropout')
    parser.add_argument('--no-batch-norm', action='store_true', default=False, help='disables batch normalization')
    parser.add_argument('--save-model', action='store_true', default=False, help='save the trained model')
    args = parser.parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_data(batch_size=args.batch_size)
    
    # Initialize model
    model = DigitClassifier(
        use_dropout=not args.no_dropout, 
        use_batch_norm=not args.no_batch_norm
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        test_loss, test_acc = test(model, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Visualize some predictions
    visualize_predictions(model, test_loader, device)
    
    # Plot training and testing metrics
    epochs = range(1, args.epochs + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    # Save the model if requested
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Model saved as mnist_cnn.pt")

if __name__ == '__main__':
    main()