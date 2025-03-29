import torch
from torchvision import transforms

from core.dataset import *
from core.models import SimpleClassifier

NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {DEVICE}")

# Download MNIST dataset if it doesnt already exist locally
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data, test_data = download_mnist("./data", transform)
print(f"Train Data: {train_data}")
print(f"Test Data: {test_data}")

# Create data loaders
print("Creating train and test  data loaders...")
train_dl, test_dl = create_dataloaders(train_data, test_data)
print("Done")

print("Initializing model...")
model = SimpleClassifier().to(DEVICE)
print("Done")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def validate_model(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    return accuracy



print(f"Beginning training for {NUM_EPOCHS} epochs")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    for images, labels in train_dl:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_labels = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

    train_accuracy = (correct / total) * 100
    val_accuracy = validate_model(test_dl, model, DEVICE)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] finished: training accuracy={train_accuracy:.4f}, validation accuracy={val_accuracy:.4f}, loss={running_loss/len(train_dl):.4f}")

print("Training complete")
