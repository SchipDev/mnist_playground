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

print(f"Beginning training for {NUM_EPOCHS} epochs")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_dl:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] finished: loss={running_loss/len(train_dl):04f}")

print("Training complete")
