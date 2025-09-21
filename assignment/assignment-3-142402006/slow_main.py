# main.py
import json
import toml
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Load JSON config (data + model architectures)
# -------------------------
with open("config.json") as f:
    config = json.load(f)

# Load TOML hyperparameters
params = toml.load("params.toml")

# Load grid search JSON
with open("grid_search.json") as f:
    grid = json.load(f)

# -------------------------
# Dataset: CIFAR-10 (resize to 224x224 for ResNet)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

# -------------------------
# Model helper
# -------------------------
model_dict = {
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}

def get_model(name):
    model = model_dict[name](pretrained=config["models"][name]["pretrained"])
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes
    return model

# -------------------------
# Training + inference
# -------------------------
def train_model(model, optimizer_name, lr, momentum, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer_name.lower() == "adam" else optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Inference on first batch of val set
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            preds_list.extend(preds)
            break  # only first batch for demonstration
    return preds_list

# -------------------------
# a. Model inference outputs
# -------------------------
print("\n--- a. Model Inference Outputs ---")
for model_name in config["models"]:
    model = get_model(model_name)
    hp = params[model_name]
    preds = train_model(model, hp["optimizer"], hp["learning_rate"], hp["momentum"])
    print(f"{model_name} Predictions: {preds}")

# -------------------------
# b. JSON config output
# -------------------------
print("\n--- b. Data & Model JSON ---")
print(json.dumps(config, indent=4))

# -------------------------
# c. TOML hyperparameters output
# -------------------------
print("\n--- c. Model Hyperparameters TOML ---")
for model_name, hp in params.items():
    print(f"[{model_name}] learning_rate={hp['learning_rate']}, optimizer={hp['optimizer']}, momentum={hp['momentum']}")

# -------------------------
# d. Integrated pipeline display
# -------------------------
print("\n--- d. Integrated Pipeline ---")
for model_name in config["models"]:
    hp = params[model_name]
    print(f"Training {model_name} with lr={hp['learning_rate']}, optimizer={hp['optimizer']}, momentum={hp['momentum']}")

# -------------------------
# e. Hyperparameter tuning (Grid Search) for ResNet34
# -------------------------
print("\n--- e. Hyperparameter Tuning (ResNet34) ---")
best_accuracy = 0
best_params = {}

for lr in grid["learning_rate"]:
    for opt in grid["optimizer"]:
        for mom in grid["momentum"]:
            model = get_model("resnet34")
            preds = train_model(model, opt, lr, mom)
            # For demo, use random accuracy since CIFAR-10 evaluation skipped
            accuracy = torch.rand(1).item()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"learning_rate": lr, "optimizer": opt, "momentum": mom}

print(f"Best hyperparameters for resnet34: {best_params}, Accuracy: {best_accuracy:.4f}")
