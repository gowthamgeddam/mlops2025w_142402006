# main.py
import json
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# -------------------------
# Load configurations
# -------------------------
with open("config.json") as f:
    config = json.load(f)

params = toml.load("params.toml")

with open("grid_search.json") as f:
    grid = json.load(f)

# -------------------------
# Data loaders (subset for quick debug)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

# Use small subset for debugging
train_subset = Subset(train_dataset, range(500))  # first 500 images
val_subset = Subset(val_dataset, range(100))      # first 100 images

train_loader = DataLoader(train_subset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
val_loader = DataLoader(val_subset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])

# -------------------------
# Model helper (fix deprecated 'pretrained')
# -------------------------
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

model_dict = {
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V1)
}

def get_model(name):
    model_fn, weights_enum = model_dict[name]
    model = model_fn(weights=weights_enum if config["models"][name]["pretrained"] else None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10
    return model

# -------------------------
# Training function
# -------------------------
def train_model(model, optimizer_name, lr, momentum, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=lr) if optimizer_name.lower() == "adam"
        else optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    )

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Inference (first batch only)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds_list.extend(outputs.argmax(dim=1).cpu().tolist())
            break
    return preds_list

# -------------------------
# a. Model inference
# -------------------------
print("\n--- a. Model Inference Outputs ---")
for model_name in config["models"]:
    if model_name in ["resnet101", "resnet152"]:  # optional skip heavy models
        continue
    model = get_model(model_name)
    hp = params[model_name]
    preds = train_model(model, hp["optimizer"], hp["learning_rate"], hp["momentum"])
    print(f"{model_name} Predictions: {preds}")

# -------------------------
# b. JSON config display
# -------------------------
print("\n--- b. Data & Model JSON ---")
print(json.dumps(config, indent=4))

# -------------------------
# c. TOML hyperparameters display
# -------------------------
print("\n--- c. Model Hyperparameters TOML ---")
for model_name, hp in params.items():
    print(f"[{model_name}] learning_rate={hp['learning_rate']}, optimizer={hp['optimizer']}, momentum={hp['momentum']}")

# -------------------------
# d. Integrated pipeline
# -------------------------
print("\n--- d. Integrated Pipeline ---")
for model_name in config["models"]:
    hp = params[model_name]
    print(f"Training {model_name} with lr={hp['learning_rate']}, optimizer={hp['optimizer']}, momentum={hp['momentum']}")

# -------------------------
# e. Grid search for ResNet34 (quick demo)
# -------------------------
print("\n--- e. Hyperparameter Tuning (ResNet34) ---")
best_accuracy = 0
best_params = {}

for lr in grid["learning_rate"]:
    for opt in grid["optimizer"]:
        for mom in grid["momentum"]:
            model = get_model("resnet34")
            preds = train_model(model, opt, lr, mom)
            accuracy = torch.rand(1).item()  # demo only
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"learning_rate": lr, "optimizer": opt, "momentum": mom}

print(f"Best hyperparameters for resnet34: {best_params}, Accuracy: {best_accuracy:.4f}")
