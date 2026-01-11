import matplotlib.pyplot as plt
import torch

# Load best model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

num_plots = 5

indices = torch.linspace(
    0, len(test_loader.dataset) - 1, steps=num_plots
).long()

for idx in indices:
    idx = idx.item()

    X, y = test_loader.dataset[idx]
    X = X.unsqueeze(0).to(device)   # (1, 100)
    y = y.unsqueeze(0).to(device)   # (1, 10)

    with torch.no_grad():
        pred = model(X)

    plt.figure(figsize=(5, 3))
    plt.plot(pred[0].cpu().numpy(), marker="o")
    plt.plot(y[0].cpu().numpy(), marker="o")
    plt.legend(["Prediction", "True"])
    plt.title(f"Test window index {idx}")
    plt.xlabel("Index inside masked window")
    plt.ylabel("Scaled pressure")
    plt.grid(alpha=0.3)
    plt.show()
