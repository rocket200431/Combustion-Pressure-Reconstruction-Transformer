import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, device,
                epochs=40, patience=8):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1:03d}] | Train {train_loss:.5f} | Val {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

def test_model(model, test_loader, device):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_loss += criterion(model(X), y).item()

    test_loss /= len(test_loader)
    print(f"Final Test Loss: {test_loss:.6f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders()

    model = PressureNet().to(device)

    train_model(model, train_loader, val_loader, device)
    test_model(model, test_loader, device)
