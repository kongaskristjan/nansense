import torch

model = MyNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_loader, val_loader = create_my_dataloaders()

for epoch in range(50):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    # Validation loop ...
