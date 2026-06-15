import torch

model = MyNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    train_one_epoch(model, train_loader, optimizer, criterion)
    scheduler.step()
