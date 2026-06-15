import torch
import nansense

model = MyNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_loader, val_loader = create_my_dataloaders()

phases = {"train": len(train_loader), "val": len(val_loader)}
session = nansense.start(model, epochs=50, phases=phases, optimizer=optimizer, port=8080)

# restorer while/with time-travel loop: wrap the epoch loop so the UI jump can rewind and replay it.
restorer = session.training_restorer(cache_dir="models/latest")
while restorer.pending():
    with restorer:
        for epoch in restorer.epochs():
            for inputs, targets in session.batches(train_loader, phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()

            # Validation loop ... (use phase="val")

session.close()
