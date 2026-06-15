import torch
import nansense

model = MyNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

session = nansense.start(model, epochs=50, phases={"train": len(train_loader)}, optimizer=optimizer, port=8080)

# Time travel: wrap the epoch loop so a UI jump can rewind and replay it.
restorer = session.training_restorer(cache_dir="models/latest")
while restorer.pending():
    with restorer:
        for epoch in restorer.epochs():
            for inputs, targets in session.batches(train_loader, phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()

session.close()
