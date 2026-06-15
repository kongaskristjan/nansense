import torch
import nansense

model = MyNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# One call serves a live UI at http://localhost:8080.
session = nansense.start(
    model,
    epochs=50,
    phases={"train": len(train_loader), "val": len(val_loader)},
    optimizer=optimizer,   # optional: weights page shows optimizer state + live LR
    scheduler=scheduler,   # optional: time travel restores the LR schedule
    port=8080,
)

# Time travel: wrap the epoch loop so a UI jump rewinds and replays it.
restorer = session.training_restorer(cache_dir="models/latest")
while restorer.pending():
    with restorer:
        for epoch in restorer.epochs():
            train_one_epoch(model, session.batches(train_loader, phase="train", epoch=epoch), optimizer, criterion)
            scheduler.step()

session.close()   # UI keeps serving the final snapshot
