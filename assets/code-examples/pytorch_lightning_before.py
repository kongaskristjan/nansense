import lightning as L

L.Trainer(max_epochs=50).fit(module, datamodule)
