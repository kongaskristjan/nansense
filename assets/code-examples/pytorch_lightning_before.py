import lightning as L

trainer = L.Trainer(max_epochs=50)
trainer.fit(module, datamodule=datamodule)
