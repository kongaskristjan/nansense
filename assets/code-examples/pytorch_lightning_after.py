import lightning as L
from nansense.lightning import NansenseCallback, fit_with_time_travel

# `model=` is the attribute path to the network inside your LightningModule.
# A live UI opens at http://localhost:8080.
callback = NansenseCallback(port=8080, model="net")

# Time travel: `fit_with_time_travel` wraps a stock Trainer so the UI's Time
# Travel button rewinds the fit to any epoch (a factory, because each jump
# needs a fresh Trainer).
fit_with_time_travel(lambda: L.Trainer(max_epochs=50), module, callback=callback, datamodule=datamodule)
