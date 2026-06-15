import lightning as L
from nansense.lightning import NansenseCallback, fit_with_time_travel

# `model="net"` is the attribute path to the network inside your LightningModule.
callback = NansenseCallback(port=8080, model="net")

# Time travel: fit_with_time_travel wraps a fresh Trainer for each rewind.
fit_with_time_travel(lambda: L.Trainer(max_epochs=50), module, callback=callback, datamodule=datamodule)
