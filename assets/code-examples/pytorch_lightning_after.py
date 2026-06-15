import lightning as L
from nansense.lightning import NansenseCallback, fit_with_time_travel

# `model="net"` is the attribute path to the network inside your LightningModule.
callback = NansenseCallback(port=8080, model="net")

# Time-travel: trainer factory enables restarting training at different epochs
make_trainer = lambda: L.Trainer(max_epochs=50)
fit_with_time_travel(make_trainer, module, datamodule=datamodule, callback=callback)
