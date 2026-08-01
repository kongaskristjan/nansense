# API reference

The public surface of the `nansense` package. Everything here is importable as `nansense.<name>`, except the Lightning integration, which lives in `nansense.lightning` (available when `lightning` is installed).

## Entry points

::: nansense.start

::: nansense.serve

## The session

::: nansense.Session
    options:
      members:
        - batches
        - epochs
        - set_schedule
        - restore_point
        - training_restorer
        - close
        - enabled
        - snapshot
        - live_position
        - mode
        - time_travel_status
        - lock
        - freeze_moment
        - set_patch_layers
        - park
        - watch_metric
        - watch_layer_tensor
        - watch_weight_tensor
        - watch_metrics_snapshot
        - instrument_errors

## PyTorch Lightning integration

::: nansense.lightning.NansenseCallback
    options:
      members:
        - session

::: nansense.lightning.fit_with_time_travel

## Positions and snapshots

::: nansense.BatchPosition

::: nansense.Mode

::: nansense.Schedule
    options:
      members:
        - epochs
        - phases
        - phase_order
        - phase_count

::: nansense.BatchSnapshot

## Watch statistics

::: nansense.StatsScope

::: nansense.WatchSnapshot

::: nansense.LayerStatsSnapshot

::: nansense.TensorStatsSnapshot

## Custom instruments

See the [Custom metrics & tensors guide](instruments.md) for usage; the registration decorators live on the session (`Session.watch_metric`, `Session.watch_layer_tensor`, `Session.watch_weight_tensor` above).

::: nansense.LayerContext

::: nansense.WeightContext

::: nansense.MetricsSnapshot

::: nansense.MetricSeries

## Time travel

::: nansense.TrainingRestorer
    options:
      members:
        - start_epoch
        - finished
        - epochs
        - pending
        - iter_epochs
        - epoch_guard
        - save_epoch_start

::: nansense.TimeTravelStatus

::: nansense.TimeTravelJump

::: nansense.TimeTravelError

## Frozen moments

::: nansense.load_moment

::: nansense.MomentError

## Numerical debugging

::: nansense.DebugSettings

::: nansense.DebugError

::: nansense.LayerReport
