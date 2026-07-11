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
        - restore_point
        - training_restorer
        - close
        - enabled
        - snapshot
        - live_position
        - mode
        - time_travel_status
        - lock

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

## Numerical debugging

::: nansense.DebugSettings

::: nansense.DebugError

::: nansense.LayerReport
