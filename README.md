# Training

## Installation

Had some errors with python 3.10, so versions < 3.10 recommended.

## Run

Run programs from directory *src/*

`main.py` not yet implemented
Run `main.py` as:
```python3 main.py [-c config.yaml]```
and `log.py` as:
```python3 log.py strong_log.csv [-c config.yaml] [-a]```

Config `[-c]` can be provided as a yaml-file. Its items should correspond to attributes in config object **Exercise** in *src/config/exercises.py*, e.g.
```
squat:
  one_rm: 220.
  min_weight: 140.
  ...: ...
bench press:
  one_rm: 170.
  min_weight: 100.
  ...: ...
deadlift:
  one_rm: 245.
  min_weight: 180.
  ...: ...
```

Argument `[-a]` will tell the program to provide plots of the data.
