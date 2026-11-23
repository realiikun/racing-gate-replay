# Updating racing gate tracks when delayed detections arrive

Version 1.0.0 accompanies the manuscript of the same title. It contains the replay and scoring programs, six flights of frozen detector outputs and predicted tracks, the reference coordinates for the clip comparison, and the individual timing measurements.

## Reproduce the paper tables

From the repository root, run the following with Python 3.11.

```sh
python paper_tables.py
python verify.py
python -m pip install -r requirements.txt
python rescore.py
```

The first two commands use the standard library. The third installs dependencies for the analyses. The fourth reads the archived labels and frozen predictions and checks 24 scores per flight. All six recordings were checked before publication. For the primary corner criterion, the flight-equal reduction from waiting to alignment is 14.41 percentage points on flights 03a, 08a and 16a and 8.38 points on additional flights 06a, 12a and 18a. On the sampled clips the reduction is 27.34 points with the dataset reference and 17.79 with the image-review reference. The common-state analysis uses 268 windows. See the reader protocol for the reader's prior knowledge and the reference-freeze sequence.

## Replay a recording from camera frames

Download the autonomous recordings from [TII Racing](https://github.com/tii-racing/drone-racing-dataset). Set the environment variable RAM_AUTONOMOUS to the directory containing the flight folders. For example, in PowerShell:

```powershell
$env:RAM_AUTONOMOUS = 'D:/datasets/drone-racing-dataset/data/autonomous'
python replay_check.py flight-03a-ellipse --output runs/03a
```

Repeat with flight-08a-lemniscate, flight-16a-trackRATM, flight-06a-ellipse, flight-12a-lemniscate or flight-18a-trackRATM. Each run combines the original frames with cached detector outputs, then compares every predicted box, corner and visibility flag with the archive. All 23,050 frames matched before publication, with zero maximum coordinate discrepancy. The frozen detector outputs let readers run this check without the trained checkpoint.

## Files and protocols

| Paper result | Files |
| --- | --- |
| Primary and additional continuous replays | continuous.py, score_continuous.py, rescore.py, results/held_out/ |
| Clip reference comparison | blind_clips/compare.py, independent_reference.json and reader_protocol.md in blind_clips/ |
| Window sensitivity and paired analyses | sensitivity.py, score_same_windows.py, revision_statistics.py, inputs/ and anchor_outages_ms.csv |
| Delay and entry analyses | revision_delay.py, revision_delay_summary.py, revision_entry.py, revision_error.py |
| Timing | timing/benchmark.py, timing/frames.json, timing/detector_timing.json and hardware/software metadata |
| Detector training | training/manifest.json, training/rebuild_manifest_dataset.py and training/args.yaml |
| Figure 1 | figure1.py and the original RAM camera frames |

The training directory records the image split and settings. Hardware timing with the original model requires the matching checkpoint. Its SHA-256 is recorded in continuous.py. A newly trained model may yield different outputs. The 72 archived measurements describe the original workstation.

See [third-party notices](THIRD_PARTY_NOTICES.md) for the dataset credit and terms. Cite the versioned software release (v1.0.0) and the source dataset when using these results.
