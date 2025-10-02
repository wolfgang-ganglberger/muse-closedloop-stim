# Slow-Wave Detection & Targeting — `_so_detection_dev`

A compact offline pipeline for **detecting sleep slow oscillations (SOs)** and **evaluating auditory-stim targeting**, adapted from (and extending) **Ngo et al., Neuron, 2013**.

---

## Quick Start

Run the full pipeline on a sample Muse CSV:

```bash
python run_sw_detection_pipeline_sample.py
```

In that file, set:

- `SAMPLE_DATA_DIR` → folder with your Muse CSV(s)  
- `SAMPLE_FILE_NAME` → optional specific filename (or `None` to use the first found)

This will:
- load data
- create a **virtual frontal** channel from Muse temporals
- detect SOs (Ngo-style)
- estimate **up-state delay** (neg→pos latency)
- compute **spindle coupling** (9–12 & 12–15 Hz RMS)
- assess **targeting performance** (fixed & adaptive delays)
- plot EEG, events, timing-error histograms, and phase polar charts

---

## What’s Here (and Why)

| File | What it does | Why it matters |
|---|---|---|
| `run_sw_detection_pipeline.py` | Orchestrates the full flow: load → virtual frontal → SO detect → targeting eval → plots. | One command to analyze a recording end-to-end. |
| `sw_detection_utils.py` | Core SO detection: pos→neg zero-crossings, **0.9–2.0 s** cycles, **1.25×** amplitude gates, train grouping, up-state delay, spindle RMS. | Faithfully mirrors Ngo’s offline detector, adapted for headband data. |
| `sw_assess_target_performance.py` | Targeting QA: fixed delay and adaptive **EMA/Kalman** delay models, timing/phase metrics (PLV, Rayleigh), diagnostics plots. | Tells you how well clicks would land on the up-state. |
| `run_sw_detection_pipeline_sample.py` | Minimal example wiring it all together. | “Change two paths, press run.” |

---

## Method Highlights

- **Virtual frontal channel** from Muse temporals (noise-weighted), then SO-band filtering.
- **SOs** via pos→neg zero-crossings, duration **0.9–2.0 s**, amplitude gates **(neg & peak-to-peak ≥ 1.25× mean)**.
- **Up-state delay**: subject-specific (median neg→pos latency) used for coupling windows and targeting.
- **Spindle coupling**: RMS envelopes (9–12, 12–15 Hz), baseline-corrected around predicted up-states.
- **SO trains**: grouping via inter-peak interval ≤ 2 s.
- **Targeting metrics**: timing error (ms), hit-rates (≤25/50/100/250 ms), **phase-locking value**, Rayleigh p, plus adaptive EMA/Kalman delay tracking.

---

## Offline vs Online

- ✅ **Offline**: Complete detection/characterization + targeting assessment.
- ⚙️ **Online (partial)**: Adaptive delay trackers are ready.

---

## Reference

Ngo HVV et al. *Driving sleep slow oscillations by auditory closed-loop stimulation — a self-limiting process.* **Neuron**, 2013.
