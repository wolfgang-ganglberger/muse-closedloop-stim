#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate muse
python live_dashboard.py