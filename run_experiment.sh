# activate conda environment
source activate muse
# Run main python script including 
# a) EEG streaming, b) event detection, c) audio stimulation, and
# d) saving data to .xdf file via LSL
python run_closedloop.py