""" Big picture:

Signal Acquisition: MuseOSCReceiver receives and buffers EEG data from the Muse headband.
EEG Streaming: EEG data is streamed to LSL via eeg_streamer, making it available for recording in LabRecorder.
Detection: DetectorThread analyzes EEG in real time, looking for events that should trigger a stimulus.
Stimulation & Marking: AudioWorker plays a sound and sends a marker to LSL when triggered.
    - The marker string can include the timestamp when the event was detected (e.g., 'stim_sent:pink:det_ts=123.456').
    - The LSL marker timestamp records the precise time when the marker was sent (i.e., stimulus onset).
Data Saving: 
1) Start LabRecorder to record the EEG and marker streams to a .xdf file.
2) LabRecorder records both the EEG stream and the marker stream into a .xdf file for later analysis.

What is saved in the .xdf file?
EEG Stream: Continuous EEG data at 256 Hz.
MuseMetrics Stream: Continuous Muse metrics data (30 channels).
StimMarkers Stream: Timestamps and labels for each stimulus event (e.g., 'stim_sent:pink:det_ts=...').

Summary
This script sets up a closed-loop experiment where EEG is monitored in real time, stimuli are delivered based on brain activity, and all relevant data (EEG + event markers) are saved for offline analysis.
Code is setup modular, easy to run, and each component is responsible for a clear part of the experiment pipeline.

"""

print("\033[33mTo save data, start LabRecorder and record the EEG and marker streams to a .xdf file.\033[0m")
import queue
import time
import random
import sys
import subprocess
sys.path.append("./closedloop-utils")
from osc_receiver import MuseOSCReceiver
from lsl_interface import start_eeg_lsl_stream, start_metrics_lsl_stream
from event_detection import DetectorThread
from stimulus_player import AudioWorker

def main():
    system_volume_pct = 75  # macOS only; set to None to leave system volume unchanged
    if system_volume_pct is not None and sys.platform == "darwin":
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {float(system_volume_pct)}"],
                check=True,
            )
        except Exception as exc:
            print(f"Warning: could not set system volume ({exc}).")

    rx = MuseOSCReceiver(port=7000)                 # Create and start the Muse OSC receiver
    rx.start()
    
    eeg_streamer = start_eeg_lsl_stream(rx)         # Starts a background thread that reads EEG data from rx and pushes it to an LSL stream ("MuseEEG"), making it visible to LabRecorder.
    metrics_streamer = start_metrics_lsl_stream(rx) # Streams Muse metrics (including sleep stage probabilities) to LSL for .xdf recording.
    
    stim_q = queue.Queue()                          # Creates a thread-safe queue for passing stimulus trigger timestamps from the detector to the audio worker.
    mode = 'slow-wave-v1'                        # either "simple-treshold" or "slow-wave-v1"
    detector = DetectorThread(
        rx,
        stim_q,
        mode=mode,
        state_lsl_name="StimState",
        slow_wave_params={
            "device_latency_s": 0.2,
            "global_on_s": 12.0,
            "global_off_s": 10.0,
            "refractory_s": 6.0,
            "use_stage_gate": True,
            "stage_prob_min_n2": 0.75,
            "stage_prob_sum_n2_n3": 0.90,
            "stage_prob_n2_min_for_sum": 0.30,
            "use_signal_quality_gate": True,
            "signal_quality_min": 0.60,
        },
    )
    detector.start()                                # Creates and starts the detector thread, which:
                                                    # Continuously reads EEG data from rx.
                                                    # Filters the signal and checks for events according to rules/code.
                                                    # When a stimulus should be triggered, puts a timestamp into stim_q.

    stim_type = 'pink'  # 'pink' or 'morlet', depending on your experimental design
    stim_rms = 0.20  # RMS amplitude of the stimulus sound. Default is 0.05
    stim_peak = 0.20  # Peak cap or target (full-scale). Set to None to disable.
    stim_peak_normalize = False  # True = always normalize to stim_peak; False = only limit if exceeded.
    stim_dur = 0.050  # Duration of the stimulus sound in seconds
    audio = AudioWorker(
        stim_type=stim_type,
        stim_rms=stim_rms,
        stim_dur=stim_dur,
        stim_peak=stim_peak,
        stim_peak_normalize=stim_peak_normalize,
    )
                                                    # Creates the audio worker, which:
                                                    # Waits for timestamps in cmd_q (to be filled from stim_q).
                                                    # Plays a pink noise burst when triggered.
                                                    # Sends an LSL marker ("stim_sent") at the moment of stimulus onset.
    for _ in range(10):
        audio.trigger()
        time.sleep(stim_dur + 2)

    print("Experiment running. Ctrl-C to quit.")
    
    try:
        while True:
            # Main loop:
            # Waits for a stimulus trigger timestamp from the detector (stim_q.get()).
            # Forwards it to the audio worker (audio.cmd_q.put(fire_t)), which then plays the sound and sends the marker.
            try:
                fire_t = stim_q.get()
                audio.trigger(fire_t)
                
                ## ---
                ## TEST PURPOSES ONLY, RANDOM TRIGGERS. COMMENT 'audio_trigger(fire_t)' above to not use real events.
                # i = 0 
                # if i == 0:
                #     last_trigger_time = time.time()
                # i += 1
                # trigger_interval = random.uniform(4, 6)  # random interval between 4 and 6 seconds
                # current_time = time.time()
                # if current_time - last_trigger_time >= trigger_interval:
                #     fire_t = current_time  # use current time as the detection timestamp
                #     audio.trigger(fire_t)
                #     last_trigger_time = current_time
                #     trigger_interval = random.uniform(4, 6)  # set next random interval
                # time.sleep(0.01)  # avoid busy waiting
                ## --- END RANDOM TRIGGERS


            except queue.Empty:
                pass  # Allows loop to continue and check for KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping (kill terminal if hangs)...")     # Stops the OSC receiver, EEG LSL streamer, and audio stream.
        rx.stop_gracefully()
        eeg_streamer.stop()
        metrics_streamer.stop()
        audio.stream.stop(); 
        audio.stream.close()

if __name__ == "__main__":
    main()
