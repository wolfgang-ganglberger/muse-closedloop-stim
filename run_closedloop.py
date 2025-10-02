""" Big picture:

Signal Acquisition: MuseOSCReceiver receives and buffers EEG data from the Muse headband.
EEG Streaming: EEG data is streamed to LSL via eeg_streamer, making it available for recording in LabRecorder.
Detection: DetectorThread analyzes EEG in real time, looking for events that should trigger a stimulus.
Stimulation & Marking: AudioWorker plays a sound and sends a marker to LSL when triggered.
    - The marker string can include the timestamp when the event was detected (e.g., 'stim_sent:pink:det_ts=123.456').
    - The LSL marker timestamp records the precise time when the marker was sent (i.e., stimulus onset).
Data Saving: LabRecorder records both the EEG stream and the marker stream into a .xdf file for later analysis.

What is saved in the .xdf file?
EEG Stream: Continuous EEG data at 256 Hz.      
StimMarkers Stream: Timestamps and labels for each stimulus event (e.g., 'stim_sent:pink:det_ts=...').

Summary
This script sets up a closed-loop experiment where EEG is monitored in real time, stimuli are delivered based on brain activity, and all relevant data (EEG + event markers) are saved for offline analysis.
Code is setup modular, easy to run, and each component is responsible for a clear part of the experiment pipeline.

"""

import queue
import time
from osc_receiver import MuseOSCReceiver
from lsl_interface import start_eeg_lsl_stream
from event_detection import DetectorThread
from stimulus_player import AudioWorker
import random

def main():
    rx = MuseOSCReceiver(port=7000)                 # Create and start the Muse OSC receiver
    rx.start()
    
    eeg_streamer = start_eeg_lsl_stream(rx)         # Starts a background thread that reads EEG data from rx and pushes it to an LSL stream ("MuseEEG"), making it visible to LabRecorder.
    
    stim_q = queue.Queue()                          # Creates a thread-safe queue for passing stimulus trigger timestamps from the detector to the audio worker.
    detector = DetectorThread(rx, stim_q)
    detector.start()                                # Creates and starts the detector thread, which:
                                                    # Continuously reads EEG data from rx.
                                                    # Filters the signal and checks for events according to rules/code.
                                                    # When a stimulus should be triggered, puts a timestamp into stim_q.

    stim_type = 'morlet'  # 'pink' or 'morlet', depending on your experimental design
    stim_rms = 0.05  # RMS amplitude of the stimulus sound. Default is 0.05
    stim_dur = 0.1  # Duration of the stimulus sound in seconds
    audio = AudioWorker(stim_type=stim_type, stim_rms=stim_rms, stim_dur=stim_dur)  
                                                    # Creates the audio worker, which:
                                                    # Waits for timestamps in cmd_q (to be filled from stim_q).
                                                    # Plays a pink noise burst when triggered.
                                                    # Sends an LSL marker ("stim_sent") at the moment of stimulus onset.

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
        audio.stream.stop(); 
        audio.stream.close()

if __name__ == "__main__":
    main()