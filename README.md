# Key-Word Spotting Game

from Christian Walter
Date: 01.09.2020


## Project description

The aim of this project was to evaluate a Key-Word-Spotting (KWS) system trained with the speech command data set from Warden (2018) in a simple Computer Game.


## Technical Details

Python 3.7 was used for coding and testing.
Several packages must by installed to run the game, try and see what is missing.
Further a microphone is necessary to play the game


## Usage

The project contains a fundamental approach for developing a KWS system.
This stretches from extracting features from the dataset, to speech commands used in the game.

To play the game open a terminal and type:

python kws_main.py

But note that the microphone parameters need to be changed to your own setup, this can be done in line 39:

mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier, energy_thres=1e-4, device=7)

where the 'device' parameter specifies the input device. 
Choose a number from the device list printed in the terminal.
Also the 'energy_thres' might need some fine tuning according to your microphone.









