# Key Word Spotting Game

Master Thesis Project @TUGraz, 01 / 2022
Author: Christian Walter


## Project description

The aim of this project was to implement a Key-Word-Spotting (KWS) system with neural networks trained on the speech command dataset [1] and applied in a simple video game.
Several low computational Convolutional Neural Networks (CNN) in the fashion of [2] were evaluated.
A pre-training method using Generative Adversarial Networks (GAN) [3] was used to increase the accuracies.
Wavenets [4] were extended with a classification of speech commands.

![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/main_menu.png)

The gameplay video can be found in the folder "docs/gameplay.mp4"


## Technical Details

Operating System: Linux
Programming Language: Python >=3.8.x
Neural Network Framework: Pytorch >=1.7
Several packages must be installed in order to run the game. 
Try to run the program and determine the missing dependencies.
Further, a microphone is necessary to play the game.


## Usage

To play the game you have to open a terminal in the project path and type: 'python kws_main.py'


## Screenshots

![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/menu_option_help.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/menu_option_kws.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/menu_option_e.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/menu_option_device.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/level1-1.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/level1-2.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/level2-2.png)
![alt text](https://raw.githubusercontent.com/chrisworld/kws_game/master/docu/screenshots/level2-2.png)


## References

[1] Warden, P., “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition”, 2018.
[2] Sainath, T., "Convolutional neural networks for small-footprint keyword spotting", 2015.
[3] Goodfellow, I. J., “Generative Adversarial Networks”, 2014.
[4] Oord, A., "WaveNet: {A} Generative Model for Raw Audio", 2016.