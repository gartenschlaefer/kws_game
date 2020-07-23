"""
Microphone playground
adapted from the tutorials of the corresponding packages 
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# sound stuff
import sounddevice as sd
import pyaudio
import wave

# system stuff
import argparse
import queue
import sys


def callback_pa(in_data, frame_count, time_info, status):
  """
  PyAudio callback
  """

  # data setup
  data = np.zeros((chunk, channels))
  data[:, 1] = np.fromstring(in_data, dtype=np.float32)

  print("indata: ", data.shape)

  # queue
  q.put(data[::downsample])

  return (data, pyaudio.paContinue)


def callback_sd(indata, frames, time, status):
  """
  Input Stream Callback
  """

  if status:
    print(status)

  print("data: ", indata.shape)

  # put
  q.put(indata[::downsample])


def pa_test(p, chunk, channels, fs, sample_format=pyaudio.paInt16, seconds=3, filename='output.wav'):
  """
  pyaudio
  """

  print('Recording')

  stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
  print('opened stream')

  frames = []  # Initialize array to store frames

  # Store data in chunks for 3 seconds
  for i in range(0, int(fs / chunk * seconds)):
    print('steam: ', i)
    data = stream.read(chunk, exception_on_overflow = False)
    frames.append(data)

  # Stop and close the stream 
  stream.stop_stream()
  stream.close()

  # Terminate the PortAudio interface
  p.terminate()
  print('Finished recording')

  # write to wav file
  write_to_wav_file(filename, channels, p, fs, frames, sample_format)


def write_to_wav_file(filename, channels, p, fs, frames, sample_format):
  """
  write to wav file
  """

  # Save the recorded data as a WAV file
  wf = wave.open(filename, 'wb')
  wf.setnchannels(channels)
  wf.setsampwidth(p.get_sample_size(sample_format))
  wf.setframerate(fs)
  wf.writeframes(b''.join(frames))
  wf.close()


def update_plot(frame):
  """
  update plot each frame
  """
  
  global plotdata

  # run forever
  while True:

    # get data
    try:
      data = q.get_nowait()

    # no data
    except queue.Empty:
      break

    # shifting
    shift = len(data)
    plotdata = np.roll(plotdata, -shift, axis=0)

    # new data
    plotdata[-shift:, :] = data

    # set new data
    for column, line in enumerate(lines):
      line.set_ydata(plotdata[:, column])

  return lines


def plot_mic_init(channels):
  """
  plot of mic signal init
  """

  global plotdata

  fig, ax = plt.subplots()
  lines = ax.plot(plotdata)

  if channels > 1:
    ax.legend(['channel {}'.format(c) for c in range(channels)], loc='lower left', ncol=channels)

  ax.axis((0, len(plotdata), -1, 1))
  ax.set_yticks([0])
  ax.yaxis.grid(True)
  ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  fig.tight_layout(pad=0)

  return fig, ax, lines


if __name__ == '__main__':
  """
  main of audio testing
  """

  # params
  fs, chunk, channels, downsample, window, interval = 44100, 1024, 2, 10, 200, 30


  # --
  # pyaudio

  # instance
  #p = pyaudio.PyAudio()

  # recording with pyaudio
  #pa_test(chunk, channels, fs)


  # --
  # plot raw waveform

  # init plot data
  plotdata = np.zeros((int(window * fs / (1000 * downsample)), channels))

  # queue for audio samples
  q = queue.Queue()

  # plot data init
  fig, ax, lines = plot_mic_init(channels)

  # setup stream sounddevice
  stream = sd.InputStream(samplerate=fs, blocksize=chunk, channels=channels, callback=callback_sd)

  # setup stream pyaudio
  #stream = p.open(format=pyaudio.paInt16, channels=channels, rate=fs, output=False, input=True, stream_callback=callback_pa).start_stream()

  # animation of waveform
  ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)

  # stream and update
  with stream:
    plt.show()
