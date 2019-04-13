import librosa
import pickle
from specgrams_helper import *

sampleRate = 22050
helper = SpecgramsHelper(2*sampleRate, tuple([192, 512]), 0.75, sampleRate, 1)


def writeWav(spectogram, filename = "trial.wav"):
	global helper, sampleRate
	outtrue = helper.melspecgrams_to_waves(spectogram)
	with tf.Session() as sess:
		outtrue = outtrue.eval()
	librosa.output.write_wav(filename, outtrue[0], sampleRate)

data = pickle.load(open("../0.pkl", 'rb'))

x = 10
currentSample = data[x:x + 1]
writeWav(currentSample, filename = "original.wav")
currentSample[0, :, :, 1] = data[:, :, :, 1].mean(axis = 0)
writeWav(currentSample, filename = "phase_mean.wav")
currentSample[0, :, :, 1] = 0
writeWav(currentSample, filename = "phase_zero.wav")


