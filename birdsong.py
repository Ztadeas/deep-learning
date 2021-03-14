import librosa
import numpy as np
import os

dir_path = "C:\\Users\\Tadeas\\Downloads\\birdsongbig"

rozdeleni = 22050*3


def preprocess():
  mfccs = []
  labels = []
  
  for v, i in enumerate(os.listdir(dir_path)):
    near_full_path = os.path.join(dir_path, i)
    for x in os.listdir(near_full_path):
      full_path = os.path.join(near_full_path, x)
      signal, sr = librosa.load(full_path)
      for p in range(signal.shape[0] // rozdeleni):
        new_signal = signal[rozdeleni*p: rozdeleni*(p+1)]
        mfcc = librosa.feature.mfcc(new_signal, n_mfcc=13)
        mfccs.append(mfcc)
        labels.append(v)
    print(f"{v}: Done")

    mfccs = np.asarray(mfccs)

  return mfccs, labels        


mfccs, labels = preprocess()

print(mfccs.shape)






