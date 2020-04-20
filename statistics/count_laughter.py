import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from io import BufferedReader

detected_laughter_folder = '../laughter-detection/detected_train_lauthter/'
laughter_file = 'dia991_utt7/laugh_0.wav'

with open ('../DialogueRNN/DialogueRNN_features/MELD_features/MELD_features_raw.pkl', 'rb') as f:
    meld_features = pickle.load(f)
    print(meld_features[2]) #emotion labels:{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    print(meld_features[8]) #sentiment labels: {'neutral': 0, 'positive': 1, 'negative': 2}

# check if detected_train_lauthter_file in folder:
if os.path.isfile(detected_laughter_folder + laughter_file):
    print(True)
else:
    print(False)

# X = [np.random.rand(100) * 10]
# Y = [np.random.rand(100) * 10]
# # 散布図を描画する
# plt.scatter(X, Y)
# plt.show()