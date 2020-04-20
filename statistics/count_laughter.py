import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from io import BufferedReader

detected_laughter_folder = '../laughter-detection/detected_train_lauthter/'
laughter_file = 'dia991_utt7/laugh_0.wav'

with open ('../DialogueRNN/DialogueRNN_features/MELD_features/MELD_features_raw.pkl', 'rb') as f:
    meld_features = pickle.load(f)
    video_sentences = meld_features[5] #video sentences
    emotion_labels = meld_features[2]#emotion labels:{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    sentiment_labels = meld_features[8] #sentiment labels: {'neutral': 0, 'positive': 1, 'negative': 2}

# change to list
sentence_lists = list(video_sentences.values())
emotion_label_lists = list(emotion_labels.values())
sentiment_label_lists = list(sentiment_labels.values())

# test
# print(sentence_lists[0])
# print(emotion_label_lists[0])





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