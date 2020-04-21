import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
import re

detected_folder = '../laughter-detection/detected_train_lauthter/' # checked folder if laughter is detected. it contains detected and also undeted folder
# laughter_file = 'dia991_utt7/'

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

laughter_file_path = glob.glob(detected_folder + '*/laugh_0.wav')
laughter_file = [os.path.basename(os.path.dirname(l)) for l in laughter_file_path]

# get detected laughter index
regex = re.compile('\d+')
laughter_index_str = [regex.findall(l) for lf in laughter_file for l in lf.splitlines()]

def functor(f, l):
  if isinstance(l,list):
    return [functor(f,i) for i in l]
  else:
    return f(l)

laughter_index= functor(int, laughter_index_str)
print(laughter_index)



# for l in laughter_index:
#     if 9 in l[0]:
#         print(l)
# print(laughter_index)
# for lf in laughter_file:
#     for l in lf.splitlines():
#       match = regex.findall(l)
#       print(match)

# laughter_file_index = {}
# for l in laughter_file_path:
#     print(os.path.basename(os.path.dirname(l)))
#     print(laughter_file)

# check if detected_train_lauthter_file in folder:
# if os.path.isfile(detected_laughter_folder + laughter_file + '/laugh_0.wav'):
#     print(laughter_file)
# else:
#     print(False)

# X = [np.random.rand(100) * 10]
# Y = [np.random.rand(100) * 10]
# # 散布図を描画する
# plt.scatter(X, Y)
# plt.show()