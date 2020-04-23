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
    video_utterances = meld_features[5]
    emotion_labels = meld_features[2]#emotion labels:{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    sentiment_labels = meld_features[8] #sentiment labels: {'neutral': 0, 'positive': 1, 'negative': 2}

def functor(f, l):
  if isinstance(l,list):
    return [functor(f,i) for i in l]
  else:
    return f(l)

# change to list
utterance_lists = list(video_utterances.values())
emotion_label_lists = list(emotion_labels.values())
sentiment_label_lists = list(sentiment_labels.values())
sentiment_label_lists_np = np.array(sentiment_label_lists)

# test
# print(utterance_lists[0])
# print(emotion_label_lists[0])

laughter_file_path = glob.glob(detected_folder + '*/laugh_0.wav')
laughter_file = [os.path.basename(os.path.dirname(l)) for l in laughter_file_path]
laughter_file_index = [l.replace('dia', '').replace('utt','').split('_', 1) for l in laughter_file]
laughter_file_index = functor(int, laughter_file_index)  # laughter_file_index = [utterance_index, sentence_index in utterance]

import pdb;pdb.set_trace()

# get detected laughter index
regex = re.compile('\d+')
laughter_index_str = [regex.findall(l) for lf in laughter_file for l in lf.splitlines()]


laughter_index= functor(int, laughter_index_str)

laughter_index_np = np.array(laughter_index)

# print(sentiment_label_lists[0])
sentiment_index2 = []
for l in laughter_index:
    sentiment_index = sentiment_label_lists[l[0]-1]
    try:
        sentiment_index2.append(sentiment_index[l[1]])
    except IndexError:
        continue

emotion_index2 = []
for l in laughter_index:
    emotion_index = emotion_label_lists[l[0]-1]
    try:
        emotion_index2.append(emotion_index[l[1]])
    except IndexError:
        continue

# print(laughter_index[0][0]-1)
# print(sentiment_index2)
# print(len(sentiment_index2))

# neutral = sentiment_index2.count(0)
# positive = sentiment_index2.count(1)
# negative = sentiment_index2.count(2)

# print(neutral, positive, negative)
# X = np.array(['neutral', 'positive', 'negative'])
# Y = np.array([neutral, positive, negative])
# plt.bar(X,Y)
# plt.show()


# emotion
print(emotion_index2)
# label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
neutral = emotion_index2.count(0)
surprise = emotion_index2.count(1)
fear = emotion_index2.count(2)
sadness = emotion_index2.count(3)
joy = emotion_index2.count(4)
disgust = emotion_index2.count(5)
anger = emotion_index2.count(6)

print(neutral, surprise, fear, sadness, joy, disgust, anger)

X = np.array(['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'])
Y = np.array([neutral, surprise, fear, sadness, joy, disgust, anger])
plt.bar(X,Y)
plt.show()


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