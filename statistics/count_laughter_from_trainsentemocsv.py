import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
import re
import pprint
import pandas as pd

detected_folder = '../laughter-detection/detected_train_lauthter/' # checked folder if laughter is detected. it contains detected and also undeted folder
# laughter_file = 'dia991_utt7/'

df = pd.read_csv('../MELD/data/MELD/train_sent_emo.csv', header=0)
video_utterances = meld_features[5]
emotion_labels = meld_features[2]#emotion labels:{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
sentiment_labels = meld_features[8] #sentiment labels: {'neutral': 0, 'positive': 1, 'negative': 2}

sample = df[(df['Dialogue_ID'] == 0) & (df['Utterance_ID'] == 3)] # sample DialogueID && UtteranceID
sample_utterance = sample['Utterance'].values.tolist() # sample utterance


def functor(f, l): # change str to int function
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

# get detected laughter index
laughter_file_path = glob.glob(detected_folder + '*/laugh_0.wav')
laughter_file = [os.path.basename(os.path.dirname(l)) for l in laughter_file_path]
laughter_index = [l.replace('dia', '').replace('utt','').split('_', 1) for l in laughter_file]
laughter_index = functor(int, laughter_index)  # laughter_index = [dialogue_index, utterance_index in dialogue]
laughter_index= sorted(laughter_index, key=lambda x: x[0])


# hold current utterance sentiment & previous utterance sentiment
current_neutral = 0
current_positive = 0
current_negative = 0
previous_neutral = 0
previous_positive = 0
previous_negative = 0
indexerror_sum = 0

for i, l in enumerate(laughter_index):
    dia_sentiment_index = l[0]
    utt_sentiment_index = l[1]
    dia_sentiment_list = sentiment_label_lists[l[0]]
    try:
        current_utt_sentiment = dia_sentiment_list[utt_sentiment_index]

        if current_utt_sentiment == 0:
            current_neutral += 1
        elif current_utt_sentiment == 1:
            current_positive += 1
        elif current_utt_sentiment == 2:
            current_negative += 1
        else:
            pass

        print('***NOT*** IndexEroor')
        print("current_dialogue:\n{}".format(video_utterances[dia_sentiment_index]))
        print('current_utterance:\n{}'.format(video_utterances[dia_sentiment_index][utt_sentiment_index]))
        print("dia_sentiment_index:{}, utt_sentiment_index:{}, dia_sentiment_list:{}".format(dia_sentiment_index, utt_sentiment_index, dia_sentiment_list))
        print('\n')
        # check previous sentiment
        if utt_sentiment_index > 0:
            previous_utt_sentiment = dia_sentiment_list[utt_sentiment_index-1]

            if previous_utt_sentiment == 0:
                previous_neutral += 1
            elif previous_utt_sentiment == 1:
                previous_positive += 1
            elif previous_utt_sentiment == 2:
                previous_negative += 1
            else:
                pass

        else:
            pass

    except IndexError:
        indexerror_sum += 1
        print('IndexError')
        print('laughter_index:{}'.format(l))
        print('current_dialogue:\n{}'.format(video_utterances[dia_sentiment_index]))
        print("dia_sentiment_index:{}, utt_sentiment_index:{}, dia_sentiment_list:{} ".format(dia_sentiment_index, utt_sentiment_index, dia_sentiment_list))
        print('\n')

    if i > 300:
        break

import pdb;pdb.set_trace()

print("current_neutral:{} \ncurrent_positive:{}, \ncurrent_negative:{}".format(current_neutral, current_positive, current_negative))
print("IndexError_SUM:{}".format(indexerror_sum))
X = np.array(['current_neutral', 'current_positive', 'current_negative'])
Y = np.array([current_neutral, current_positive, current_negative])
plt.bar(X,Y)
plt.show()

    # try:
    #     sentiment_index2.append(sentiment_index[l])
    # except IndexError:
    #     continue

    # import pdb;pdb.set_trace()


# emotion_index2 = []
# for l in laughter_index:
#     emotion_index = emotion_label_lists[l[0]-1]
#     try:
#         emotion_index2.append(emotion_index[l[1]])
#     except IndexError:
#         continue

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
# print(emotion_index2)
# # label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
# neutral = emotion_index2.count(0)
# surprise = emotion_index2.count(1)
# fear = emotion_index2.count(2)
# sadness = emotion_index2.count(3)
# joy = emotion_index2.count(4)
# disgust = emotion_index2.count(5)
# anger = emotion_index2.count(6)

# print(neutral, surprise, fear, sadness, joy, disgust, anger)

# X = np.array(['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'])
# Y = np.array([neutral, surprise, fear, sadness, joy, disgust, anger])
# plt.bar(X,Y)
# plt.show()


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