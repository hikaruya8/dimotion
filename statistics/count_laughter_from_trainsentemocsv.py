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
# video_utterances = meld_features[5]
# emotion_labels = meld_features[2]#emotion labels:{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
# sentiment_labels = meld_features[8] #sentiment labels: {'neutral': 0, 'positive': 1, 'negative': 2}

sample = df[(df['Dialogue_ID'] == 0) & (df['Utterance_ID'] == 3)] # sample DialogueID && UtteranceID
sample_utterance = sample['Utterance'].values.tolist() # sample utterance





def functor(f, l): # change str to int function
  if isinstance(l,list):
    return [functor(f,i) for i in l]
  else:
    return f(l)

# # change to list
# utterance_lists = list(video_utterances.values())
# emotion_label_lists = list(emotion_labels.values())
# sentiment_label_lists = list(sentiment_labels.values())
# sentiment_label_lists_np = np.array(sentiment_label_lists)


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

# emotion
# label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
# current emotion sum
current_neu = 0
current_sur = 0
current_fea = 0
current_sad = 0
current_joy = 0
current_dis = 0
current_ang = 0

# previous emotion sum
pre_neu = 0
pre_sur = 0
pre_fea = 0
pre_sad = 0
pre_joy = 0
pre_dis = 0
pre_ang = 0

# check index error sum
indexerror_sum = 0


for i, l in enumerate(laughter_index):
    dia_index = l[0]
    utt_index = l[1]
    current_df =  df[(df['Dialogue_ID'] == dia_index) & (df['Utterance_ID'] == utt_index)]
    current_utt = current_df['Utterance'].values.tolist()
    current_senti = current_df['Sentiment'].values.tolist()
    current_emo = current_df['Emotion'].values.tolist()

    try:
        if current_senti == ['neutral']:
            current_neutral += 1
        elif current_senti == ['positive']:
            current_positive += 1
        elif current_senti == ['negative']:
            current_negative += 1
        else:
            pass

        # label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        if current_emo == ['neutral']:
            current_neu += 1
        elif current_emo == ['surprise']:
            current_sur += 1
        elif current_emo == ['fear']:
            current_fea += 1
        elif current_emo == ['sadness']:
            current_sad += 1
        elif current_emo == ['joy']:
            current_joy += 1
        elif current_emo == ['disgust']:
            current_dis += 1
        elif current_emo == ['anger']:
            current_ang += 1
        else:
            pass

        # check previous sentiment
        if utt_index > 0:
            previous_df =  df[(df['Dialogue_ID'] == dia_index) & (df['Utterance_ID'] == (utt_index-1))]
            pre_utt = previous_df['Utterance'].values.tolist()
            pre_senti = previous_df['Sentiment'].values.tolist()
            pre_emo = previous_df['Emotion'].values.tolist()

            if pre_senti == ['neutral']:
                previous_neutral += 1
            elif pre_senti == ['positive']:
                previous_positive += 1
            elif pre_senti== ['negative']:
                previous_negative += 1
            else:
                pass

         # check previous emotion
            if pre_emo == ['neutral']:
                pre_neu += 1
            elif pre_emo == ['surprise']:
                pre_sur += 1
            elif pre_emo== ['fear']:
                pre_fea += 1
            elif pre_emo == ['sadness']:
                pre_sad += 1
            elif pre_emo == ['disgust']:
                pre_dis += 1
            elif pre_emo == ['anger']:
                pre_ang += 1
            else:
                pass

        else:
            pass

    except IndexError:
        indexerror_sum += 1


    except IndexError:
        indexerror_sum += 1
        print('***IndexEroor***')



def current_senti_graph():
    print("current_neutral:{} \ncurrent_positive:{}, \ncurrent_negative:{}".format(current_neutral, current_positive, current_negative))
    print("IndexError_SUM:{}".format(indexerror_sum))
    X = np.array(['current_neutral', 'current_positive', 'current_negative'])
    Y = np.array([current_neutral, current_positive, current_negative])
    plt.title('Current_Sentiment')

    plt.bar(X,Y)
    plt.show()

def previous_senti_graph():
    print("previous_neutral:{} \n previous_positive:{}, \n previous_negarive:{}".format(previous_neutral, previous_positive, previous_negative))
    X = np.array(['previous_neutral', 'previous_positive', 'previous_negative'])
    Y = np.array([previous_neutral, previous_positive, previous_negative])
    plt.title('Previous_Sentiment')
    plt.bar(X,Y)
    plt.show()



# emotion
# label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

def current_emo_graph():
    print('current_neutral:{}, current_surprise:{}, current_fear:{}, current_sadness:{}, current_joy:{}, current_disgust:{}, current_anger:{}'.format(current_neu, current_sur, current_fea, current_sad, current_joy, current_dis, current_ang))
    X = np.array(['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'])
    Y = np.array([current_neu, current_sur, current_fea, current_sad, current_joy, current_dis, current_ang])
    plt.title('Current_Emotion')
    plt.bar(X,Y)
    plt.show()

def previous_emo_graph():
    print('pre_neutral:{}, pre_surprise:{}, pre_fear:{}, pre_sadness:{}, pre_joy:{}, pre_disgust:{}, pre_anger:{}'.format(pre_neu, pre_sur, pre_fea, pre_sad, pre_joy, pre_dis, pre_ang))
    X = np.array(['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'])
    Y = np.array([pre_neu, pre_sur, pre_fea, pre_sad, pre_joy, pre_dis, pre_ang])
    plt.title('Previous_Emotion')
    plt.bar(X,Y)
    plt.show()

if __name__ == '__main__':
    # current_senti_graph()
    # previous_senti_graph()
    # current_emo_graph()
    previous_emo_graph()

