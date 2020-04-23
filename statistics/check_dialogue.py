import pickle
import pandas as pd
import pprint
pd.set_option('display.max_rows', None)

def data_emotion_p_file():
    with open('../MELD/data/pickles/data_emotion.p', 'rb') as f:
        data = pickle.load(f)
        main_data = data[0]
        data_frame = pd.DataFrame(main_data)
        print(data_frame)

def meld_features_raw_pkl():
    with open ('../DialogueRNN/DialogueRNN_features/MELD_features/MELD_features_raw.pkl', 'rb') as f:
        meld_features = pickle.load(f)
        video_utterances = meld_features[5]
        emotion_labels = meld_features[2]#emotion labels:{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        sentiment_labels = meld_features[8] #sentiment labels: {'neutral': 0, 'positive': 1, 'negative': 2}
        data = pd.DataFrame([video_utterances, emotion_labels, sentiment_labels])
        print(data.T)
        # pprint.pprint(video_utterances)

if __name__ == '__main__':
    # data_emotion_p_file()
    meld_features_raw_pkl()
