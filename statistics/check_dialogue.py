import pickle
import pandas as pd

def data_emotion_p_file():
    with open('../MELD/data/pickles/data_emotion.p', 'rb') as f:
        data = pickle.load(f)
        main_data = data[0]
        data_frame = pd.DataFrame(main_data)
        pd.set_option('display.max_rows', None)
        print(data_frame)

def

if __name__ == '__main__':
    # data_emotion_p_file()
