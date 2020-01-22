import os
from PIL import Image
import csv
import numpy as np

import torch
import torch.utils.data
from torch import nn
import torchvision

def make_datapath_list(root_path):
    """
    動画を画像データにしたフォルダへのファイルパスリストを作成する。
    root_path : str、データフォルダへのrootパス
    Returns：ret : video_list、動画を画像データにしたフォルダへのファイルパスリスト
    """

    # 動画を画像データにしたフォルダへのファイルパスリスト
    video_list = list()

    # root_pathにある、クラスの種類とパスを取得
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したフォルダへのパスを取得
    for class_list_i in (class_list):  # クラスごとのループ


        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)

        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in os.listdir(class_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            # フォルダでないmp4ファイルは無視
            if ext == '.mp4':
                continue

            # 動画ファイルを画像に分割して保存したフォルダのパスを取得
            video_img_directory_path = os.path.join(class_path, name)

            # vieo_listに追加
            video_list.append(video_img_directory_path)

    return video_list


# class VideoTransform():
#     """
#     動画を画像にした画像ファイルの前処理クラス。学習時と推論時で異なる動作をします。
#     動画を画像に分割しているため、分割された画像たちをまとめて前処理する点に注意してください。
#     """

#     def __init__(self, resize, crop_size, mean, std):
#         self.data_transform = {
#             'train': torchvision.transforms.Compose([
#                 # DataAugumentation()  # 今回は省略
#                 GroupResize(int(resize)),  # 画像をまとめてリサイズ　
#                 GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
#                 GroupToTensor(),  # データをPyTorchのテンソルに
#                 GroupImgNormalize(mean, std),  # データを標準化
#                 Stack()  # 複数画像をframes次元で結合させる
#             ]),
#             'val': torchvision.transforms.Compose([
#                 GroupResize(int(resize)),  # 画像をまとめてリサイズ　
#                 GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
#                 GroupToTensor(),  # データをPyTorchのテンソルに
#                 GroupImgNormalize(mean, std),  # データを標準化
#                 Stack()  # 複数画像をframes次元で結合させる
#             ])
#         }

#     def __call__(self, img_group, phase):
#         """
#         Parameters
#         ----------
#         phase : 'train' or 'val'
#             前処理のモードを指定。
#         """
#         return self.data_transform[phase](img_group)


#     # 前処理で使用するクラスたちの定義


# class GroupResize():
#     ''' 画像をまとめてリスケールするクラス。
#     画像の短い方の辺の長さがresizeに変換される。
#     アスペクト比は保たれる。
#     '''

#     def __init__(self, resize, interpolation=Image.BILINEAR):
#         '''リスケールする処理を用意'''
#         self.rescaler = torchvision.transforms.Resize(resize, interpolation)

#     def __call__(self, img_group):
#         '''リスケールをimg_group(リスト)内の各imgに実施'''
#         return [self.rescaler(img) for img in img_group]


# class GroupCenterCrop():
#     ''' 画像をまとめてセンタークロップするクラス。
#         （crop_size, crop_size）の画像を切り出す。
#     '''

#     def __init__(self, crop_size):
#         '''センタークロップする処理を用意'''
#         self.ccrop = torchvision.transforms.CenterCrop(crop_size)

#     def __call__(self, img_group):
#         '''センタークロップをimg_group(リスト)内の各imgに実施'''
#         return [self.ccrop(img) for img in img_group]

# class GroupToTensor():
#     ''' 画像をまとめてテンソル化するクラス。
#     '''

#     def __init__(self):
#         '''テンソル化する処理を用意'''
#         self.to_tensor = torchvision.transforms.ToTensor()

#     def __call__(self, img_group):
#         '''テンソル化をimg_group(リスト)内の各imgに実施
#         0から1ではなく、0から255で扱うため、255をかけ算する。
#         0から255で扱うのは、学習済みデータの形式に合わせるため
#         '''

#         return [self.to_tensor(img)*255 for img in img_group]


# class GroupImgNormalize():
#     ''' 画像をまとめて標準化するクラス。
#     '''

#     def __init__(self, mean, std):
#         '''標準化する処理を用意'''
#         self.normlize = torchvision.transforms.Normalize(mean, std)

#     def __call__(self, img_group):
#         '''標準化をimg_group(リスト)内の各imgに実施'''
#         return [self.normlize(img) for img in img_group]

# class Stack():
#     ''' 画像を一つのテンソルにまとめるクラス。
#     '''

#     def __call__(self, img_group):
#         '''img_groupはtorch.Size([3, 224, 224])を要素とするリスト
#         '''
#         ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
#                          for x in img_group], dim=0)  # frames次元で結合
#         # x.flip(dims=[0])は色チャネルをRGBからBGRへと順番を変えています（元の学習データがBGRであったため）
#         # unsqueeze(dim=0)はあらたにframes用の次元を作成しています


# # Kinetics-400のラベル名をIDに変換する辞書と、逆にIDをラベル名に変換する辞書を用意


# def get_label_id_dictionary(label_dicitionary_path='./video_download/kinetics_400_label_dicitionary.csv'):
#     label_id_dict = {}
#     id_label_dict = {}

#     with open(label_dicitionary_path, encoding="utf-8_sig") as f:

#         # 読み込む
#         reader = csv.DictReader(f, delimiter=",", quotechar='"')

#         # 1行ずつ読み込み、辞書型変数に追加します
#         for row in reader:
#             label_id_dict.setdefault(
#                 row["class_label"], int(row["label_id"])-1)
#             id_label_dict.setdefault(
#                 int(row["label_id"])-1, row["class_label"])

#     return label_id_dict,  id_label_dict


# # 確認
# label_dicitionary_path = './video_download/kinetics_400_label_dicitionary.csv'
# label_id_dict, id_label_dict = get_label_id_dictionary(label_dicitionary_path)
# label_id_dict





# 動作確認
root_path = './MELD/MELD.Raw/train_splits/'
video_list = make_datapath_list(root_path)
print(video_list[0])
print(video_list[1])