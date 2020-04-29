import sys

import pandas as pd
import numpy as np
import os
import json
import csv
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Bidirectional, Embedding, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as kb
from keras import initializers
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
# !pip install python_highcharts
from highcharts import Highchart
import json


# %%

def getJsonList(filePath):
    fileList = []
    for i, j, k in os.walk(filePath):
        fileList = k
    return fileList


def calcWeight(item):
    zeros = 0
    for i in item['pose_keypoints_2d']:
        if i == 0:
            zeros += 1
    for i in item['face_keypoints_2d']:
        if i == 0:
            zeros += 1
    return zeros


def getJsonForOneFrame(fileName, filePath):
    l = getJsonList(filePath)
    with open(fileName, 'r') as f:
        temp = json.loads(f.read())
        allPeople = temp['people']
    if len(allPeople) == 0:
        empty = []
        for i in range(190):
            empty.append(0)
        return empty
    target = allPeople[0]
    for dic in allPeople:
        if calcWeight(dic) < calcWeight(target):
            target = dic
    matrix = []
    if (len(target['pose_keypoints_2d']) == 0):
        for i in range(50):
            matrix.append(0)
    else:
        for i in range(75):
            if (i + 1) % 3 == 0:
                pass
            else:
                matrix.append(target['pose_keypoints_2d'][i])
    if (len(target['face_keypoints_2d']) == 0):
        for i in range(140):
            matrix.append(0)
    else:
        for i in range(210):
            if (i + 1) % 3 == 0:
                pass
            else:
                matrix.append(target['face_keypoints_2d'][i])
    return matrix


def writeData(path, pathType, fullDfList):
    fileList = getJsonList(path)
    fileList = set(fileList)
    print(len(fileList))
    dataCat = pd.read_csv('data.csv')
    for index, row in dataCat.iterrows():
        temp = int(row['name'][:-1])
        if (row["type"] != pathType):
            pass
        else:
            startName = row["name"] + "_" + '0'.zfill(12) + "_keypoints.json"
            endName = row["name"] + "_" + '30'.zfill(12) + "_keypoints.json"
            if endName and endName not in fileList:
                print('Video ', row["name"], " is not long enough")
            else:
                initFrame = 0
                while (row["name"] + "_" + str(initFrame + 29).zfill(12) + "_keypoints.json" in fileList):
                    matrix = []
                    for i in range(30):
                        if (i + 1) % 5 == 0:
                            file = row["name"] + "_" + str(initFrame + i).zfill(12) + "_keypoints.json"
                            matrixLine = getJsonForOneFrame(path + '/' + file, path)
                            matrix.append(matrixLine)
                        else:
                            pass
                    if row['type'] == 'shortyes':
                        label = True
                    elif row['type'] == 'shortno':
                        label = False
                    else:
                        label = getLable(dataCat, row['name'], initFrame)
                    data = [[label, matrix]]
                    print("Data:", label, row['name'], initFrame, 'matrix:', len(matrix), len(matrix[0]))
                    df = pd.DataFrame(data, columns=['lable', 'matrix'])
                    fullDfList.append(df)
                    initFrame += 30
    return fullDfList


def draw(folderName, videoName, modelName):
    # videoName = '6'#define your test video name here
    fileList = getJsonList(folderName)
    fileList = set(fileList)
    print(len(fileList))
    fullData = []
    initFrame = 0
    while (folderName + "_" + str(initFrame + 29).zfill(12) + "_keypoints.json" in fileList):
        matrix = []
        for i in range(30):
            file = folderName + "_" + str(initFrame + i).zfill(12) + "_keypoints.json"
            matrixLine = getJsonForOneFrame(folderName + '/' + file, folderName)

            matrix.append(matrixLine)
        fullData.append(matrix)
        initFrame += 30

    def cleanX(raw):
        length = len(raw)
        temp = np.zeros((length, 10, 190))
        for i in range(len(raw)):
            for j in range(10):
                for k in range(190):
                    temp[i][j][k] = raw[i][j][k]
            print("progress:{0}%".format(round((i + 1) * 100 / len(raw))), end="\r")
        return temp

    temp = cleanX(fullData)

    #     model1 = load_model('10/balancetalking.h5')
    #     r1 = model1.predict(temp)

    #     model2 = load_model('10/balancedrinking.h5')
    #     r2 = model2.predict(temp)

    def ProcessSilentData(x_train, x_test):
        def proc(x_train_silent):
            length = len(x_train_silent)
            temp = np.zeros((length, 10, 46))
            for i in range(len(x_train_silent)):
                for j in range(10):
                    for k in range(190):
                        if 146 <= k <= 185:
                            temp[i][j][k - 146] = x_train_silent[i][j][k]
                    temp[i][j][40] = x_train_silent[i][j][67]  # face
                    temp[i][j][41] = x_train_silent[i][j][68]
                    temp[i][j][42] = x_train_silent[i][j][8]  # left hand
                    temp[i][j][43] = x_train_silent[i][j][9]
                    temp[i][j][44] = x_train_silent[i][j][14]  # right hand
                    temp[i][j][45] = x_train_silent[i][j][15]

            return temp

        x_train_40 = proc(x_train)
        x_test_40 = proc(x_test)
        x_train = x_train_40
        x_test = x_test_40

        # define distance between two points
        def calDis(material, a, b):
            # material = x_train[0][0]
            ax = (a - 1) * 2
            ay = ax + 1
            bx = (b - 1) * 2
            by = bx + 1
            d1 = abs(material[ax] - material[bx])
            d2 = abs(material[ay] - material[by])
            dis = np.sqrt(d1 * d1 + d2 * d2)
            return dis

        def getSilent(x):  # x_train[0][0]
            mouthLen = calDis(x, 13, 17)
            sideLen = calDis(x, 1, 12) + calDis(x, 1, 2) + calDis(x, 8, 7) + calDis(x, 6, 7)
            mouthWid = calDis(x, 14, 20) + calDis(x, 15, 19) + calDis(x, 16, 18)
            handLen = calDis(x, 21, 22) + calDis(x, 21, 23)
            if mouthLen == 0:
                silentWeight = 0
            else:
                silentWeight = mouthWid / mouthLen
            if sideLen == 0:
                sideWeight = 0
                handWeight = 0
            else:
                sideWeight = mouthWid / sideLen
                handWeight = handLen / sideLen
            if sideLen == 0:
                return [-1, -1]
            else:
                return [sideWeight, silentWeight]

        def proX(x_train):
            pro_x_train = np.zeros((len(x_train), 10, 4))
            for i in range(len(x_train)):
                for j in range(len(x_train[i])):
                    var0, var1 = getSilent(x_train[i][j])
                    pro_x_train[i][j][0] = var0
                    pro_x_train[i][j][1] = var1
            for i in range(len(pro_x_train)):  # 10 2
                varList1 = []
                varList2 = []
                for j in range(10):
                    varList1.append(pro_x_train[i][j][0])
                    varList2.append(pro_x_train[i][j][1])
                var2 = np.var(varList1)
                var3 = np.var(varList2)
                for j in range(10):
                    pro_x_train[i][j][2] = var2
                    pro_x_train[i][j][3] = var3
            return pro_x_train

        return proX(x_train), proX(x_test)

    # print(x_train.shape)

    x_train_silent, x_test = ProcessSilentData(temp, temp)
    x_train_eat, x_test = ProcessSilentData(temp, temp)

    model4 = load_model('model/talking.h5')  # 还是两个点的silenttalking不要动摇
    r4 = model4.predict(x_train_silent)

    model3 = load_model('model/confusing.h5')  # eating不要只搞两个点
    r3 = model3.predict(x_train_eat)

    # print(r4)

    X_TRAIN = []
    for i in range(len(r4)):
        # X_TRAIN.append([[r1[i][0]], [r2[i][0]], [r3[i][0]], [r4[i][0]]])
        X_TRAIN.append([[r4[i][0]], [r3[i][0]]])
    X_TRAIN = np.array(X_TRAIN)
    # print(X_TRAIN.shape)

    model = load_model(modelName)
    predict = model.predict(X_TRAIN)
    # return predict###########################

    a = predict.tolist()
    result = []
    frames = 30
    for i in a:
        result.append([frames, i[0]])
        frames += 30
    # print(result)
    chart = Highchart()
    chart.set_options('chart', {'inverted': False})
    options = {
        'title': {
            'text': 'Prediction for video ' + folderName + '.mp4'
        },
        'subtitle': {
            'text': '1 means talking while 0 means non-talking'
        },
        'xAxis': {
            'title': {
                'text': 'Second'
            }
        },
        'yAxis': {
            'title': {
                'text': 'Flag'
            },
        }
    }
    chart.set_dict_options(options)
    chart.add_data_set(result, series_type='line', name='prediction')
    newresult = []
    for r in result:
        newresult.append([r[0] / 30, 0.5])
    chart.add_data_set(newresult, series_type='line', name='talking/silent')

    index = []
    for i in result:
        for j in range(30):
            index.append(i[1])

    import cv2
    video = videoName
    result_video = folderName + '_result.mp4'
    cap = cv2.VideoCapture(video)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
    frame_id = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            left_x_up = int(frame_width / frame_id)
            left_y_up = int(frame_height / frame_id)
            right_x_down = int(left_x_up + frame_width / 10)
            right_y_down = int(left_y_up + frame_height / 10)
            word_x = left_x_up + 5
            word_y = left_y_up + 25
            if frame_id >= len(index):
                pass
            else:
                cv2.putText(frame, 'frame_%s' % index[frame_id], (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (55, 255, 155), 2)
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break

    frameRate = 30  # default video frame rate
    for i in result:
        i[0] = i[0] / frameRate

    d = {}
    d['talking'] = result
    file_name = 'timeLabel.json'
    with open(file_name, 'w') as file_object:
        json.dump(d, file_object)

    file_name = '430000313.json'
    with open(file_name, 'w') as file_object:
        json.dump(d, file_object)


    return chart


# %%
def main(argv):
    #folder_name = 'test1'  # type your folder path here, which contains openpose json files
    folder_name = argv[1]
    #video_name = 'test1.mp4'  # type your video path here, we want to know frame rate to create time labels
    video_name = argv[2]
    model_name = 'model/ensemble.h5'  # type your model name here
    chart1 = draw(folder_name, video_name, model_name)  # return a frame/label chart and saved a time/label json file
    chart1.save_file('timeLabel')  # save the chart
    chart1.save_file('430000313')  # save the chart


if __name__ == "__main__":
    main(sys.argv)