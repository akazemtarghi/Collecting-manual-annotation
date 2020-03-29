import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
def GetCoordinates(m1,bbb):

    a = m1[bbb]

    list_x = []
    list_y = []

    for i in range(len(a['regions'])):
        a = m1[bbb]
        p = a['regions'][i]
        x = p['shape_attributes']['cx']
        y = p['shape_attributes']['cy']
        list_x.append(x)
        list_y.append(y)
    return list_x, list_y

def Read_json(filename):

    with open(filename) as f:
        m1 = json.load(f)
    bbb = list(m1.keys())[0]
    a = m1[bbb]
    kk = len(a['regions'])

    List_x, List_y = GetCoordinates(m1, bbb)
    return List_x, List_y


