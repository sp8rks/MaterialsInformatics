# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:07:00 2020

@author: Steven Kauwe
"""
import pandas as pd
from time import sleep
from cbfv import composition


df = pd.read_csv('test_data_extended_feats.csv')

print('Featurizing DataFrame without extended features')
sleep(1)
output = composition.generate_features(df)
X_cbfv = output[0]


print('Featurizing DataFrame with extended features')
sleep(1)
output = composition.generate_features(df, extend_features=True)
X_extended = output[0]
