# coding:utf-8

import pandas as pd

'''
gt_src_df = pd.read_csv('gt.csv', header= None)
print gt_src_df.columns.values, gt_src_df.shape

gt_src_df.columns = ['name', 'x_min', 'y_min', 'x_max', 'y_max']

print gt_src_df.columns.values, gt_src_df.shape

gt_src_df.to_csv("new_gt.csv", index=False, sep=',', quotechar="'", header=True, encoding='utf-8')
'''

df = pd.read_csv("new_gt.csv")
filename = "201711070000000001001.jpg"
condition = df['name'] == filename

subdf = df[condition]
ar = subdf.iloc[:, 1:5].values[0]
print ar[-1]
