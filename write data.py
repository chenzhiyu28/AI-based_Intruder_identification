import csv
from statistics_record import *
from pprint import pprint

# label all data in training set
non_intruders_data = features_1 + features_2
intruders_data = features_5 + features_6

for record in non_intruders_data:
    record.append("non_intruder")

for record in intruders_data:
    record.append("intruder")

training_data = non_intruders_data + intruders_data

# label all data in test set
non_intruders_test = features_1_predict + features_2_predict
intruders_test = features_5_predict + features_6_predict

for record in non_intruders_test:
    record.append("non_intruder")

for record in intruders_test:
    record.append("intruder")

test_data = non_intruders_test + intruders_test

# 写入csv 文件

with open('data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    for row in training_data:
        writer.writerow(row)


with open('test.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    for row in test_data:
        writer.writerow(row)
