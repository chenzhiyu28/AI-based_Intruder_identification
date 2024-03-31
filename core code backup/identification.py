import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from statistics_record import *
# some code modified from K. Zhou, "Week3-coding-guide," 03/08/2023. [Online]. Available:
# https://learn.uq.edu.au/bbcswebdav/pid-9218109-dt-content-rid-56119911_1/xid-56119911_1


# not used later
class TrainingTestingNormalization:
    def __init__(self, training_data_frames, test_data_frames, features_indexes_for_testing):
        # always assume the last column is the label. (and all others are numerical feature values)
        # currently only numerical numbers are considered in training&testing
        self.ground_truth_label = training_data_frames.iloc[:, -1].values  # 所有record 的 label (ndarray)
        self.feature_values = training_data_frames.iloc[:, :10].values  # numerical features (1到10列),仅数据 (ndarray)

        self.ground_truth_label_test = test_data_frames.iloc[:, -1].values  # 和上面一样
        self.feature_values_test = test_data_frames.iloc[:, :10].values


# ------------------------- Stage 1: 创建 training 和 testing set -------------------------
training_data_frame = pd.read_csv('data.csv')
test_data_frame = pd.read_csv('test.csv')
"""
# ------------------------- Stage 1: nominal 转 numerical -------------------------

# 提取出 petal_width, 并转换成frame
petal_width_frame = df.iloc[:, 3].to_frame()  # 所有行, 第4列的所有数据

# 对petal_width进行One-hot编码，并转换为数组格式
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(petal_width_frame).toarray()  # 把nominal 转换成 类似 bitmap的数据
feature_labels = ohe.categories_[0]  # nominal的 feature name 和上面bitmap 对应

# 将One-hot编码后的数组转为DataFrame，列名为feature_labels
features = pd.DataFrame(feature_array, columns=feature_labels)  # 把原来的第4列拆成类似 bitmap, (让nominal 变成 numerical)


# 将原始的numerical数据、One-hot编码后的数据和类别标签合并成一个新的DataFrame
numerical_data = df.iloc[:, :3]  # 前3列(0,1,2)数字类数据
class_labels = df.iloc[:, 4]  # 第5列label数据
df_new = pd.concat([numerical_data, features, class_labels], axis=1)  # 把raw data 前3列 numerical 和 生成的bitmap组合起来
"""

# 提取类别标签、numerical和nominal数据
ground_truth_label = training_data_frame.iloc[:, -1].values  # 所有record 的 label (ndarray)
feature_values = training_data_frame.iloc[:, :10].values  # numerical features (1到10列),仅数据 (ndarray)
""" X_nom = df.iloc[:, 3:-1].values  # nominal features () """

ground_truth_label_test = test_data_frame.iloc[:, -1].values  # 和上面一样
feature_values_test = test_data_frame.iloc[:, :10].values

# 分割数据集为训练集和测试集 (ndarray) 这里不需要了因为已经分开 training 和 testing
# features_train, features_test, y_train, y_test = train_test_split(feature_values, ground_truth_label, random_state=0)

"""
print(type(training_data_frame))  # <class 'pandas.core.frame.DataFrame'>
print(type(feature_values))  # <class 'numpy.ndarray'>
print(feature_values_test)

print(len(feature_values))  # 120
print(len(feature_values_test))  # 40
"""

# ------------------------- Stage 2: normalization on numerical values -------------------------

# 对numerical数据进行标准化
scaler = StandardScaler()
scaler.fit(feature_values)

feature_values_normalized = scaler.transform(feature_values)
features_values_test_normalized = scaler.transform(feature_values_test)

# print(features_test)

# 合并标准化后的numerical数据和nominal数据 (这里只有numerical, 不需要了)
#X_train = np.concatenate((X_num_train, X_nom_train), axis=1)
#X_test = np.concatenate((X_num_test, X_nom_test), axis=1)

# ------------------------- Stage 3: construct decision trees -------------------------

# 初始化一个 decision tree 对象, random_state 让每次结果都一样
decision_tree = DecisionTreeClassifier(random_state=0)
# 输入feature_values 和 labels ,并用fit() 来训练完decision tree
decision_tree = decision_tree.fit(feature_values_normalized, ground_truth_label)

# ------------------------- Stage 4: predict -------------------------
# 预测 test_set的 label
labels_predicted = decision_tree.predict(features_values_test_normalized)

# accuracy 值
acc_dt = metrics.accuracy_score(ground_truth_label_test, labels_predicted)
print("The test accuracy of decision tree on the dataset is: ", acc_dt)

# f1 值
f1_dt = metrics.f1_score(ground_truth_label_test, labels_predicted, average='macro')
print("The test macro f1-score of decision tree on the dataset is: ", f1_dt)

plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=FEATURES, class_names=["non-intruder", "intruder"], rounded=True)
plt.show()

# ------------------------- Stage 5: constructing random forests -------------------------
random_forest = RandomForestClassifier(bootstrap=False, n_estimators=3, random_state=0)
#random_forest = RandomForestClassifier(n_estimators=3, max_depth=5, random_state=0)
random_forest = random_forest.fit(feature_values_normalized, ground_truth_label)

labels_predicted_random_forest = random_forest.predict(features_values_test_normalized)

# ------------------------- Stage 6: predict the result and evaluate randon forest -------------------------
acc_rf = metrics.accuracy_score(ground_truth_label_test, labels_predicted_random_forest)
print("The test accuracy of random forest on the dataset is: ", acc_rf)

f1_rf = metrics.f1_score(ground_truth_label_test, labels_predicted_random_forest, average='macro')
print("The test macro f1-score of random forest on the dataset is: ", f1_rf)
