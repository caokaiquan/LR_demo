import pandas as pd
import matplotlib.pyplot as plt
admission = pd.read_csv('admissions.csv')
print(admission.head())
plt.scatter(admission['gpa'],admission['admit'])
plt.show()

################################################################################################

import numpy as np
def logit(x):
    return 1/(1+np.exp(-x))

# x = np.linspace(-5,5,50,dtype = float)
# y = logit(x)
#
# plt.plot(x,y)
# plt.show()

# from sklearn.linear_model import LinearRegression
# linear_model = LinearRegression()
# linear_model.fit(admission[['gpa']],admission['admit'])
from sklearn.linear_model import LogisticRegression
# logistic_model = LogisticRegression()
# logistic_model.fit(admission[['gpa']],admission['admit'])
#
# pred_probs = logistic_model.predict_proba(admission[['gpa']])
# plt.scatter(admission['gpa'],pred_probs[:,1]) #pred_probs[:,1]预测能通过的概率，pred_probs[:,0]预测不能通过的概率
# plt.show()
# print(pred_probs)

logistic_model = LogisticRegression()
logistic_model.fit(admission[['gpa']],admission['admit'])
fitted_labels = logistic_model.predict(admission[['gpa']])
plt.scatter(admission['gpa'],fitted_labels)
plt.show()

#模型效果评判
admissions = pd.read_csv('admissions.csv')
model = LogisticRegression()
model.fit(admissions[['gpa']],admissions['admit'])
labels = model.predict(admissions[['gpa']])
admissions['predicted_label'] = labels
admissions['predicted_label'].value_counts()

admissions['actual_label'] = admissions['admit']
matches = admissions['predicted_label'] == admissions['actual_label']
correct_predictions = admissions[matches]
correct_predictions.head()
accuracy = len(correct_predictions)/len(admissions)
print(accuracy)


true_positive_filter = (admissions['predicted_label'] == 1) & (admissions['actual_label'] == 1)
true_positives = len(admissions[true_positive_filter])

true_negative_filter = (admissions['predicted_label'] == 0) & (admissions['actual_label'] == 0)
true_negatives = len(admissions[true_negative_filter])
print(true_positive_filter,true_negative_filter)

false_positive_filter = (admissions['predicted_label'] == 1) & (admissions['actual_label'] == 0)
false_positives = len(admissions[false_positive_filter])

false_negative_filter = (admissions['predicted_label'] == 0) & (admissions['actual_label'] == 1)
false_negatives = len(admissions[false_negative_filter])

print(true_positives/(true_positives + false_negatives)) #recall
print(true_positives/(true_positives + false_positives)) #precision

################################################################################################

#拆分数据集为训练集与测试集
np.random.seed(8)
admissions = pd.read_csv('admissions.csv')
admissions['actual_label'] = admissions['admit']
admissions = admissions.drop('admit',axis = 1)
shuffle_index = np.random.permutation(admissions.index)
print(shuffle_index)
shuffled_admissions = admissions.loc[shuffle_index]

train = shuffled_admissions.iloc[0:515]
test = shuffled_admissions.iloc[515:]

model = LogisticRegression()
model.fit(train['gpa'],train['actual_label'])

labels = model.predict(test['gpa'])
test['predicted_label'] = labels

matches = test['predicted_label'] == test['actual_label']
correct_predictions = [matches]
accuracy = len(correct_predictions) / len(test)
print(accuracy)

#ROC曲线
from sklearn import metrics
probabilities = model.predict_proba(test[['gpa']])
fpr,tpr,thresholds = metrics.roc_curve(test['actual_label'],probabilities[:,1])
print(thresholds)
plt.plot(fpr,tpr)
plt.show()

from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test[['gpa']])
auc_score = roc_auc_score(test['actual_label'],probabilities[:,1]) #求roc曲线面积
print(auc_score)


# True Positive(真正, TP)：将正类预测为正类数.
#
# True Negative(真负 , TN)：将负类预测为负类数.
#
#
# False Positive(假正, FP)：将负类预测为正类数 → 误报 (Type I error). FPR为误诊率
#
#
# False Negative(假负 , FN)：将正类预测为负类数 →漏报 (Type II error).


# 直观上，TPR 代表能将正例分对的概率，FPR 代表将负例错分为正例的概率。




# ROC曲线指受试者工作特征曲线/接收器操作特性(receiver operating characteristic，ROC)曲线,是反映灵敏性和特效性连续变量的综合指标,是用构图法揭示敏感性和特异性的相互关系，它通过将连续变量设定出多个不同的临界值，从而计算出一系列敏感性和特异性。ROC曲线是根据一系列不同的二分类方式（分界值或决定阈），以真正例率（也就是灵敏度）（True Positive Rate,TPR）为纵坐标，假正例率（1-特效性）（False Positive Rate,FPR）为横坐标绘制的曲线。
#
# ROC观察模型正确地识别正例的比例与模型错误地把负例数据识别成正例的比例之间的权衡。TPR的增加以FPR的增加为代价。ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）。
# 纵坐标：真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# TPR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）
# 横坐标：假正率（False Positive Rate , FPR）
# FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）
# 形式：
# sklearn.metrics.roc_curve(y_true,y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
# 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
# 这里理解thresholds:
# 分类器的一个重要功能“概率输出”，即表示分类器认为某个样本具有多大的概率属于正样本（或负样本）。
# “Score”表示每个测试样本属于正样本的概率。
# 接下来，我们从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。其实，我们并不一定要得到每个测试样本是正样本的概率值，只要得到这个分类器对该测试样本的“评分值”即可（评分值并不一定在(0,1)区间）。评分越高，表示分类器越肯定地认为这个测试样本是正样本，而且同时使用各个评分值作为threshold。我认为将评分值转化为概率更易于理解一些。
#理想情况下，TPR应该接近1，FPR应该接近0。ROC曲线上的每一个点对应于一个threshold，对于一个分类器，每个threshold下会有一个TPR和FPR。比如Threshold最大时，TP=FP=0，对应于原点；Threshold最小时，TN=FN=0，对应于右上角的点(1,1)。
# 理想目标：TPR=1，FPR=0，即图中(0,1)点，故ROC曲线越靠拢(0,1)点，越偏离45度对角线越好，Sensitivity、Specificity越大效果越好。


################################################################################################


#交叉验证
import numpy as np
admissions = pd.read_csv('admissions.csv')
admissions['actual_label'] = admissions['admit']
admissions = admissions.drop('admit',axis=1)

shuffle_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffle_index]
admissions = shuffled_admissions.reset_index()
admissions.ix[0:128,'fold'] = 1
admissions.ix[129:257,'fold'] = 2
admissions.ix[258:386,'fold'] = 3
admissions.ix[387:514,'fold'] = 4
admissions.ix[515:644,'fold'] = 5
admissions['fold'] = admissions['fold'].astype('int')

model = LogisticRegression()
train_iteration_one = admissions[admissions['fold'] != 1]
test_iteration_one = admissions[admissions['fold'] == 1]
model.fit(train_iteration_one[['gpa']],train_iteration_one['actual_label'])

labels = model.predict(test_iteration_one[['gpa']])
test_iteration_one['predicted_label'] = labels

matches = test_iteration_one['predicted_label'] == test_iteration_one['actual_label']
correct_predictions = test_iteration_one[matches]
iteration_one_accuracy = len(correct_predictions)/len(test_iteration_one)
print(iteration_one_accuracy)

#五次交叉
fold_ids = [1,2,3,4,5]
def train_and_test(df,folds)
    fold_accuracies = []
    for fold in folds:
        model = LogisticRegression()
        train = admissions[admissions['fold'] != fold]
        test = admissions[admissions['fold'] == fold]
        model.fit(train[['gpa']],train['actual_label'])
        labels = model.predict(test[['gpa']])
        test['predicted_label'] = labels

        matches = test['predicted_label'] == test['actual_label']
        correct_predictions = test[matches]
        fold_accuracies.append(len(correct_predictions)/len(test))
    return fold_accuracies

accuracies = train_and_test(admissions,fold_ids)
print(accuracies)
print(np.mean(accuracies))

################################################################################################

#用sklearn交叉验证
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

admissions = pd.read_csv('admissions.csv')
admissions['actual_label'] = admissions['admit']
admissions.drop('admit',axis = 1,inplace=True)
kf = KFold(len(admissions),5,shuffle=True,random_state = 8)
lr = LogisticRegression()
accuracies = cross_val_score(lr,admissions[['gpa']],admissions['actual_label'],scoring='accuracy',cv = kf)
average_accuracy = sum(accuracies)/len(accuracies)
print(accuracies,average_accuracy)

################################################################################################

###################多分类LR
import pandas as pd
import matplotlib.pyplot as plt
columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
cars = pd.read_table('auto-mpg.data',delim_whitespace=True,names = columns)
cars.head()

dummy_cylinders = pd.get_dummies(cars['cylinders'],prefix='cy1')
cars = pd.concat([cars,dummy_cylinders],axis = 1)
dummy_years = pd.get_dummies(cars['year'],prefix='year')
cars = pd.concat([cars,dummy_years],axis = 1)
cars = cars.drop('year',axis = 1)
cars = cars .drop('cylinders',axis = 1)
cars.head()

import numpy as np
shuffled_rows = np.random.permutation(cars.index)
shuffled_cars  = cars.iloc[shuffled_rows]
highest_train_row = int(cars.shape[0] * 0.7)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]

from sklearn.linear_model import LogisticRegression
#用LR训练一个多分类模型，需训练三次
unique_origins = cars['origin'].unique()
unique_origins.sort()
models = {}
features = [c for c in train.columns if c.startswith('cy1') or c.startswith('year')]  #先利用我们刚刚生成的特征进行测试

for origin in unique_origins:
    model = LogisticRegression()
    X_train = train[features]
    y_train = train['origin'] == origin          #=的为1，不等于的为0，true就是1，false就是0
    model.fit(X_train,y_train)
    model[origin] = model

testing_probs = pd.DataFrame(columns = unique_origins)
for origin in unique_origins:
    X_test = test[features]
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]

predicted_origins = testing_probs.idxmax()
print(predicted_origins)