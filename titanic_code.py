# 导入数据包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import re

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 使用pandas读入数据
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

PassengerId=test['PassengerId']
all_data = pd.concat([train, test], ignore_index = True)

# 使用describe(),info(),head()查看数据
train.info()
print("*"*40)
print(train.head())
print("*"*40)
pd.set_option('display.max_columns', None)
print(train.describe())
print("*"*40)

# 数据初步分析，使用统计学与绘图，了解数据之间的相关性，为构造特征工程以及建立模型做准备
print(train['Survived'].value_counts())
# sex性别与生存率的关系
sns.barplot(x="Sex", y="Survived", data=train)
plt.show()
# pclass社会等级与生存率的关系
sns.barplot(x="Pclass", y="Survived", data=train)
plt.show()
# sibsp配偶及兄弟姐妹数与生存率的关系
sns.barplot(x="SibSp", y="Survived", data=train)
plt.show()
# parch父母与子女数与生存率的关系
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()
# 年龄与生存情况的关系
facet = sns.FacetGrid(train, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlabel('Age')
plt.ylabel('density')
plt.show()
# Embarked登陆港口与生存情况的分析
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.show()

# 新增特征title，研究姓名中不同称谓与生存率的关系
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data)
plt.show()

# 新增FamilyLabel特征,FamilySize=Parch+SibSp+1,研究家庭总人数与生存率的关系
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=all_data)
plt.show()

# 按生存率把FamilySize分为三类，构成FamilyLabel特征
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data)
plt.show()

# 新增deck特征，研究不同cabin的乘客生存率
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data)
plt.show()

# 新增ticketgroup特征，研究共票号乘客的生存率
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=all_data)
plt.show()

# 将ticketgroup分为三类
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data)
plt.show()

# 数据清洗
# 补充缺失值
from sklearn.ensemble import RandomForestRegressor

# 填充年龄缺失值
age_df = all_data[['Age', 'Pclass','Sex','Title']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1:])
all_data.loc[all_data['Age'].isnull(), 'Age'] = predictedAges

# Embarked缺失值填充为C
# print(all_data[all_data['Embarked'].isnull()])
# print(all_data.groupby(by=["Pclass", "Embarked"]).Fare.median())
all_data[all_data['Embarked'].isnull()]
all_data.groupby(by=["Pclass", "Embarked"]).Fare.median()
all_data['Embarked'] = all_data['Embarked'].fillna('C')

# fare填充
# 查看缺失fare值的乘客信息
# print(all_data[all_data['Fare'].isnull()])
all_data[all_data['Fare'].isnull()]
# 用Embarked为S，Pclass为3的乘客的Fare中位数填充
fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)
'''
# 标签特征处理，使用get_dummies()得到one-hot标签
all_data = pd.get_dummies(all_data, columns=['Embarked', 'Sex', 'Title', 'Deck'])
# 选择数值型的列进行归一化
scaler = MinMaxScaler()
all_data[['Age', 'Fare']] = scaler.fit_transform(all_data[['Age', 'Fare']])
'''

# 同组识别
# 把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性。
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
# 发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难。
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Female_Child

sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived')
plt.show()
# 绝大部分成年男性组的平均存活率
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Male_Adult
# 把不符合普遍规律的反常组选出来单独处理
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)
# 为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'


# 特征转换
# 选取特征，转换为数值变量，划分训练集和测试集
all_data=pd.concat([train, test])
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.values[:,1:]
y = train.values[:,0]

# 建模和优化
# 参数优化
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest


pipe=Pipeline([('select',SelectKBest(k=20)),
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)),
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
gsearch.fit(X,y)
print(gsearch.best_params_, gsearch.best_score_)

# 模型训练
'''
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True,
                                  n_estimators = 26,
                                  max_depth = 6,
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

# 交叉验证
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

# 预测
import os
sub = r'.\submission1.csv'
predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv(sub, index=False)
'''

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# from titanic_code import all_data

# 特征工程后的数据
features = X
labels = y

# 将数据集分为训练集和测试集（70%训练集，30%测试集）
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 使用SVM算法训练模型
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 使用KNN算法训练模型
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# 预测和评估模型性能
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Metrics for {model_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print()

# 评估SVM模型
evaluate_model(svm_model, X_test, y_test, "SVM")

# 评估KNN模型
evaluate_model(knn_model, X_test, y_test, "KNN")







