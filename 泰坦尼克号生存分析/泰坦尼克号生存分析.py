import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 Matplotlib 后端为 'Agg'，避免 GUI 相关错误
import matplotlib

matplotlib.use('TkAgg')

# 加载数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 查看数据基本信息
print(train_data.info())
print(test_data.info())
print("----------------------------------")
# 处理缺失值
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())  # 年龄用中位数填充
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])  # 登船港口用众数填充
train_data = train_data.drop(columns=['Cabin'])  # 删除 Cabin 列（缺失率过高）

# 测试集同样处理
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data = test_data.drop(columns=['Cabin'])

# 转换 Sex 和 Embarked
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 测试集同样处理
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 总体幸存率
survival_rate = train_data['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.2%}")

# 按性别分析
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Sex')
plt.savefig('survival_by_sex.png')  # 保存图表
plt.close()

# 按舱位等级分析
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Pclass')
plt.savefig('survival_by_pclass.png')  # 保存图表
plt.close()

# 年龄分布
sns.histplot(train_data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')  # 保存图表
plt.close()

# 年龄与幸存率
sns.boxplot(x='Survived', y='Age', data=train_data)
plt.title('Age Distribution by Survival')
plt.savefig('age_by_survival.png')  # 保存图表
plt.close()

# 创建家庭规模特征
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# 家庭规模与幸存率
sns.barplot(x='FamilySize', y='Survived', data=train_data)
plt.title('Survival Rate by Family Size')
plt.savefig('survival_by_family_size.png')  # 保存图表
plt.close()

# 提取称呼
train_data['Title'] = train_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 将称呼分组
train_data['Title'] = train_data['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

# 测试集同样处理
test_data['Title'] = test_data['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

# 将 Title 转换为数值型
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_data['Title'] = train_data['Title'].map(title_mapping)
test_data['Title'] = test_data['Title'].map(title_mapping)

# 删除无用特征
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket'])
test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket'])

# 划分训练集与验证集
from sklearn.model_selection import train_test_split

X = train_data.drop(columns=['Survived'])
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型（逻辑回归示例）
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 验证集预测
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2%}")

# 测试集预测
test_pred = model.predict(test_data)

import matplotlib.pyplot as plt

# 获取逻辑回归模型的系数
coefficients = model.coef_[0]
feature_names = X_train.columns

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.show()

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# # 计算混淆矩阵
# cm = confusion_matrix(y_val, y_pred)

# # 绘制混淆矩阵热力图
# plt.figure(figsize=(6, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Predicted 0', 'Predicted 1'],
#             yticklabels=['Actual 0', 'Actual 1'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# from sklearn.metrics import classification_report

# # 打印分类报告
# print(classification_report(y_val, y_pred, target_names=['Not Survived', 'Survived']))