import pandas as pd

# 加载数据集
user_data = pd.read_csv('UserBehavior.csv')

# 列名矫正
user_data.columns = ['用户ID', '商品ID', '商品类目ID', '行为类型', '时间戳']

# 查看数据基本信息
print(user_data.info())

# 缺失值处理
isnullSum = user_data.isnull().sum()
# print('缺失数量:\n', isnullSum)  # 没有缺失值

# 重复值处理
duplicatedSum = user_data.duplicated().sum()
# print('重复数量:\n', duplicatedSum)  # 49
user_data.drop_duplicates(inplace=True)

# 异常值检测与处理
valid_values = {'pv', 'buy', 'cart', 'fav'}
invalid_mask = ~user_data['行为类型'].isin(valid_values)
invalid_user = user_data[invalid_mask]
# print("基础筛选结果:\n", invalid_user)  # Empty DataFrame 无异常值


# 日期格式转换
user_data['时间戳'] = pd.to_datetime(user_data['时间戳'], unit='s', errors='coerce')

# 按用户和时间排序
user_data = user_data.sort_values(['用户ID', '时间戳'])

# 映射行为到漏斗阶段（按转化优先级排序）
behavior_priority = {
    'pv': 1,  # 点击
    'fav': 2,  # 收藏
    'cart': 3,  # 加购
    'buy': 4  # 购买
}

# 添加行为优先级列
user_data['行为优先级'] = user_data['行为类型'].map(behavior_priority)

# 删除用户未完成首次点击的记录
valid_users = user_data[user_data['行为类型'] == 'pv']['用户ID'].unique()
user_data = user_data[user_data['用户ID'].isin(valid_users)]

# 标记用户是否完成购买
user_data['完成购买'] = user_data.groupby('用户ID')['行为类型'].transform(lambda x: 'buy' in x.values)

# 另存为
user_data.to_csv("UserBehavior_cleaned_v1_20240318.csv", index=False)
