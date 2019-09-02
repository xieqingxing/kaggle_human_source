# -- coding:utf-8 --
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./Human-Resources-Analytics-master/HR_comma_sep.csv')
# check none value
print(df.isnull().any())
# 改名
df = df.rename(columns={'satisfaction_level': 'satisfaction_level',
                        'last_evaluation': 'last_evaluation',
                        'number_project': 'number_project',
                        'average_montly_hours': 'average_montly_hours',
                        'time_spend_company': 'time_spend_company',
                        'Work_accident': 'Work_accident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'left'
                        })
# 查看部门和薪资的唯一值有多少份
df1 = pd.Series(df['department']).unique()
df2 = pd.Series(df['salary']).unique()
# print(df1)
# 把数据数值化
df['department'].replace(list(pd.Series(df['department']).unique()),np.arange(10),inplace=True)
df['salary'].replace(list(pd.Series(df['salary']).unique()),[0,1,2],inplace=True)
# print(df['salary'])

# 把left列移到表的前面,方便分析
front = df['left']
df.drop(labels='left',axis=1,inplace=True)
df.insert(0,'left',front)
# print(df.head())

# 查看数据形状和结构 (14999,10)
# print(df.shape)
# print(df.dtypes)

# left：是否离职
# satisfaction_level：满意度
# last_evaluation：绩效评估
# number_project：完成项目数
# average_montly_hours：平均每月工作时间
# time_spend_company：为公司服务的年限
# work_accident：是否有工作事故
# promotion：过去5 年是否有升职
# salary：薪资水平

left_rate = df.left.value_counts() / 14999
print(left_rate)
# 0    0.761917
# 1    0.23808

# 探索分析
# 简单的聚合计算,均值,方差,最大最小值,四分位数
left_summary=df.groupby('left')
left_summary.mean()
format=lambda x: '%.2f'%x
print(df.describe().applymap(format))

# 相关分析
corr = df.corr()

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
plt.show()
# 热力图可以得出每个维度的相关关系


##department  vs  left
depart_left_table=pd.crosstab(index=df['department'],columns=df['left'])
##职位：'sales', 'accounting', 'hr', 'technical', 'support', 'management','IT', 'product_mng', 'marketing', 'RandD'
depart_left_table.plot(kind='bar',figsize=(5,5),stacked=True)
# 职位和是否离职的关系
##department  vs  salary
depart_salary_table=pd.crosstab(index=df['department'],columns=df['salary'])
depart_salary_table.plot(kind="bar",figsize=(5,5),stacked=True)
##salary  vs left
salary_left_table=pd.crosstab(index=df['salary'],columns=df['left'])
salary_left_table.plot(kind='bar',figsize=(5,5),stacked=True)
##promotion  vs  left
promotion_left_table=pd.crosstab(index=df['promotion'],columns=df['left'])
promotion_left_table.plot(kind='bar',figsize=(5,5),stacked=True)
##number_project  vs  left
project_left_table=pd.crosstab(index=df['number_project'],columns=df['left'])
project_left_table.plot(kind='bar',figsize=(5,5),stacked=True)
df.loc[(df['left']==1),'number_project'].plot(kind='hist',normed=1,bins=15,stacked=False,alpha=1)
##time_spend_company  vs  left
company_left_table=pd.crosstab(index=df['time_spend_company'],columns=df['left'])
company_left_table.plot(kind='bar',figsize=(5,5),stacked=True)
df.loc[(df['left']==1),'time_spend_company'].plot(kind='hist',normed=1,bins=10,stacked=False,alpha=1)
##average_montly_hours  vs  left
hours_left_table=pd.crosstab(index=df['average_montly_hours'],columns=df['left'])
fig=plt.figure(figsize=(10,5))
letf=sns.kdeplot(df.loc[(df['left']==0),'average_montly_hours'],color='b',shade=True,label='no left')
left=sns.kdeplot(df.loc[(df['left']==1),'average_montly_hours'],color='r',shade=True,label='left')
##last_evaluation  vs  left
evaluation_left_table=pd.crosstab(index=df['last_evaluation'],columns=df['left'])
fig=plt.figure(figsize=(10,5))
left=sns.kdeplot(df.loc[(df['left']==0),'last_evaluation'],color='b',shade=True,label='no left')
left=sns.kdeplot(df.loc[(df['left']==1),'last_evaluation'],color='r',shade=True,label='left')
##satisfaction_level  vs  left
satis_left_table=pd.crosstab(index=df['satisfaction_level'],columns=df['left'])
fig=plt.figure(figsize=(10,5))
left=sns.kdeplot(df.loc[(df['left']==0),'satisfaction_level'],color='b',shade=True,label='no left')
left=sns.kdeplot(df.loc[(df['left']==1),'satisfaction_level'],color='r',shade=True,label='left')
##last_evaluation  vs  satisfaction_level
df1=df[df['left']==1]
fig, ax = plt.subplots(figsize=(10,10))
pd.scatter_matrix(df1[['satisfaction_level','last_evaluation']],color='k',ax=ax)
plt.savefig('scatter.png',dpi=1000,bbox_inches='tight')

# 总结
#
# 员工离职概述：
#
# 离职员工工作时间大部分是~6hours /天（工作）和~10小时/天（劳累）；
#
# 大部分离职员工薪资都在low~medium这一档，薪资水平低；
#
# 离职员工，几乎都没有得到升职；
#
# 大多数离职员工的评价分数在0.6以下和0.8以上；
#
# 离职员工大多数有2个项目，但同样有4-7个项目的员工离开，3个项目的员工离职率最低；
#
# 完成项目数，每月平均工作时间，绩效评估有正相关关系。意味着你工作越多，得到的评价就越高；
#
# 离职率、满意度与薪酬呈负相关关系。这意味着较低的满意度和工资产生了较高的离职率；
#
# 公司需要考虑的问题：
#
# 1、失去优秀员工会让公司产生多大损失？招新人和优秀老员工之间的成本与变现孰轻孰重？
#
# 2、什么原因产生了较低的满意度？
#
# 3、为什么离开的员工平均比没有离开的员工得到更高的评价，甚至是项目数量的增加？低评价的员工不应该更倾向于离开公司吗？
#
# 优秀员工看中的是良好的待遇，和更好的职业发展，这些因素都直接影响员工的主观感受，公司给予了员工高的评价，但没有相应转化到薪资和升职的变量中，即使一部分离职的优秀员工给予了公司不错的满意度，但依然不能阻挡他们会追寻更好的工作机会。


