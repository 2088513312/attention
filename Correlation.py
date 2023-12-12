import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据文件
data = pd.read_csv('data.csv')

# 计算相关系数矩阵,相关系数是一个统计指标，用于衡量两个变量之间的关系强度和方向，
# 范围从-1到1，值越接近1表示两个变量正相关，越接近-1表示两个变量负相关，值为0表示两个变量之间没有线性关系。
corr_matrix = data.corr()

# 绘制相关系数矩阵的热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
