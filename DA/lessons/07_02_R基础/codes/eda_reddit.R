setwd('G:/baiduDrive/online_course/udacity/DA/lessons/07_02_R基础/codes')
reddit = read.csv('reddit.csv')

# 查看每个就业组有多少人
table(reddit$employment.status) 
summary(reddit)

# 有序因子
library(ggplot2)
str(reddit)
levels(reddit$age.range)
qplot(data = reddit, x = age.range) # 画图
# qplot(data = reddit, x = income.range) # 画图

# setting levels of ordered factors solution
reddit$age.range = ordered(reddit$age.range, levels = c('Under 18', '18-24','25-34','35-44','45-54','55-64','65 or Above'))
qplot(data = reddit, x = age.range) # 画图