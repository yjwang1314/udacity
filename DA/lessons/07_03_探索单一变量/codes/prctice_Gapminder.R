# The Gapminder website contains over 500 data sets with information about
# the world's population. Your task is to download a data set of your choice
# and create 2-5 plots that make use of the techniques from Lesson 3.

# You might use a simple histogram, a boxplot split over a categorical variable,
# or a frequency polygon. The choice is yours!

# 未成功生成直方图

# 加载数据
library(ggplot2)
library(readr)

md = read.csv('indicator age of marriage.csv', header = T, row.names = 1, check.names = F)

data_1 = subset(md, !is.na(md$'2005'))


# obs & var
dim(md)

# variable details
str(md)

summary(md$'2005')

qplot(x = '2005', data = subset(md, !is.na(md$'2005')))

ggplot(aes(x = '2005'), data = subset(md, !is.na(md$'2005'))) +
  geom_histogram(stat = 'count', binwidth=0.5)
