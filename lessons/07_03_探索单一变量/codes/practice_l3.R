# 加载数据
library(ggplot2)
?diamonds
data(diamonds)

# obs & var
dim(diamonds)

# variable details
str(diamonds)

# Create a histogram of the price of
# all the diamonds in the diamond data set.
qplot(x = price, data = diamonds)
summary(diamonds$price)

# 
dim(subset(diamonds, diamonds$price < 500))
dim(subset(diamonds, diamonds$price < 250))
dim(subset(diamonds, diamonds$price >= 1500))

# Explore the largest peak in the
# price histogram you created earlier.

# Try limiting the x-axis, altering the bin width,
# and setting different breaks on the x-axis.
qplot(x = price, data = diamonds, binwidth = 100) +
  scale_x_continuous(limits = c(0, 5000),
                     breaks = seq(0, 5000, 500))

# Break out the histogram of diamond prices by cut.
qplot(x = price, data = diamonds, binwidth = 100) +
  scale_x_continuous(limits = c(0, 5000),
                     breaks = seq(0, 5000, 1000)) +
  facet_wrap(~cut, ncol = 2)

#
subset(diamonds, diamonds$price == max(diamonds$price))
subset(diamonds, diamonds$price == min(diamonds$price))
by(diamonds$price, diamonds$cut, summary)
table(diamonds$cut)

# Free scales
qplot(x = price, data = diamonds) +
  facet_wrap(~cut, scales="free_y", ncol = 2)

# Create a histogram of price per carat
# and facet it by cut. You can make adjustments
# to the code from the previous exercise to get
# started.

# Adjust the bin width and transform the scale
# of the x-axis using log10.

ggplot(aes(x = price/carat), data = diamonds) +
  geom_histogram() +
  scale_x_log10() +
  facet_wrap(~cut, scales="free_y", ncol = 2)

# Investigate the price of diamonds using box plots,
# numerical summaries, and one of the following categorical
# variables: cut, clarity, or color.

p1 = qplot(x = cut, y = price,
      data = diamonds,
      geom = 'boxplot') +
  coord_cartesian(ylim = c(0, 15000))

p2 = qplot(x = clarity, y = price,
           data = diamonds,
           geom = 'boxplot') +
  coord_cartesian(ylim = c(0, 15000))

p3 = qplot(x = color, y = price,
           data = diamonds,
           geom = 'boxplot') +
  coord_cartesian(ylim = c(0, 17000))

library(gridExtra)
grid.arrange(p1, p2, p3, ncol = 2)

# 四分位数间距 IQR
IQR(subset(diamonds, color == 'D')$price) 
IQR(subset(diamonds, color == 'J')$price)

# Investigate the price per carat of diamonds across
# the different colors of diamonds using boxplots.

qplot(x = color, y = price/carat,
           data = diamonds,
           geom = 'boxplot')

#
qplot(x = carat, data = diamonds, binwidth = 0.1) +
  scale_x_continuous(breaks = seq(0, 5, 0.5))

table(diamonds$carat)
