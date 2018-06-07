# In this problem set, you'll continue
# to explore the diamonds data set.
library(ggplot2)
library(dplyr)
library(gridExtra)

# 1
# Your first task is to create a
# scatterplot of price vs x.
# using the ggplot syntax.
data(diamonds)
?diamonds
ggplot(aes(x = price, y = x), data = diamonds) +
  geom_point()

# 2
# 有一些异常值和价格与x存在指数关系

# 3
# correlation
with(diamonds, cor.test(price, x))
with(diamonds, cor.test(price, y))
with(diamonds, cor.test(price, z))

# 4
# Create a simple scatter plot of price vs depth
ggplot(aes(x = price, y = depth), data = diamonds) +
  geom_point()

# 5
# Change the code to make the transparency of the
# points to be 1/100 of what they are now and mark
# the x-axis every 2 units. See the instructor notes
# for two hints.
ggplot(aes(x = price, y = depth), data = diamonds) +
  geom_point(alpha = 1/100) +
  scale_x_continuous(breaks = seq(0, 53940, 2000))

# 7
with(diamonds, cor.test(depth, price))

# 8
# Create a scatterplot of price vs carat
# and omit the top 1% of price and carat
# values.
ggplot(aes(x = price, y = carat), data = diamonds) +
  geom_point() +
  xlim(0, quantile(diamonds$price, 0.99)) +
  ylim(0, quantile(diamonds$carat, 0.99))
  
# 9
# Create a scatterplot of price vs. volume (x * y * z).
# This is a very rough approximation for a diamond's volume.

# Create a new variable for volume in the diamonds data frame.
# This will be useful in a later exercise.
diamonds$volume = diamonds$x * diamonds$y * diamonds$z
ggplot(aes(x = price, y = volume), data = diamonds) +
  geom_point()

# 11
diamonds_sub = subset(diamonds, volume != 0 & volume < 800)
with(diamonds_sub, cor.test(price, volume))

# 12
# Subset the data to exclude diamonds with a volume
# greater than or equal to 800. Also, exclude diamonds
# with a volume of 0. Adjust the transparency of the
# points and add a linear model to the plot. (See the
# Instructor Notes or look up the documentation of
# geom_smooth() for more details about smoothers.)

# Do you think this would be a useful model to estimate
# the price of diamonds? Why or why not?

ggplot(aes(x = price, y = volume), data = diamonds_sub) +
  geom_line() +
  geom_smooth()

# 13
# Use the function dplyr package
# to create a new data frame containing
# info on diamonds by clarity.

# Name the data frame diamondsByClarity

# The data frame should contain the following
# variables in this order.

#       (1) mean_price
#       (2) median_price
#       (3) min_price
#       (4) max_price
#       (5) n

# where n is the number of diamonds in each
# level of clarity.
cla_group = group_by(diamonds, clarity)
diamonds_gp = summarise(cla_group,
                        mean_price = mean(as.numeric(price)),
                        median_price = median(as.numeric(price)),
                        min_price = min(as.numeric(price)),
                        max_price = max(as.numeric(price)),
                        n = n())
diamonds_gp = arrange(diamonds_gp, clarity)

# 14
# We’ve created summary data frames with the mean price
# by clarity and color. You can run the code in R to
# verify what data is in the variables diamonds_mp_by_clarity
# and diamonds_mp_by_color.

# Your task is to write additional code to create two bar plots
# on one output image using the grid.arrange() function from the package
# gridExtra.

diamonds_by_clarity <- group_by(diamonds, clarity)
diamonds_mp_by_clarity <- summarise(diamonds_by_clarity, mean_price = mean(price))

diamonds_by_color <- group_by(diamonds, color)
diamonds_mp_by_color <- summarise(diamonds_by_color, mean_price = mean(price))

p1 = ggplot(aes(x = clarity, y = mean_price), data = diamonds_mp_by_clarity) +
  geom_bar(stat = 'identity')

p2 = ggplot(aes(x = color, y = mean_price), data = diamonds_mp_by_color) +
  geom_bar(stat = 'identity')

grid.arrange(p1, p2, ncol = 1)

# 15
