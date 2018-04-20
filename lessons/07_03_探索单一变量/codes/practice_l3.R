# 加载数据
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

