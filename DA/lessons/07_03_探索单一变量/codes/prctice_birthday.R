# Your task is to investigate the distribution of your friends'
# birth months and days.

# Here some questions you could answer, and we hope you think of others.

# **********************************************************************

# How many people share your birthday? Do you know them?
# (Reserve time with them or save money to buy them a gift!)

# Which month contains the most number of birthdays?

# How many birthdays are in each month?

# Which day of the year has the most number of birthdays?

# Do you have at least 365 friends that have birthdays on everyday
# of the year?

# Once you load the data into R Studio, you can use the strptime() function
# to extract the birth months and birth days. We recommend looking up the
# documentation for the function and finding examples online.

# 加载数据
library(ggplot2)
library(readr)

bd = read_csv("bd.csv", 
              col_types = cols(End = col_character(), 
                               Start = col_character(),
                               Duration = col_character()))

bd$Title = gsub('的生日', '', bd$Title) # gsub 代替，修改title
bd$bdate = as.Date(bd$Start, format = "%m/%d") # 转日期格式
bd$bmonth = format(bd$bdate, '%m') # 转成月
bd$bday = format(bd$bdate, '%d') # 转成日

# How many people share your birthday? Do you know them?
my_bd = subset(bd, bd$Title == '王毅俊')$bdate
same_bd = subset(bd, bd$bdate == my_bd)$Title # 2, Sin Chang

# Which month contains the most number of birthdays?
table(bd$bmonth) # Oct, 婴儿产品的旺季？
qplot(x = bmonth, data = bd)

# Which day of the year has the most number of birthdays?
table(bd$bday) # 24
qplot(x = bday, data = bd)

# Do you have at least 365 friends that have birthdays on everyday
# of the year?
# no
