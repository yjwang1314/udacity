library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape2)
data(diamonds)

# 1
# Create a histogram of diamond prices.
# Facet the histogram by diamond color
# and use cut to color the histogram bars.

# The plot should look something like this.
# http://i.imgur.com/b5xyrOu.jpg

# Note: In the link, a color palette of type
# 'qual' was used to color the histogram using
# scale_fill_brewer(type = 'qual')

ggplot(aes(x = price), data = diamonds) +
  scale_x_log10() +
  geom_histogram(aes(color = cut)) +
  facet_wrap(~color) +
  scale_fill_brewer(type = 'qual')

# 2
# Create a scatterplot of diamond price vs.
# table and color the points by the cut of
# the diamond.

# The plot should look something like this.
# http://i.imgur.com/rQF9jQr.jpg

ggplot(aes(x = table, y = price), data = diamonds) +
  geom_point(aes(color = cut)) +
  scale_color_brewer(type = 'qual')
  
# 4
# Create a scatterplot of diamond price vs.
# volume (x * y * z) and color the points by
# the clarity of diamonds. Use scale on the y-axis
# to take the log10 of price. You should also
# omit the top 1% of diamond volumes from the plot.

# Note: Volume is a very rough approximation of
# a diamond's actual volume.

# The plot should look something like this.
# http://i.imgur.com/excUpea.jpg

# Note: In the link, a color palette of type
# 'div' was used to color the scatterplot using
# scale_color_brewer(type = 'div')

diamonds$volume = diamonds$x * diamonds$y * diamonds$z 
ggplot(aes(x = volume, y = price), data = diamonds) +
  geom_point(aes(color = clarity)) +
  scale_y_log10() +
  xlim(0, quantile(diamonds$volume, 0.99)) +
  scale_color_brewer(type = 'div')

# 5
# Your task is to create a new variable called 'prop_initiated'
# in the Pseudo-Facebook data set. The variable should contain
# the proportion of friendships that the user initiated.

pf = read.delim('pseudo_facebook.tsv')
pf$prop_initiated = pf$friendships_initiated / pf$friend_count

# 6
# Create a line graph of the median proportion of
# friendships initiated ('prop_initiated') vs.
# tenure and color the line segment by
# year_joined.bucket.

# Recall, we created year_joined.bucket in Lesson 5
# by first creating year_joined from the variable tenure.
# Then, we used the cut function on year_joined to create
# four bins or cohorts of users.

# (2004, 2009]
# (2009, 2011]
# (2011, 2012]
# (2012, 2014]

# The plot should look something like this.
# http://i.imgur.com/vNjPtDh.jpg
# OR this
# http://i.imgur.com/IBN1ufQ.jpg

pf$year_joined = 2014 - ceiling(pf$tenure / 365)
pf$year_joined.bucket = cut(pf$year_joined, breaks = c(2004, 2009, 2011, 2012, 2014))
ggplot(aes(x = tenure, y = prop_initiated),
       data = subset(pf, !is.na(prop_initiated))) +
  geom_line(aes(color = year_joined.bucket),
            stat = 'summary',
            fun.y = median)

# 7
# Smooth the last plot you created of
# of prop_initiated vs tenure colored by
# year_joined.bucket. You can bin together ranges
# of tenure or add a smoother to the plot.

# There won't be a solution image for this exercise.
# You will answer some questions about your plot in
# the next two exercises.
ggplot(aes(x = tenure, y = prop_initiated),
       data = subset(pf, !is.na(prop_initiated))) +
  geom_line(aes(color = year_joined.bucket),
            stat = 'summary',
            fun.y = median) +
  geom_smooth()
  
# 9
year_groups = group_by(subset(pf, !is.na(prop_initiated)), year_joined.bucket)
pf.fc_by_year = summarise(year_groups,
                          mean_prop_init = mean(as.numeric(prop_initiated)),
                          median_prop_init = median(as.numeric(prop_initiated)),
                          n = n())

# 10
# Create a scatter plot of the price/carat ratio
# of diamonds. The variable x should be
# assigned to cut. The points should be colored
# by diamond color, and the plot should be
# faceted by clarity.

# The plot should look something like this.
# http://i.imgur.com/YzbWkHT.jpg.

# Note: In the link, a color palette of type
# 'div' was used to color the histogram using
# scale_color_brewer(type = 'div')

ggplot(aes(x = cut, y = price/carat), data = diamonds) +
  geom_point(aes(color = color)) +
  facet_wrap(~clarity) +
  scale_color_brewer(type = 'div')

