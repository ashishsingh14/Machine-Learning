# Machine-Learning

This repository contains the code clustering using K-means, dbscan, mean shift algorithm. It also includes two methods to calculate the threshold values in order to detect the outliers in the data. The data used here is the time series data.
Two methods for threshold values are -
1) First one calculates the threshold value by calculating the distance of each point to its corresponding cluster. Then we add the distance values multiplied by the total number of points for each cluster. This step is repeated for each cluster.
After this the value obtained is divided by the total number of points. The points whose distance is larger than this threshold value are considered as outliers.

2) In order to check whether a farthest point is an outlier or not we calculate the ratio (distance of nearest point to centroid)/ (distance of farthest point to centroid). If this ratio is less than alpha then this farthest point is considered as an outlier point. Here alpha is a parameter which the user need to tune accordingly. Here in this case it ranges from  0.0001 to 0.1.

Feel free to comment or suggest any feedback.
