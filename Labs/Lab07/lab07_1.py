"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-13-19
Assignment: Lab 07 - Regression

Notes:

Exercise 7.1 - Intro to Pandas

"""

###########################################################################################

from __future__ import print_function

import pandas as pd
import numpy as np

###########################################################################################

var = pd.__version__
print("Version: " + var)
print("\n")

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

df = pd.DataFrame({'City name': city_names, 'Population': population})
print(df)
print("\n")

california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
chdf = california_housing_dataframe.describe()
print(chdf)
print("\n")

chdhead = california_housing_dataframe.head()
print(chdhead)
print("\n")

chdhist = california_housing_dataframe.hist('housing_median_age')
print("\n" + str(chdhist) + "\n")

cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(type(cities['City name']))
print(cities['City name'])
print("\n")
print(type(cities['City name'][1]))
print(cities['City name'][1])
print("\n")
print(type(cities[0:2]))
print(cities[0:2])
print("\n")

arithmetic = population / 1000.
print(arithmetic)
print("\n")

numpy = np.log(population)
print(numpy)
print("\n")

lamb = population.apply(lambda val: val > 1000000)
print(lamb)
print("\n")

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)
print("\n")

###########################################################################################
"""
Exercise #1
Modify the cities table by adding a new boolean column that is True if and only if both of the following are True:

The city is named after a saint.
The city has an area greater than 50 square miles.
Note: Boolean Series are combined using the bitwise, rather than the traditional boolean, operators. 
For example, when performing logical and, use & instead of and.

Hint: "San" in Spanish means "saint."

Resources Used:

URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html
"""

test = cities
test['Boolean series 1'] = cities['Area square miles'].apply(lambda greater: greater > 50)
test['Boolean series 2'] = city_names.str.contains("San", regex=False)
print(test)
print("\n")

cities['Exercise 1 Boolean Series'] = city_names.str.contains("San", regex=False) & \
                                      cities['Area square miles'].apply(lambda greater: greater > 50)
print(cities)
print("\n")

###########################################################################################

city_name_index = city_names.index
print(city_name_index)
print("\n")

city_index = cities.index
print(city_index)
print("\n")

sort_by_name = cities.reindex([2, 0, 1])
print(sort_by_name)
print("\n")

city_name_shuffle = cities.reindex(np.random.permutation(cities.index))
print(city_name_shuffle)
print("\n")

###########################################################################################
"""
Exercise #2
The reindex method allows index values that are not in the original DataFrame's index values. 
Try it and see what happens if you use such values! Why do you think this is allowed?
"""

try_it = cities.reindex([2, 5, 10, 0, 4, 1])
print(try_it)
print("\n")

"""
Using index values that don't originally exist in the dataframe causes all of the columns in each row
to be assigned the value of NaN.

I guess it is permitted as a way of quickly creating new rows and assigning default column values.
This allows new data to be inserted anywhere in the data frame in between existing entries.
"""
###########################################################################################
###########################################################################################

"""

###########################################################################################
Why would one use Pandas rather than the standard data manipulation features provided by NumPy?

Pandas is more suited to data gathering and management while Numpy is more suited to numerical analysis of data.

Pandas allows column labels whereas Numpy assigns index values.

Pandas dataframe allows heterogeneous data whereas Numpy arrays requires homogeneous data.

Etc.

###################################################

Resources Used:

URL: https://www.quora.com/What-are-the-advantages-of-using-Pandas-over-Numpy-for-ML-and-Data-Analysis

###########################################################################################
Under what circumstances would it be useful to reorder/shuffle a Pandas DataFrame?

If you want a random sample of selections from the entire data frame for statistical purposes you could use this
feature with the random.permutation function.

###########################################################################################

"""
