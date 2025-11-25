import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#Import the data from the csv file
futuristic_df = pd.read_csv('futuristic_city_traffic.csv')
real_df = pd.read_csv('traffic_weather_full2020.csv')

#Check the quality of the data imported 
missing_percentage = futuristic_df.isnull().sum() / len(futuristic_df) * 100
duplicates = futuristic_df.duplicated().sum()

print(missing_percentage)
print(f"Number of exact duplicate rows: {duplicates}")

missing_percentage = real_df.isnull().sum() / len(real_df) * 100
duplicates = real_df.duplicated().sum()

print(missing_percentage)
print(f"Number of exact duplicate rows: {duplicates}")