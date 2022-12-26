#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS2500
Project 2
Nicolai Jacobsen, Daniel Vahey and James D'Elia
Universities and their Impact on Boston Real Estate

Our code uses object-oriented design as well as clusters, which allows for
analysis on real estate prices around universities. We will scatter plot every
property and university on a map of Boston, using plotly.express and mapbox. 
In each object, we will have longitude and latitude as attributes which will
allow us to easily plot the data. Each property will also have price as an 
attribute and universities will have student population as an attribute, so 
that we can exclude smaller universities that should not be part of our analysis
"""
import pandas as pd
from project_2_classes import Property, University
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import plotly.express as px
px.set_mapbox_access_token('pk.eyJ1Ijoibmljb2xhaTI0MDYiLCJhIjoiY2wyZTA0eWl5' + 
                           'MDJvZzNkcWxpY3VkZDV1bSJ9.LKPcWeroAsiOv9cCePGUDQ')

UNIVERSITIES = 'Colleges_and_Universities.csv'
REAL_ESTATE = 'zillow-boston.csv'

def read_csv(filename):
    """
    Function to read in csv files using Pandas
    
    Parameters:
    -----------
    filename : str 
        the name of the file
    Returns:
    -------
    data : df
        contains data from csv
    """
    # Read in csv using pandas
    data = pd.read_csv(filename)
    
    return data 

def create_property_objects(df):
    """
    Function to create objects under the property class with relevant 
    attributes

    Parameters
    ----------
    df : dataframe
        contains data for every property
    Returns
    -------
    object_lst : list of objects
        contains every property object as a list of objects
    """
    # create empty list
    object_lst = []
    # iterate over each row in the df
    for i, row in df.iterrows():
        # find lat and long
        lat = float(row['latitude'])
        long = float(row['longitude'])
        # exclude properties outside of Boston or wrong coordinates
        if long < -74:
            continue
        # exclude properties without coordinates
        if math.isnan(lat) or math.isnan(long):
            continue
        # find the price of the property
        price = float(row['price'])
        # Create the property object under the Property class
        prop_obj = Property(lat, long, price)
        # append to the list
        object_lst.append(prop_obj)
    
    return object_lst

def create_uni_objects(df):
    """
    Function to create objects for every university with relevant attributes

    Parameters
    ----------
    df : dataframe
        contains university data

    Returns
    -------
    object_lst : list of objects
        contains every university as a list of objects
    """
    # create empty list
    object_lst = []
    # iterate over each row in df
    for i, row in df.iterrows():
        # find latitude
        lat = float(row['Latitude'])
        # exclude if latitude is outside of boston
        if lat < 40:
            continue
        # find logitude
        long = float(row['Longitude'])
        # find name of uni
        name = row['Name']
        # Abbreviate the name so that it does not take up excessive space
        name_lst = name.split()
        abbr = []
        # Take the first letter of every word in the name in the abbreviation
        for word in name_lst:
            first_letter = word[0]
            abbr.append(first_letter)
        # Join the letters
        name = ''.join(abbr)
        # find the number of students
        num_students = int(row['NumStudent'])
        # Exclude universities with less than 2000 students
        if num_students < 2000:
            continue
        # Create the university object
        uni_obj = University(name, lat, long, num_students)
        # append to list
        object_lst.append(uni_obj)
    
    return object_lst

def distance(point1, point2):
    """
    Function to find the euclidean distance between objects

    Parameters
    ----------
    point1 : obj
        object with coordinates
    point2 : obj
        object with coordinates

    Returns
    -------
    euc_dist : float
        the euclidean distance between the coordinates
    """
    # create tuples containing the coordinates
    point_1 = (point1.lat, point1.long)
    point_2 = (point2.lat, point2.long)
    # use Math to calculate the euclidean distance
    euc_dist = math.dist(point_1, point_2)
    return euc_dist

def find_closest(k, uni, prop_lst):
    """
    Function to find the k closest properties to a University object and return
    a list with cluster of k closest objects

    Parameters
    ----------
    k : int
        the number of properties in the cluster
    uni : obj
        University object
    prop_lst : list of objects
        list of property objects

    Returns
    -------
    distances[:k] : lst
        list with k closest objects
    """
    distances = []
    # iterate over properties
    for prop in prop_lst:
        # find the euclidean distance to the university
        euc_dist = distance(uni, prop)
        # add as attribute to object
        prop.euc_dist = euc_dist
        # check if distance is less than the distance to the current closest
        # university
        if euc_dist < prop.uni_dist:
            # If so, set as new closest university and set as an attribute
            prop.uni_dist = euc_dist
            prop.closest_uni = uni
        # append to list of properties
        distances.append(prop)
    # sort by euclidean distance using the __lt__ in the class
    distances.sort()
    # return the k first objects in the list, which is the cluster of k
    return distances[:k]
        
def plot_objects(lst):
    """
    Function to plot objects

    Parameters
    ----------
    lst : list of objects
    """
    lats = []
    longs = []
    color = []
    sizes = []
    # iterate over each object in list
    for obj in lst:
        # find lat, long and append to list
        lat = obj.lat
        lats.append(lat)
        long = obj.long
        longs.append(long)
        # set color and size depending on class
        col = ''
        if isinstance(obj, Property):
            col = 'red'
            size = 0.5
        else:
            col = 'blue'
            size = 5
        color.append(col)
        sizes.append(size)
    # create dict from lists
    d = {'latitude': lats, 'longitude': longs, 'color': color, 'size': sizes}
    # create dataframe from dict
    df = pd.DataFrame(d)
    # use Mapbox to scatter plot the object on a map, for us this only produced
    # an output in Jupyter notebook
    fig = px.scatter_mapbox(df, lat = 'latitude', lon = 'longitude', 
                            color = 'color', size = 'size', 
                            title = 'Scatter plot of universities and ' + 
                            'properties in Boston')
    fig.show()
def find_clusters(uni_obj_lst, property_obj_lst):
    """
    Function to get the clusters for each uni and get the data for each cluster

    Parameters
    ----------
    uni_obj_lst : list of objects
        List of University objects
    property_obj_lst : list of objects
        List of Property objects

    Returns
    -------
    cluster_dct : dict
        contains the list of properties in the cluster for each uni

    """
    cluster_dct = {}
    # iterate over each university
    for uni in uni_obj_lst:
        # find the 25 closest properties
        cluster = find_closest(25, uni, property_obj_lst)
        price_sum = 0
        # iterate over each property in the cluster
        for prop in cluster:
            # add to price sum
            price_sum += prop.price 
        # find the mean price for the cluster
        mean_price = price_sum / len(cluster)
        # add as attribute to university object
        uni.mean_cluster_price = mean_price
        # create key and value in dict with the cluster
        cluster_dct[str(uni)] = cluster 
        
    return cluster_dct 

def plot_cluster_prices(uni_obj_lst, avg_price):
    """
    Function to bar plot each cluster and show the average price for each 
    university cluster

    Parameters
    ----------
    uni_obj_lst : list of objects
        list of University class objects
    avg_price : float
        The average price of properties in our dataset
    """
    # iterate over each uni
    for uni in uni_obj_lst:
        name = str(uni)
        # find the mean cluster price from the object
        price_mean = uni.mean_cluster_price
        # bar plot the result
        plt.bar(name, price_mean)
        plt.xticks(rotation = 60)
        # plot horizontal line for the average price
        plt.axhline(y = avg_price)
    plt.title('Average cluster prices for each University')
    plt.ylabel('The average price of Real Estate in the Cluster (millions)')
    plt.xlabel('University')
    plt.show()

def plot_linear(property_obj_lst):
    """
    Function to scatter plot the average price of properties when grouped in
    distance intervals to universities, and plot and print the linear 
    regression correlation

    Parameters
    ----------
    property_obj_lst : list of objects
        list of Property class objects
    """
    dist_dict = {}
    price_avg_lst = []
    dist_lst = []
    # create the distance intervals and add as keys to the dict with empty
    # lists as values
    for dist in np.arange(0, 0.10, 0.005):
        dist_dict[dist] = []
    # iterate over each property
    for prop in property_obj_lst:
        # find the distance to the closest uni
        uni_dist = prop.uni_dist
        # find the relevant distance interval for the uni_dist
        for dist in np.arange(0, 0.10, 0.005):
            if (dist - 0.005) < uni_dist < dist:
                # append to the list with the relevant distance interval
                dist_dict[dist].append(prop)
    # iterate over each list for every distance interval
    for dist in np.arange(0, 0.10, 0.005):
        price_sum = 0
        for prop in dist_dict[dist]:
            # add to sum
            price_sum += prop.price
        # check that list is not empty
        if len(dist_dict[dist]) > 0:
            # find the price average for the distance interval
            price_avg = price_sum / len(dist_dict[dist])
            # remove one of the outliers where the price was extremely high
            if price_avg < 3000000:
                # append to list
                price_avg_lst.append(price_avg)
                dist_lst.append(dist)
    # create dict and then dataframe with the price averages
    d = {'dist': dist_lst, 'price_avg': price_avg_lst}
    df = pd.DataFrame(d)
    # scatter plot the data
    plt.scatter(dist_lst, price_avg_lst)
    # produce a regression plot
    sns.regplot(x=d['dist'], y=d['price_avg'])
    plt.title('Average Price of Real Estate vs Distance from University')
    plt.ylabel('Average Price of Real Estate (millions)')
    plt.xlabel('Distance from university')
    plt.show()
    # find the correlation
    print('The correlation between Average price of Real Estate relative to' +
          ' the distance to university')
    print(df.corr())
            
def mean_property_price(obj_lst):
    """
    Function to calculate the average property price for the entire dataset

    Parameters
    ----------
    obj_lst : list of objects
        list of property class objects

    Returns
    -------
    avg_price : float
        the average price in the object list
    """
    price_sum = 0
    # find the price of each property and add to sum
    for obj in obj_lst:
        price = obj.price
        price_sum += price
    # find the average price
    avg_price = price_sum / len(obj_lst)
    print('The average price of real estate properties in our dataset is:')
    print(avg_price)
    
    return avg_price

if __name__ == "__main__":
    # read in the real estate property csv file and create df
    real_estate = read_csv(REAL_ESTATE)
    # read in the university csv file and create df
    universities = read_csv(UNIVERSITIES)
    # create objects for each property and university
    property_obj_lst = create_property_objects(real_estate)
    uni_obj_lst = create_uni_objects(universities)
    # create clusters around each university
    cluster_dct = find_clusters(uni_obj_lst, property_obj_lst)
    # create a combined list with unis and properties for plotting
    total_obj_lst = uni_obj_lst + property_obj_lst
    # scatter plot the properties and universities on a map
    plot_objects(total_obj_lst)
    # find the mean price of properties in our dataset
    avg_price = mean_property_price(property_obj_lst)
    # plot the average cluster prices
    plot_cluster_prices(uni_obj_lst, avg_price)
    # plot the average price for distance intervals with linear regression
    plot_linear(property_obj_lst)
    
    
    