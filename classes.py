#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS2500
Project 2
Nicolai Jacobsen, Daniel Vahey and James D'Elia
Universities and their Impact on Boston Real Estate

CLASSES
"""

class University:
    """ Class to create University objects """
    def __init__(self, name, lat, long, students):
        self.name = name
        # coordinates for plotting and distance calculations
        self.lat = lat
        self.long = long
        self.students = students
    def __str__(self):
        # printing name for universities for ease of identification
        return self.name
        
class Property: 
    def __init__(self, lat, long, price):
        self.lat = lat
        self.long = long
        self.price = price
        # high university distance to start the loop
        self.uni_dist = 1000000
        self.closest_uni = None
    def __lt__(self, other):
        # less than function to sort the property list by distance to 
        # university
        return self.euc_dist < other.euc_dist
