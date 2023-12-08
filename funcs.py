import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import json
from pprint import pprint
import pandas as pd
from shapely.geometry import Polygon
import shapely


def get_polygons(col_):

    polys_ = []
    for i in range(len(col_)):
        try:
            xx, yy = col_.iloc[i].exterior.coords.xy
            xx = xx.tolist()
            yy = yy.tolist()
            polys_.append([xx,yy])
        except:
            geoms_ = col_.iloc[i].geoms
            for j in range(len(geoms_)):
                xx, yy = geoms_[j].exterior.coords.xy
                xx = xx.tolist()
                yy = yy.tolist()
                polys_.append([xx,yy])
    return polys_

def bounds_by_polygons(polygons_):
    polygon_ = polygons_[0]
    xx = polygon_[0]
    yy = polygon_[1] 
    a_x = np.min(xx)
    b_x = np.max(xx)
    a_y = np.min(yy)
    b_y = np.max(yy)
    for i in range(len(polygons_)):
        polygon_ = polygons_[i]
        xx = polygon_[0]
        yy = polygon_[1]
        a_x = np.minimum(np.min(xx),a_x)
        b_x = np.maximum(np.max(xx),b_x)
        a_y = np.minimum(np.min(yy),a_y)
        b_y = np.maximum(np.max(yy),b_y)
    return [[a_x,b_x],[a_y,b_y]]

def make_grids(hx,hy, bounds_):
    Nx = int((bounds_[0][1]-bounds_[0][0])/hx) + 1 
    Ny = int((bounds_[1][1]-bounds_[1][0])/hy) + 1
    xgrid = np.linspace(start=bounds_[0][0],stop=bounds_[0][1],num=Nx)
    ygrid = np.linspace(start=bounds_[1][0],stop=bounds_[1][1],num=Ny)
    return xgrid,ygrid
    

def make_masks_by_polygons_(polygons_):
    bounds_ = bounds_by_polygons(polygons_)

