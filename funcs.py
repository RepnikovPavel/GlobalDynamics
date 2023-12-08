import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import json
from pprint import pprint
import pandas as pd
from shapely.geometry import Polygon,MultiPolygon
import shapely
from numba import jit 
import numpy as np
import cv2
from shapely.geometry import box
from tqdm import tqdm


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


def get_polygons_per_state(geometries_):
    # state index, coord index, value index
    polys_ = []
    for i in range(len(geometries_)):
        state_geometries = []
        try:
            xx, yy = geometries_.iloc[i].exterior.coords.xy
            xx = xx.tolist()
            yy = yy.tolist()
            state_geometries.append([xx,yy])
        except:
            geoms_ = geometries_.iloc[i].geoms
            for j in range(len(geoms_)):
                xx, yy = geoms_[j].exterior.coords.xy
                xx = xx.tolist()
                yy = yy.tolist()
                state_geometries.append([xx,yy])
        polys_.append(state_geometries)
    return polys_



# def fill_mask(mask_):
#     nx = len(mask_)
#     ny = len(mask_[0])
#     for i in range(nx):
#         fill_pos_ = []
#         for j in range(ny):
#             if mask_[i][j] == 1:
#                 fill_pos_.append(j)
         


#         fill_ = False
#         for j in range(ny):
#             if mask_[i][j] == 1:
#                 fill_ = True


def get_binary_mask_per_state(polygorns_per_state,
                              xgrid, ygrid,hx,hy):
    masks_ = []
    for i in range(len(polygorns_per_state)):
        mask_per_state = np.zeros(shape=(len(xgrid),len(ygrid)),dtype=np.intc)  
        for j in range(len(polygorns_per_state[i])):
            jth_poly_of_ith_state = polygorns_per_state[i][j]
            xx = jth_poly_of_ith_state[0]
            yy = jth_poly_of_ith_state[1]
            for k in range(len(xx)):
                xi = np.intc(np.rint((xx[k]-xgrid[0])/hx))
                yi = np.intc(np.rint((yy[k]-ygrid[0])/hy))
                mask_per_state[xi][yi] = 1
        
        masks_.append(mask_per_state)
    return masks_

def get_grid_points_by_mask(mask_, xgrid,ygrid):
    xx = []
    yy = []
    for i in range(len(mask_)):
        for j in range(len(mask_[i])):
            if mask_[i][j] == 1:
                xx.append(xgrid[i])
                yy.append(ygrid[j])
    return xx,yy

def get_source_polygons(geometries_):
    # state index, coord index, value index
    polys_ = []
    for i in range(len(geometries_)):
        state_geometries = []
        if type(geometries_.iloc[i])==Polygon:
            geo_i =  Polygon([(x,y) for x, y in zip(*geometries_.iloc[i].exterior.coords.xy)])
            state_geometries.append(geo_i)
        elif type(geometries_.iloc[i])==MultiPolygon:
            geoms_ = geometries_.iloc[i].geoms
            for j in range(len(geoms_)):
                geo_i =  Polygon([(x,y) for x, y in zip(*geoms_[j].exterior.coords.xy)])
                state_geometries.append(geo_i)
        else:
            print('unknown error')
            print(type(geometries_.iloc[i]))
        polys_.append(state_geometries)
    return polys_


def get_binary_mask_per_state_cv2(source_polygons_,
                                xgrid, ygrid):
    masks= []
    for i in tqdm(range(len(source_polygons_))):
        mask_per_state = None 
        for j in range(len(source_polygons_[i])):
            mask = np.zeros([len(xgrid), len(ygrid)],dtype=np.intc)
            polygon = source_polygons_[i][j]
            points = [[(y-ygrid[0])/(ygrid[-1]-ygrid[0])*(len(ygrid)-1),(x-xgrid[0])/(xgrid[-1]-xgrid[0])*(len(xgrid)-1)] for x, y in zip(*polygon.boundary.coords.xy)]
            mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=1)
            if mask_per_state is None:
                mask_per_state = mask
            else:
                mask_per_state = np.logical_or(mask_per_state,mask)
        masks.append(mask_per_state)
    return masks

def CUPmasks(masks_):
    omask_ = None
    for i in range(len(masks_)):
        if omask_ is None:
            omask_ = np.copy(masks_[i])
        else:
            omask_ = np.logical_or(omask_, masks_[i])
    return omask_.astype(np.intc)