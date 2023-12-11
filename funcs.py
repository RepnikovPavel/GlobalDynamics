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
from typing import Tuple

from scipy.integrate import quad

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

def make_grids(hx,hy,bounds_):
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


def place_gauss_on_grid(number_of_gaussians, bounds ,xgrid,ygrid):
    u = np.zeros(shape=(len(xgrid),len(ygrid)))
    a_x = bounds[0][0]
    b_x = bounds[0][1]
    a_y = bounds[1][0]
    b_y = bounds[1][1]
    mu_x = np.zeros(shape=(number_of_gaussians,))
    mu_y = np.zeros(shape=(number_of_gaussians,))
    for i in range(number_of_gaussians):
        mu_x[i] = np.random.uniform(low=a_x,
                                    high=b_x)
        mu_y[i] = np.random.uniform(low=a_y,
                                    high=b_y)


def FloatDistr(data,fig_size=(4,3),title=''):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0],fig_size[1])
    x = []
    for i in range(len(data)):
        if np.isnan(data[i]):
            continue
        else:
            x.append(data[i])
    u_vs = np.unique(x)

    if len(x) == 0:
        ax.set_title(title +' is empty data')
    elif len(u_vs)==1:
        ax.set_title(title + ' all data is repeated with value: {}'.format(u_vs[0]))
    else:
        x = np.asarray(x)
        q25, q75 = np.percentile(x, [25, 75])
        bins = 0
        if q25==q75:
            bins = np.minimum(100,len(u_vs))
        else:
            bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
            bins = np.minimum(100, round((np.max(x) - np.min(x)) / bin_width))
        nan_rate = np.sum(np.isnan(data))/len(data)
        ax.set_title(title+'. n of unique values {}'.format(len(u_vs)))
        ax.set_xlabel('nan rate {}'.format(nan_rate))
        density,bins = np.histogram(x,bins=bins,density=True)
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density,width=widths)

    return fig,ax

def triangle_m_f(x: float, l_point: float, m_point: float, r_point: float) -> float:
    if x <= l_point:
        return 0.0
    elif x >= r_point:
        return 0.0
    elif l_point < x <= m_point:
        return 1 / (m_point - l_point) * x + l_point / (l_point - m_point)
    elif m_point < x < r_point:
        return 1 / (m_point - r_point) * x + r_point / (r_point - m_point)

def make_func(type_of_func: str, func_params: list):
    """
    type_of_func: "triangle_m_f" only supported
    """
    if type_of_func == "triangle_m_f":
        def m_f(x: float) -> float:
            return triangle_m_f(x, func_params[0], func_params[1], func_params[2])

        return m_f
    else:
        print("smth went wrong in make_func function")
        raise SystemExit

class Distrib:
    support: Tuple[float, float]
    distrib = None
    grid: np.array
    num_of_segments: int
    max_x: float

    def __init__(self, func, supp_of_func: Tuple[float, float], num_of_segments=5,max_x=None):
        integral = quad(func, supp_of_func[0],
                        supp_of_func[1])[0]

        def distr(x: float):
            return 1 / integral * func(x)
        self.max_x = max_x
        self.distrib = distr
        self.support = supp_of_func
        self.num_of_segments = num_of_segments
        self.grid = np.linspace(start=supp_of_func[0], stop=supp_of_func[1], num=self.num_of_segments+1)

    def __call__(self, x: float):
        return self.distrib(x)

# @jit(nopython=True)
# def progonka(A,B):
#     N = len(A)
#     P = np.zeros(shape=(N,))
#     Q = np.zeros(shape=(N,))
#     x = np.zeros(shape=(N,))
#     c0 = A[0][1]
#     b0 = -A[0][0]
#     d0 = B[0]
#     P[0] = c0/b0
#     Q[0] = -d0/b0
#     for i in range(1,N-1):
#         a =  A[i][i-1]
#         b = -A[i][i]
#         c = A[i][i+1]
#         d = B[i]
#         P[i] = c/(-a*P[i-1]+b)
#         Q[i] = (-d+a*Q[i-1])/(-a*P[i-1]+b)

#     # a =  A[i][i-1]
#     # b = -A[i][i]
#     # d = B[i]
#     # Q[N-1] = (-d+a*Q[N-2])/(-a*P[N-2]+b)

#     for i in range(N-1,0,-1):
#         x[i-1] = P[i-1]*x[i] + Q[i-1]
#     return x
    
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

@jit(nopython=True)
def progonka(A,B):
    n = len(B)
    d = B
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)
    for i in range(n):
        b[i] = A[i][i]
    for i in range(1,n):
        a[i-1] = A[i][i-1]
        c[i-1] = A[i-1][i]
    w= np.zeros(n-1)
    g= np.zeros(n)
    p = np.zeros(n)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p

