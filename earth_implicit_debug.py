from funcs import *
from tqdm import tqdm
import matplotlib
import string
import plotly.graph_objs as go
from scipy import ndimage, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button, Slider

world_pop = gpd.read_file('./data/pop.geo.json')
world_pop['POP2005']=world_pop['POP2005'].astype(float)
world_pop['area']=  world_pop.to_crs(6933).area.astype(float)*0.000001
world_pop['density'] = (world_pop['POP2005'].div(world_pop['area']))


country_name_index = {}
index_country_name = {}
for i in range(len(world_pop)):
    row_i = world_pop.iloc[i]
    country_name_index.update({row_i['NAME']:i})
    index_country_name.update({i:row_i['NAME']})

polygons_ = get_polygons(world_pop['geometry'])
source_polygons = get_source_polygons(world_pop['geometry'])
N_vec_ = world_pop['POP2005'].values
for i in range(len(N_vec_)):
    if pd.isna(N_vec_[i]):
        N_vec_[i]=1.0
Names_ = world_pop['NAME'].values
Areas_ = world_pop['area'].values

hx = 1.0
hy = 1.0

bounds_ = bounds_by_polygons(polygons_)
xgrid,ygrid = make_grids(hx,hy,bounds_)
Nx = len(xgrid)
Ny = len(ygrid)


masks_per_state = get_binary_mask_per_state_cv2(source_polygons,
                                xgrid, ygrid)
mask_of_earth = CUPmasks(masks_per_state)
mask_of_water = np.logical_not(mask_of_earth)

masks_with_N = [masks_per_state[i].astype(np.float64)*N_vec_[i] for i in range(len(masks_per_state))]

mask_of_earth_N = None 
for i in range(len(masks_with_N)):
    if mask_of_earth_N is None:
        mask_of_earth_N = masks_with_N[i]
    else:
        mask_of_earth_N += masks_with_N[i]

cities = pd.read_csv('./data/worldcities.csv')

cities_mask = np.zeros(shape=(len(xgrid),len(ygrid)),dtype=np.float64)
for i in range(len(cities)):
    x = cities.iloc[i]['lng']
    y = cities.iloc[i]['lat']
    z = cities.iloc[i]['population']
    xi = int((x-xgrid[0])/(xgrid[-1]-xgrid[0])*(len(xgrid)-1))
    yi = int((y-ygrid[0])/(ygrid[-1]-ygrid[0])*(len(ygrid)-1))
    if pd.isna(z):
        z = 1.0
    cities_mask[xi][yi] +=  z

poly_per_state = get_polygons_per_state(world_pop['geometry'])


borders_per_state = get_binary_mask_per_state(poly_per_state,xgrid,ygrid,hx,hy)
all_borders_ = CUPmasks(borders_per_state)
# all_borders_ = ndimage.median_filter(all_borders_, size=2)
kmask = 1.0-np.logical_or(all_borders_.astype(bool), mask_of_water).astype(np.float64)



u_nu = np.zeros(shape=(Nx,Ny))
total_unaccounted = 0.0
total_sum_ = 0.0
internal_mask_per_state = []
internal_index_cnames_ = {}
internal_cnames_index_ = {}
tmp_ind = 0
for i in range(len(masks_per_state)):
    state_mask = masks_per_state[i].astype(bool)
    state_border_mask = borders_per_state[i].astype(bool)
    internal_mask_ = np.logical_and(np.logical_not(state_border_mask), state_mask).astype(np.float64)
    border_people_amt_ = np.sum(cities_mask*state_border_mask.astype(np.float64))
    vertex_amt_internal_ = np.sum(internal_mask_)
    if vertex_amt_internal_ ==0.0:
        total_unaccounted += border_people_amt_
        continue
    total_sum_ += border_people_amt_
    internal_people_ = internal_mask_*cities_mask + border_people_amt_/vertex_amt_internal_*internal_mask_
    u_nu += internal_people_
    internal_mask_per_state.append(internal_mask_)
    internal_index_cnames_.update({tmp_ind:index_country_name[i]})
    internal_cnames_index_.update({index_country_name[i]:tmp_ind})
    tmp_ind +=1
all_internal_masks_ = CUPmasks(internal_mask_per_state)
where_not_zero = u_nu !=0.0
vertex_amnt_where_not_zero = np.sum(where_not_zero)
u_nu[where_not_zero] += total_unaccounted/vertex_amnt_where_not_zero
u_nu[where_not_zero] = u_nu[where_not_zero]/np.sum(u_nu)*np.sum(N_vec_)

@jit(nopython=True)
def F(uij, current_sum):
    return 0.0

@jit(nopython =True)
def D(x,y,u,kmask,xindex,yindex):
    return kmask[xindex][yindex]
# @jit(nopython =True)
# def D(x,y,u,kmask,xindex,yindex,F_at_point):
#     return kmask[xindex][yindex]*F_at_point/10**5

@jit(nopython=True)
def solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_,years_,water_mask_,earth_,kmask,intmask,borders_mask):
    solutions = np.zeros(shape=(len(years_),Nx,Ny))
    solutions[0] = np.copy(u_)
    u = np.copy(u_)    
    B = np.zeros(shape=(Nx,))
    A = np.zeros(shape=(Nx, Nx))

    By = np.zeros(shape=(Ny,))
    Ay = np.zeros(shape=(Ny, Ny))
    end_ = False
    l_ = 1
    u_next = np.zeros(shape=(Nx,Ny))
    u_tau = np.zeros(shape=(Nx,Ny))
    iters_ = 5
    for k in range(Nt-1):
        u_s = np.copy(u)       
        current_sum = np.sum(u_s)
        if np.isnan(current_sum) or np.isinf(current_sum):
            return solutions[:l_]
        print(k,Nt, current_sum/10**9) 
        for j in range(1,Ny-1):
            # B = np.zeros(shape=(Nx,))
            # A = np.zeros(shape=(Nx, Nx))
            for s in range(iters_):
                u_s = None
                if s == 1:
                    u_s = np.copy(u) 
                else:
                    u_s = np.copy(u_next)
                A[0][0] = -1.0/hx
                A[0][1] = 1.0/hx
                A[Nx-1][Nx-1] = 1.0/hx 
                A[Nx-1][Nx-2] = -1.0/hx
                    
                # A[0][0] = 1.0
                # A[0][1] = 0.0
                # A[Nx-1][Nx-1] = 1.0
                # A[Nx-1][Nx-2] = 0.0

                for i in range(1,Nx-1):
                    if borders_mask[i][j] == 1:
                        if intmask[i+1][j] == 1:
                            A[i][i-1] = 0.0
                            A[i][i] = -1.0/hx
                            A[i][i+1] = 1.0/hx
                            continue
                        elif intmask[i-1][j] == 1:
                            A[i][i-1] = -1.0/hx
                            A[i][i] = 1.0/hx
                            A[i][i+1] = 0.0
                            continue
                    A[i][i-1] = 1.0/4.0/hx**2*(D(x[i+1],y[j],u_s[i+1][j],kmask,i+1,j)-D(x[i-1],y[j],u_s[i-1][j],kmask,i-1,j)) - 1.0/hx**2*D(x[i],y[j],u_s[i][j],kmask,i,j)
                    A[i][i]   = 2.0/tau+2.0/hx**2*D(x[i],y[j],u_s[i][j],kmask,i,j)
                    A[i][i+1] = -1.0/4.0/hx**2*(D(x[i+1],y[j],u_s[i+1][j],kmask,i+1,j) - D(x[i-1],y[j],u_s[i-1][j],kmask,i-1,j)) - 1.0/hx**2*D(x[i],y[j],u_s[i][j],kmask,i,j)
                B[0] = 0.0
                B[Nx-1] = 0.0
                for i in range(1,Nx-1):
                    if borders_mask[i][j] == 1:
                        B[i] = 0.0
                    else:
                        B[i] = F(u_s[i][j],current_sum)+2.0/tau*u[i][j] + \
                            1.0/(4.0*hy**2)*(D(x[i],y[j+1],u[i][j+1],kmask,i,j+1)-D(x[i],y[j-1],u[i][j-1],kmask,i,j-1))*(u[i][j+1]-u[i][j-1])+\
                            1.0/(hy**2)*D(x[i],y[j],u[i][j],kmask,i,j)*(u[i][j+1]-2.0*u[i][j]+u[i][j-1])
                u_next[:,j] = progonka(A,B)

        u_tau_pola = np.copy(u_next)
        for i in range(1,Nx-1):

            for s in range(iters_):
                u_s = None
                if s == 1:
                    u_s = np.copy(u_tau_pola) 
                else:
                    u_s = np.copy(u_tau)
                Ay[0][0] = -1.0/hy
                Ay[0][1] = 1.0/hy
                Ay[Ny-1][Ny-1] = 1.0/hy 
                Ay[Ny-1][Ny-2] = -1.0/hy
                # Ay[0][0] = 1.0
                # Ay[0][1] = 0.0
                # Ay[Ny-1][Ny-1] = 1.0
                # Ay[Ny-1][Ny-2] = 0.0
                for j in range(1,Ny-1):
                    if borders_mask[i][j] == 1:
                        if intmask[i][j+1] == 1:
                            Ay[j][j-1] = 0.0
                            Ay[j][j] = -1.0/hy
                            Ay[j][j+1] = 1.0/hy
                            continue
                        elif intmask[i][j-1] == 1:
                            Ay[j][j-1] = -1.0/hy
                            Ay[j][j] = 1.0/hy
                            Ay[j][j+1] = 0.0
                            continue
                    Ay[j][j-1] = 1.0/(4.0*hy**2)*D(x[i],y[j+1],u_s[i][j+1],kmask,i,j+1) - 1.0/(4.0*hy**2)*D(x[i],y[j-1],u_s[i][j-1],kmask,i,j-1) - 1.0/hy**2*D(x[i],y[j],u_s[i][j],kmask,i,j)
                    Ay[j][j]   = 2.0/tau+2.0/(hy**2)*D(x[i],y[j],u_s[i][j],kmask,i,j)
                    Ay[j][j+1] = -1.0/(4.0*hy**2)*D(x[i],y[j+1],u_s[i][j+1],kmask,i,j+1) + 1.0/(4.0*hy**2)*D(x[i],y[j-1],u_s[i][j-1],kmask,i,j-1) - 1.0/hy**2*D(x[i],y[j],u_s[i][j],kmask,i,j)

                By[0] = 0.0
                By[Ny-1] = 0.0
                for j in range(1,Ny-1):
                    if borders_mask[i][j] == 1:
                        By[j] = 0.0
                    else:
                        By[j] = F(u_s[i][j],current_sum)+2.0/tau*u_tau_pola[i][j] + \
                            1.0/(4.0*hx**2)*(D(x[i+1],y[j],u_tau_pola[i+1][j],kmask,i+1,j)-D(x[i-1],y[j],u_tau_pola[i-1][j],kmask,i-1,j))*(u_tau_pola[i+1][j]-u_tau_pola[i-1][j])+\
                            1.0/(hx**2)*D(x[i],y[j],u_tau_pola[i][j],kmask,i,j)*(u_tau_pola[i+1][j]-2.0*u_tau_pola[i][j]+u_tau_pola[i-1][j])
                u_tau[i,:] = progonka(Ay,By)

        u = np.copy(u_tau)
        for ye_ in years_:
            if ye_ == t[k+1]:
                solutions[l_] = np.copy(u_tau)
                l_ +=1  
    return solutions


print(Nx,Ny)

tau = hx/10**3
T = 1.0
t0 = 2005.0
Nt = int((t0+T-t0)/tau) + 1
years_ = np.arange(t0,T+t0)
tgrid = np.linspace(start=t0, stop=t0+T,num=Nt)
t = np.copy(tgrid)
x = np.copy(xgrid)
y = np.copy(ygrid)
u_vec = solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_nu,years_,mask_of_water.astype(np.float64),mask_of_earth.astype(np.float64),kmask,all_internal_masks_,all_borders_)