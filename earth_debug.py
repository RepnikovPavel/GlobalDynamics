from funcs import *
from tqdm import tqdm
import matplotlib
import string
import plotly.graph_objs as go
from scipy import ndimage, datasets
from matplotlib.widgets import Button, Slider


hx = 1.0
hy = 1.0

world_pop = gpd.read_file('./data/pop.geo.json')
world_pop['POP2005']=world_pop['POP2005'].astype(float)
world_pop['area']=  world_pop.to_crs(6933).area.astype(float)*0.000001
world_pop['density'] = (world_pop['POP2005'].div(world_pop['area']))

country_name_index = {}
for i in range(len(world_pop)):
    row_i = world_pop.iloc[i]
    country_name_index.update({row_i['NAME']:i})

polygons_ = get_polygons(world_pop['geometry'])
source_polygons = get_source_polygons(world_pop['geometry'])
N_vec_ = world_pop['POP2005'].values
for i in range(len(N_vec_)):
    if pd.isna(N_vec_[i]):
        N_vec_[i]=1.0
Names_ = world_pop['NAME'].values
Areas_ = world_pop['area'].values

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

where_not_zero = u_nu !=0.0
vertex_amnt_where_not_zero = np.sum(where_not_zero)
u_nu[where_not_zero] += total_unaccounted/vertex_amnt_where_not_zero
u_nu[where_not_zero] = u_nu[where_not_zero]/np.sum(u_nu)*np.sum(N_vec_)

@jit(nopython=True)
def F(uij, current_sum):
    C0 = 279.0*10**9
    return uij*current_sum/C0

@jit(nopython =True)
def D(x,y,u,kmask,xindex,yindex):
    return kmask[xindex][yindex]

@jit(nopython =True)
def solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_,years_,water_mask_,earth_,kmask):
    solutions = np.zeros(shape=(len(years_),Nx,Ny))
    solutions[0] = np.copy(u_)
    u = np.copy(u_)
    end_ = False
    l_ = 1
    u_next = np.zeros(shape=(Nx,Ny))
    # d_m = 
    # sources_ = []
    # Lxvec =[]
    for k in range(Nt-1):
        # if np.mod(int(float(k+1)/float(Nt-2)*100),10) == 0:
        current_sum = np.sum(u)
        if np.isnan(current_sum) or np.isinf(current_sum):
            return solutions[:l_]
        print(k,Nt, current_sum/10**9)
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                # if u[i][j] < 0.0:
                #     u__ = u[i][j]
                #     print(1)
                sq1x = D(x[i+1],y[j],u[i+1][j],kmask,i+1,j)*D(x[i],y[j],u[i][j],kmask,i,j)
                sq2x = D(x[i],y[j],u[i][j],kmask,i,j)*D(x[i-1],y[j],u[i-1][j],kmask,i-1,j)
                sq1y = D(x[i],y[j+1],u[i][j+1],kmask,i,j+1)*D(x[i],y[j],u[i][j],kmask,i,j)
                sq2y = D(x[i],y[j],u[i][j],kmask,i,j)*D(x[i],y[j-1],u[i][j-1],kmask,i,j-1)
                # if np.isnan(sq1x) or np.isnan(sq2x) or np.isnan(sq1y) or np.isnan(sq1y) or np.isinf(sq1x) or np.isinf(sq2x) or np.isinf(sq1y) or np.isinf(sq1y):
                #     print(sq1x,sq2x)
                #     print(sq1y,sq2y)

                # if sq1x< 0.0 or  sq2x < 0.0 or sq1y<0.0 or  sq2y < 0.0:
                #     print(sq1x,sq2x)
                #     print(sq1y,sq2y)

                L_x = 1.0/hx**2*(u[i+1][j]-u[i][j])*np.sqrt(sq1x) - 1.0/hx**2*(u[i][j]-u[i-1][j])*np.sqrt(sq2x)
                L_y = 1.0/hy**2*(u[i][j+1]-u[i][j])*np.sqrt(sq1y) - 1.0/hy**2*(u[i][j]-u[i][j-1])*np.sqrt(sq2y)
                # source_ = F(t[k],x[i],y[j],u,i,j)
                source_ = F(u[i][j],current_sum)
                # sources_.append(source_)
                # Lxvec.append(L_x)
                u_next[i][j] = np.maximum(u[i][j] + tau*(L_x+ L_y + source_),0.0)

        # fig,ax = FloatDistr(sources_)
        # ax.set_title('F')
        # ax.set_xscale('log')
        # fig,ax = FloatDistr(Lxvec)
        # ax.set_title('D')
        # ax.set_xscale('symlog')
        # plt.show()
        # break
    
        # is_nan = np.sum(np.isnan(u_next)) > 0
        # is_inf = np.sum(np.isinf(u_next)) > 0
        # is_neg = np.sum(u_next < 0.0) > 0
        # if is_nan or is_inf or is_neg: 
        #     end_ = True
        #     if is_nan:
        #         print('nan')
        #     if is_inf:
        #         print('inf')
        #     if is_neg:
        #         print('neg')
        # if end_:
        # return solutions[:l_]
        u = np.copy(u_next)
        for ye_ in years_:
            if ye_ == t[k+1]:
                solutions[l_] = np.copy(u_next)
                l_ +=1  
    return solutions

tau = hx/10**3
T = 40.0
t0 = 2005.0
Nt = int((t0+T-t0)/tau) + 1
years_ = np.arange(t0,T+t0)
tgrid = np.linspace(start=t0, stop=t0+T,num=Nt)
t = np.copy(tgrid)
x = np.copy(xgrid)
y = np.copy(ygrid)
u_vec = solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_nu,years_,mask_of_water.astype(np.float64),mask_of_earth.astype(np.float64),kmask)


years_ = years_[:len(u_vec)]
fig, ax = plt.subplots()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.set_size_inches(16,9)
# cax = ax.matshow(u_vec[0].T,cmap=matplotlib.cm.jet,norm=matplotlib.colors.LogNorm())
cax = ax.matshow(u_vec[0].T,cmap=matplotlib.cm.jet,norm=matplotlib.colors.LogNorm())
ax.set_title(f'{int(years_[0])} {np.round(np.sum(u_vec[0])/10**9,1)} млрд')
ax.invert_yaxis()
fig.subplots_adjust(left=0.0, bottom=0.0)
# cbar = fig.colorbar(cax)
fig.tight_layout()

axfreq = fig.add_axes([0.1, 0.0, 0.5, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='',
    valmin=years_[0],
    valmax=years_[-1],
    valinit=years_[0]
)

def update(val):
    # ax.matshow(u_vec[int(val- years_[0])].T,cmap=matplotlib.cm.jet,norm=matplotlib.colors.LogNorm())
    cax.set_data(u_vec[int(val- years_[0])].T)
    ax.set_title(f'{int(val)} {np.round(np.sum(u_vec[int(val- years_[0])])/10**9,1)} млрд')
    fig.canvas.draw_idle()
    # plt.draw()

freq_slider.on_changed(update)
plt.show()