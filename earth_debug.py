from funcs import *
from tqdm import tqdm
import matplotlib
import string
import plotly.graph_objs as go

hx = 1.0
hy = 1.0
tau = hx/10**4
T = 2.0
t0 = 2005.0

@jit(nopython=True)
def F(t,x,y,u):
    return 0.0
@jit(nopython =True)
def D(x,y,u,kmask,xindex,yindex):
    return kmask[xindex][yindex]

@jit(nopython=True)
def solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_, water_mask_,earth_,kmask):
    solutions = np.zeros(shape=(Nt,Nx,Ny))
    solutions[0] = np.copy(u_)
    u = np.copy(u_)
    end_ = False
    iters_ = 1
    # N = Nx*Ny
    # A = np.zeros(shape=(N,N))
    for k in range(Nt-1):
        print(k,Nt, np.sum(u)/10**9)
        u_s = np.copy(u)
        for s in range(iters_):
            for i in range(1,Nx-1):
                for j in range(1,Ny-1):
                    # L_x = (D(x[i+1],y[j],u_s[i+1][j],kmask,i+1,j)-D(x[i-1],y[j],u_s[i-1][j],kmask,i-1,j))*(u[i+1][j]-u[i-1][j])/4.0/hx**2 + D(x[i],y[j],u_s[i][j],kmask,i,j)*(u[i+1][j]-2.0*u[i][j]+u[i-1][j])/hx**2
                    # L_y = (D(x[i],y[j+1],u_s[i][j+1],kmask,i,j+1)-D(x[i],y[j-1],u_s[i][j-1],kmask,i,j-1))*(u[i][j+1]-u[i][j-1])/4.0/hy**2 + D(x[i],y[j],u_s[i][j],kmask,i,j)*(u[i][j+1]-2.0*u[i][j]+u[i][j-1])/hy**2
                    L_x = 1.0/hx**2*(u[i+1][j]-u[i][j])*np.sqrt(D(x[i+1],y[j],u_s[i+1][j],kmask,i+1,j)*D(x[i],y[j],u_s[i][j],kmask,i,j)) - 1.0/hx**2*(u[i][j]-u[i-1][j])*np.sqrt(D(x[i],y[j],u_s[i][j],kmask,i,j)*D(x[i-1],y[j],u_s[i-1][j],kmask,i-1,j))
                    L_y = 1.0/hy**2*(u[i][j+1]-u[i][j])*np.sqrt(D(x[i],y[j+1],u_s[i][j+1],kmask,i,j+1)*D(x[i],y[j],u_s[i][j],kmask,i,j)) - 1.0/hy**2*(u[i][j]-u[i][j-1])*np.sqrt(D(x[i],y[j],u_s[i][j],kmask,i,j)*D(x[i],y[j-1],u_s[i][j-1],kmask,i,j-1))
                    u_s[i][j] = u[i][j] + tau*(L_x+ L_y + F(t[k],x[i],y[j],u_s[i][j]))
                    # if u_s[i][j] < 0.0:
                    #     print(1)
                    #     fig,ax = plt.subplots()
                    #     ax.scatter(y, u_s[i,:],label='N')
                    #     ax.scatter(y, kmask[i,:],label=r'$\chi$')
                    #     ax.scatter([y[j]],[0.0],label='bad point')

                    #     ax.set_yscale('symlog')
                    #     for p_ in range(len(y)):
                    #         ax.axvline(y[p_],c='k',alpha=0.3)
                    #     ax.legend()
                    #     plt.show()

        is_nan = np.sum(np.isnan(u_s)) > 0
        is_inf = np.sum(np.isinf(u_s)) > 0
        is_neg = np.sum(u_s < 0.0) > 0
        if is_nan or is_inf or is_neg: 
            end_ = True
            if is_nan:
                print('nan')
            if is_inf:
                print('inf')
            if is_neg:
                print('neg')
        if end_:
            return solutions[:k+1]
        u = np.copy(u_s)
        solutions[k+1] = np.copy(u_s) 
    return solutions

world_pop = gpd.read_file('./data/pop.geo.json')
world_pop['POP2005']=world_pop['POP2005'].astype(float)
world_pop['area']=  world_pop.to_crs(6933).area.astype(float)*0.000001
world_pop['density'] = (world_pop['POP2005'].div(world_pop['area']))

country_name_index = {}
for i in tqdm(range(len(world_pop))):
    row_i = world_pop.iloc[i]
    country_name_index.update({row_i['NAME']:i})

polygons_ = get_polygons(world_pop['geometry'])
source_polygons = get_source_polygons(world_pop['geometry'])
N_vec_ = world_pop['POP2005'].values
for i in tqdm(range(len(N_vec_))):
    if pd.isna(N_vec_[i]):
        N_vec_[i]=1.0
Names_ = world_pop['NAME'].values
Areas_ = world_pop['area'].values

bounds_ = bounds_by_polygons(polygons_)
xgrid,ygrid = make_grids(hx,hy,bounds_)


masks_per_state = get_binary_mask_per_state_cv2(source_polygons,
                                xgrid, ygrid)
mask_of_earth = CUPmasks(masks_per_state)
mask_of_water = np.logical_not(mask_of_earth)


masks_with_N = [masks_per_state[i].astype(np.float64)*N_vec_[i] for i in range(len(masks_per_state))]

mask_of_earth_N = None 
for i in tqdm(range(len(masks_with_N))):
    if mask_of_earth_N is None:
        mask_of_earth_N = masks_with_N[i]
    else:
        mask_of_earth_N += masks_with_N[i]

cities = pd.read_csv('./data/worldcities.csv')


cities_mask = np.zeros(shape=(len(xgrid),len(ygrid)),dtype=np.float64)
for i in tqdm(range(len(cities))):
    x = cities.iloc[i]['lng']
    y = cities.iloc[i]['lat']
    z = cities.iloc[i]['population']
    xi = int((x-xgrid[0])/(xgrid[-1]-xgrid[0])*(len(xgrid)-1))
    yi = int((y-ygrid[0])/(ygrid[-1]-ygrid[0])*(len(ygrid)-1))
    if pd.isna(z):
        z = 1.0
    cities_mask[xi][yi] +=  z


poly_per_state = get_polygons_per_state(world_pop['geometry'])
Nx = len(xgrid)
Ny = len(ygrid)
u_nu = np.zeros(shape=(Nx,Ny))
for k in country_name_index:
    country_name_ = k
    c_mask = np.copy(masks_per_state[country_name_index[country_name_]])
    c_N = N_vec_[country_name_index[country_name_]]
    c_grid_count = np.sum(c_mask)
    c_boundary_conditions_mask = np.logical_not(c_mask).astype(np.float64)
    c_cities_vs = cities_mask*c_mask.astype(np.float64)
    c_cities_vs_sum = np.sum(c_cities_vs)
    number_of_cities_in_country_with_non_zero_people=  np.sum(c_cities_vs>0)
    if number_of_cities_in_country_with_non_zero_people == 0:
        continue
    amnt_per_vertex = c_N/c_grid_count
    c_cities_vs = c_cities_vs/np.sum(c_cities_vs)
    c_cities_vs = c_cities_vs*c_N
    u_nu += c_cities_vs

borders_per_state = get_binary_mask_per_state(poly_per_state,xgrid,ygrid,hx,hy)
all_borders_ = CUPmasks(borders_per_state)
kmask = 1.0-np.logical_or(all_borders_.astype(bool), mask_of_water).astype(np.float64)



Nt = int((t0+T-t0)/tau) + 1
tgrid = np.linspace(start=t0, stop=t0+T,num=Nt)
t = np.copy(tgrid)
x = np.copy(xgrid)
y = np.copy(ygrid)
u_vec = solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_nu,mask_of_water.astype(np.float64),mask_of_earth.astype(np.float64),kmask)

yaers_ = np.arange(t0,T+t0)
solutions_ = []
for i in range(len(u_vec)):
    t_ = tgrid[i]
    if np.isin(t_, yaers_):
        solutions_.append(u_vec[i])

from matplotlib.widgets import Button, Slider

fig, ax = plt.subplots()
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.set_size_inches(16,9)
cax = ax.matshow(solutions_[0].T,cmap=matplotlib.cm.jet,norm=matplotlib.colors.LogNorm())
ax.set_title(f'{int(yaers_[0])} {np.round(np.sum(solutions_[0])/10**9,1)} млрд')
ax.invert_yaxis()
fig.subplots_adjust(left=0.0, bottom=0.0)
# cbar = fig.colorbar(cax)
fig.tight_layout()

axfreq = fig.add_axes([0.1, 0.0, 0.5, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='',
    valmin=yaers_[0],
    valmax=yaers_[-1],
    valinit=yaers_[0]
)

def update(val):
    # ax.matshow(solutions_[0].T,cmap=matplotlib.cm.jet,norm=matplotlib.colors.LogNorm())
    cax.set_data(solutions_[int(val- yaers_[0])].T)
    ax.set_title(f'{int(val)} {np.round(np.sum(solutions_[int(val- yaers_[0])])/10**9,1)} млрд')
    fig.canvas.draw_idle()
    # plt.draw()

freq_slider.on_changed(update)
plt.show()