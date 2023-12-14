from funcs import *
from tqdm import tqdm
import matplotlib

@jit(nopython=True)
def F(u):
    # return u**2.0
    # return 1.0
    return u**2


NU_x= Distrib(make_func("triangle_m_f", [-0.5,0.0,0.5]), supp_of_func=(-0.5,0.0,0.5))
# NU_y= Distrib(make_func("triangle_m_f", [-1.0,-0.5,0.0]), supp_of_func=(-1.0,-0.5,0.0))

def NU(x,y,xgrid,ygrid):
    return NU_x(x)*NU_x(y)  

@jit(nopython =True)
def D(x,y,u):
    mu1 = 0.25
    sigma = 0.05
    mu2 = -0.25
    # return u**2.0*(1.0 - np.exp(-0.5*np.square((x-mu1)/sigma)))*(1.0 - np.exp(-0.5*np.square((x-mu2)/sigma)))
    # return u**2.0
    # return u**2
    # return np.exp(x-2+y)
    # return 1.0
    # return 1.0/(u+0.001)
    return 1.0


@jit(nopython=True)
def solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_):
    u = np.copy(u_)    
    end_ = False
    for k in range(Nt-1):
        print(k,Nt)
        u_s = np.copy(u)
        for s in range(1):
            for i in range(1,Nx-1):
                for j in range(1,Ny-1):
                    L_x = (D(x[i+1],y[j],u_s[i+1][j])-D(x[i-1],y[j],u_s[i-1][j]))*(u[i+1][j]-u[i-1][j])/4.0/hx**2 + D(x[i],y[j],u_s[i][j])*(u[i+1][j]-2.0*u[i][j]+u[i-1][j])/hx**2
                    L_y = (D(x[i],y[j+1],u_s[i][j+1])-D(x[i],y[j-1],u_s[i][j-1]))*(u[i][j+1]-u[i][j-1])/4.0/hy**2 + D(x[i],y[j],u_s[i][j])*(u[i][j+1]-2.0*u[i][j]+u[i][j-1])/hy**2

                    u_s[i][j] = u[i][j] + tau*(L_x+ L_y + F(u_s[i][j]))

            for j in range(1,Ny-1):
                u_s[0][j] = u_s[1][j]
                u_s[Nx-1][j] = u_s[Nx-2][j]
            for i in range(1,Nx-1):
                u_s[i][0] = u_s[i][1]
                u_s[i][Ny-1] = u_s[i][Ny-2]
            
            u_s[0][0] = (u_s[0][1]+u_s[1][0])*0.5
            u_s[0][Ny-1] = (u_s[0][Ny-2]+u_s[1][Ny-1])*0.5
            u_s[Nx-1][0] = (u_s[Nx-2][0]+u_s[Nx-1][1])*0.5
            u_s[Nx-1][Ny-1] = (u_s[Nx-2][Ny-1]+u_s[Nx-1][Ny-2])*0.5

        if np.sum(np.isnan(u_s)) > 0 or np.sum(np.isinf(u_s)) > 0 or np.sum(u_s < 0.0) > 0: 
            end_ = True
        if end_:
            return u, k
        u = np.copy(u_s)
    return u,Nt-1


hx = 0.1

ax= -1.0
bx = 1.0
Nx=  int((bx-ax)/hx)+1
xgrid = np.linspace(ax,bx,num=Nx)

hy = 0.1
ay= -1.0
by = 1.0
Ny=  int((by-ay)/hy)+1
ygrid = np.linspace(ay,by,num=Ny)

tau = hx/10**4
T = 0.7
t0 = 0.0
Nt = int((t0+T-t0)/tau) + 1
tgrid = np.linspace(start=t0, stop=t0+T,num=Nt)
t = np.copy(tgrid)
x = np.copy(xgrid)
y= np.copy(ygrid)
u_nu = np.zeros(shape=(len(xgrid),len(ygrid)))

for i in range(len(x)):
    for j in range(len(y)):
        u_nu[i][j] = NU(x[i],y[j],x,y)

u, last_t_index = solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_nu)
# u[-1][0][0]= np.nan
# u[-1][0][Ny-1]= np.nan
# u[-1][Nx-1][0]= np.nan
# u[-1][Nx-1][Ny-1]= np.nan


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, cmap=matplotlib.cm.jet,
                    linewidth=0, antialiased=False)
# ax.plot_surface(X, Y, u[0],
                    # linewidth=0, antialiased=False,alpha=0.2)
fig.colorbar(surf)
ax.set_title(r'$u({})$'.format(last_t_index *tau))
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

