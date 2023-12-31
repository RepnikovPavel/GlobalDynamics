from funcs import *
from tqdm import tqdm
import matplotlib

@jit(nopython=True)
def F(t,x,y,u):
    return np.sin(x)*np.sin(y)*np.exp(-t)

@jit(nopython=True)
def solution(t,x,y):
    return np.sin(x)*np.sin(y)*(np.exp(t)-1.0)*np.exp(-2.0*t)


NU_x= Distrib(make_func("triangle_m_f", [-0.5,0.0,0.5]), supp_of_func=(-0.5,0.0,0.5))
# NU_y= Distrib(make_func("triangle_m_f", [-1.0,-0.5,0.0]), supp_of_func=(-1.0,-0.5,0.0))

def NU(x,y,xgrid,ygrid):
    # return NU_x(x)*NU_y(y)  
    # return NU_x(x)*NU_x(y)  
    return  0.0

@jit(nopython =True)
def D(x,y):
    return 1.0


@jit(nopython=True)
def solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_):
    u = np.copy(u_)    
    end_ = False
    iters_ = 20
    B = np.zeros(shape=(Nx,))
    A = np.zeros(shape=(Nx, Nx))
    By = np.zeros(shape=(Ny,))
    Ay = np.zeros(shape=(Ny, Ny))
    # errors_ = []
    for k in range(Nt-1):
        print(k, Nt-1)
        u_s = np.copy(u)
        # error_ = 0.0
        for s in range(iters_):
            u_tau_pola = np.zeros(shape=(Nx,Ny))
            for j in range(1,Ny-1):
                A[0][0] = 1.0
                A[0][1] = 0.0
                A[Nx-1][Nx-1] = 1.0
                A[Nx-1][Nx-2] = 0.0
                for i in range(1,Nx-1):
                    A[i][i-1] = -D(0.5*(x[i]+x[i-1]),y[j])/hx**2
                    A[i][i]   = (2.0/tau + D(0.5*(x[i]+x[i+1]),y[j])/hx**2 + D(0.5*(x[i]+x[i-1]),y[j])/hx**2)
                    A[i][i+1] =  -D(0.5*(x[i]+x[i+1]),y[j])/hx**2
                B[0] = 0.0
                B[Nx-1] = 0.0
                for i in range(1,Nx-1):
                    Ly = 1.0/hy**2*D(x[i],0.5*(y[j+1]+y[j]))*(u[i][j+1]- u[i][j]) - 1.0/hy**2*D(x[i],0.5*(y[j]+y[j-1]))*(u[i][j]- u[i][j-1])
                    B[i] = (F(0.5*(t[k]+t[k+1]),x[i],y[j],u_s[i][j]) + 2.0/tau*u[i][j] + Ly)
                u_tau_pola[:,j] = progonka(A,B)
                # error_ += np.sum(np.abs(np.matmul(A,u_tau_pola[:,j].reshape(-1,1)).flatten() - B))
            u_s = np.copy(u_tau_pola)
        
        u_tau_pola= np.copy(u_s)
        for s in range(iters_):
            u_tau = np.zeros(shape=(Nx,Ny))
            for i in range(1,Nx-1):
                Ay[0][0] = 1.0
                Ay[0][1] = 0.0
                Ay[Ny-1][Ny-1] = 1.0 
                Ay[Ny-1][Ny-2] = 0.0    
                for j in range(1,Ny-1):
                    Ay[j][j-1] = -D(x[i],0.5*(y[j]+y[j-1]))/hy**2
                    Ay[j][j]   = (2.0/tau + D(x[i],0.5*(y[j]+y[j+1]))/hy**2 + D(x[i],0.5*(y[j]+y[j-1])))
                    Ay[j][j+1] = -D(x[i],0.5*(y[j]+y[j+1]))/hy**2

                By[0] = 0.0
                By[Ny-1] = 0.0
                for j in range(1,Ny-1):
                    Lx = 1.0/hx**2*D(0.5*(x[i+1]+x[i]),y[j])*(u_tau_pola[i+1][j]- u_tau_pola[i][j]) - 1.0/hx**2*D(0.5*(x[i]+x[i-1]),y[j])*(u_tau_pola[i][j]- u_tau_pola[i-1][j])
                    By[j] = (F(0.5*(t[k]+t[k+1]),x[i],y[j],u_s[i][j])+2.0/tau*u_tau_pola[i][j] +Lx)
                u_tau[i,:] = progonka(Ay,By)
                # error_ += np.sum(np.abs(np.matmul(Ay,u_tau[i,:].reshape(-1,1)).flatten() - By))
            u_s = np.copy(u_tau)
        # errors_.append(error_)
        is_nan = np.sum(np.isnan(u_tau)) > 0
        is_inf = np.sum(np.isinf(u_tau)) > 0
        is_neg = np.sum(u_tau < 0.0) > 0

        if is_nan or is_inf or is_neg:
            end_ = True
        if end_:
            return u, k
        u = np.copy(u_s)
    # fig,ax = plt.subplots()
    # ax.plot(np.arange(Nt-1), errors_)
    # plt.show()
    return u, Nt-1


hx = 0.1
ax= 0.0
bx = np.pi
Nx=  int((bx-ax)/hx)+1
xgrid = np.linspace(ax,bx,num=Nx)

hy = 0.1
ay= 0.0
by = np.pi
Ny=  int((by-ay)/hy)+1
ygrid = np.linspace(ay,by,num=Ny)

tau = hx/10**3
# tau = 0.001
T = 0.1
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
u[0][0]= np.nan
u[0][Ny-1]= np.nan
u[Nx-1][0]= np.nan
u[Nx-1][Ny-1]= np.nan

u_analitic = np.zeros(shape=(len(xgrid),len(ygrid)))
for i in range(len(x)):
    for j in range(len(y)):
        u_analitic[i][j] = solution(t[last_t_index],x[i],y[j])



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u- u_analitic, cmap=matplotlib.cm.jet,
                    linewidth=0, antialiased=False)
fig.colorbar(surf)
ax.set_title(r'$error$')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y = np.meshgrid(x, y)
# surf = ax.plot_surface(X, Y, u, cmap=matplotlib.cm.jet,
#                     linewidth=0, antialiased=False)
# fig.colorbar(surf)
# ax.set_title(r'$u_{method}$')
# ax.set_xlabel('x')
# ax.set_ylabel('y')


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y = np.meshgrid(x, y)
# surf = ax.plot_surface(X, Y, u_analitic, cmap=matplotlib.cm.jet,
#                     linewidth=0, antialiased=False)
# fig.colorbar(surf)
# ax.set_title(r'$u_{solution}$')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()
