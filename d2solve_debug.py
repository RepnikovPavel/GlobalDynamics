from funcs import *
from tqdm import tqdm
import matplotlib

# @jit(nopython=True)
# def F(u):
#     # return u**2.0
#     # return 10.0
#     # return u**2
#     # return np.random.rand()*10
#     return  u**2

# @jit(nopython=True)
# def leftGU(x,y):
#     return 0.0

# @jit(nopython=True)
# def rightGU(x,y):
#     return 0.0

# NU_x= Distrib(make_func("triangle_m_f", [-0.5,0.0,0.5]), supp_of_func=(-0.5,0.0,0.5))
# # NU_y= Distrib(make_func("triangle_m_f", [-1.0,-0.5,0.0]), supp_of_func=(-1.0,-0.5,0.0))

# def NU(x,y,xgrid,ygrid):
#     # return NU_x(x)*NU_y(y)  
#     return NU_x(x)*NU_x(y)  
#     # return 0.0
#     # mu = xgrid[int(len(xgrid)/2)]
#     # sigma = 0.05
#     # return np.exp(-0.5*np.square((x-mu)/sigma))*10.0
#     # print(int(len(xgrid)/2))
#     # print(x)
#     # if x == xgrid[int(len(xgrid)/2)]:
#     #     return 10.0
#     # else:
#         # return 0.0

# @jit(nopython =True)
# def D(x,y,u):
#     mu1 = 0.25
#     sigma = 0.05
#     mu2 = -0.25
#     # return u**2.0*(1.0 - np.exp(-0.5*np.square((x-mu1)/sigma)))*(1.0 - np.exp(-0.5*np.square((x-mu2)/sigma)))
#     # return u**2.0
#     # return u**2
#     return 1.0

# @jit(nopython=True)
# def F(t,x,y,u):
#     return np.sin(x)*np.sin(y)*np.exp(-t)

# @jit(nopython=True)
# def solution(t,x,y):
#     return np.sin(x)*np.sin(y)*(np.exp(t)-1.0)*np.exp(-2.0*t)

@jit(nopython=True)
def F(t,x,y,u):
    return u**2


NU_x= Distrib(make_func("triangle_m_f", [-0.5,0.0,0.5]), supp_of_func=(-0.5,0.0,0.5))
# NU_y= Distrib(make_func("triangle_m_f", [-1.0,-0.5,0.0]), supp_of_func=(-1.0,-0.5,0.0))

def NU(x,y,xgrid,ygrid):
    # return NU_x(x)*NU_y(y)  
    # return NU_x(x)*NU_x(y)  
    return  0.0

@jit(nopython =True)
def D(x,y,u):
    return u**2



@jit(nopython=True)
def solve(Nx,Ny,Nt,hx,hy,tau,t,x,y,u_):
    u = np.copy(u_)    
    B = np.zeros(shape=(Nx,))
    A = np.zeros(shape=(Nx, Nx))

    By = np.zeros(shape=(Ny,))
    Ay = np.zeros(shape=(Ny, Ny))
    end_ = False
    for k in range(Nt-1):
        u_s = np.copy(u)        
        for j in range(1,Ny-1):
            # errors_ = []
            for s in range(20):
                A[0][0] = -1.0/hx
                A[0][1] = 1.0/hx
                A[Nx-1][Nx-1] = 1.0/hx 
                A[Nx-1][Nx-2] = -1.0/hx
                # A[0][0] = 1.0
                # A[0][1] = 0.0
                # A[Nx-1][Nx-1] = 1.0 
                # A[Nx-1][Nx-2] = 0.0
                for i in range(1,Nx-1):
                    A[i][i-1] = 1.0/4.0/hx**2*(D(x[i+1],y[j],u_s[i+1][j])-D(x[i-1],y[j],u_s[i-1][j])) - 1.0/hx**2*D(x[i],y[j],u_s[i][j])
                    A[i][i]   = 2.0/tau+2.0/hx**2*D(x[i],y[j],u_s[i][j])
                    A[i][i+1] = -1.0/4.0/hx**2*(D(x[i+1],y[j],u_s[i+1][j]) - D(x[i-1],y[j],u_s[i-1][j])) - 1.0/hx**2*D(x[i],y[j],u_s[i][j])
                B[0] = 0.0
                B[Nx-1] = 0.0
                for i in range(1,Nx-1):
                    B[i] = F((t[k]+t[k+1])*0.5,x[i],y[j],u_s[i][j])+2.0/tau*u[i][j] + \
                        1.0/(4.0*hy**2)*(D(x[i],y[j+1],u[i][j+1])-D(x[i],y[j-1],u[i][j-1]))*(u[i][j+1]-u[i][j-1])+\
                        1.0/(hy**2)*D(x[i],y[j],u[i][j])*(u[i][j+1]-2.0*u[i][j]+u[i][j-1])
                u_s[:,j] = progonka(A,B)
            #     u_s[:,j] = np.matmul(np.linalg.inv(A),B)
            #     error_ = np.max(np.abs(np.matmul(A,u_s[:,j].reshape(-1,1))-B))
            #     errors_.append(error_)

            # if np.max(errors_) > 0.0:
            #     fig,ax = plt.subplots()
            #     ax.plot(np.arange(20),errors_)
            #     plt.show()

        u_tau_pola = np.copy(u_s)
        for i in range(1,Nx-1):
            for s in range(20):
                Ay[0][0] = -1.0/hy
                Ay[0][1] = 1.0/hy
                Ay[Ny-1][Ny-1] = 1.0/hy 
                Ay[Ny-1][Ny-2] = -1.0/hy

                # Ay[0][0] = 1.0
                # Ay[0][1] = 0.0
                # Ay[Ny-1][Ny-1] = 1.0 
                # Ay[Ny-1][Ny-2] = 0.0

                for j in range(1,Ny-1):
                    Ay[j][j-1] = 1.0/(4.0*hy**2)*D(x[i],y[j+1],u_s[i][j+1]) - 1.0/(4.0*hy**2)*D(x[i],y[j-1],u_s[i][j-1]) - 1.0/hy**2*D(x[i],y[j],u_s[i][j])
                    Ay[j][j]   = 2.0/tau+2.0/(hy**2)*D(x[i],y[j],u_s[i][j])
                    Ay[j][j+1] = -1.0/(4.0*hy**2)*D(x[i],y[j+1],u_s[i][j+1]) + 1.0/(4.0*hy**2)*D(x[i],y[j-1],u_s[i][j-1]) - 1.0/hy**2*D(x[i],y[j],u_s[i][j])

                By[0] = 0.0
                By[Ny-1] = 0.0
                for j in range(1,Ny-1):
                    By[j] = F(t[k+1],x[i],y[j],u_s[i][j])+2.0/tau*u_tau_pola[i][j] + \
                        1.0/(4.0*hx**2)*(D(x[i+1],y[j],u_tau_pola[i+1][j])-D(x[i-1],y[j],u_tau_pola[i-1][j]))*(u_tau_pola[i+1][j]-u_tau_pola[i-1][j])+\
                        1.0/(hx**2)*D(x[i],y[j],u_tau_pola[i][j])*(u_tau_pola[i+1][j]-2.0*u_tau_pola[i][j]+u_tau_pola[i-1][j])
                u_s[i,:] = progonka(Ay,By)
                # u_s[i,:]= np.matmul(np.linalg.inv(Ay),By)
                # error_ = np.max(np.abs(np.matmul(Ay,u_s[i,:].reshape(-1,1))-By))
                # if np.max(error_) > 0.0:
                #     print(1)
                # u_s[i,:] = np.matmul(np.linalg.inv(Ay),By)

        if np.sum(np.isnan(u_s)) > 0 or np.sum(np.isinf(u_s)) > 0 or np.sum(u_s < 0.0) > 0:
            end_ = True
        if end_:
            return u, k
        u = np.copy(u_s)
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

# u_analitic = np.zeros(shape=(len(xgrid),len(ygrid)))
# for i in range(len(x)):
#     for j in range(len(y)):
#         u_analitic[i][j] = solution(t[last_t_index],x[i],y[j])



# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y = np.meshgrid(x, y)
# surf = ax.plot_surface(X, Y, u- u_analitic, cmap=matplotlib.cm.jet,
#                     linewidth=0, antialiased=False)
# fig.colorbar(surf)
# ax.set_title(r'$error$')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, cmap=matplotlib.cm.jet,
                    linewidth=0, antialiased=False)
fig.colorbar(surf)
ax.set_title(r'$u_{method}$')
ax.set_xlabel('x')
ax.set_ylabel('y')



# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y = np.meshgrid(x, y)
# surf = ax.plot_surface(X, Y, u_analitic, cmap=matplotlib.cm.jet,
#                     linewidth=0, antialiased=False)
# fig.colorbar(surf)
# ax.set_title(r'$u_{solution}$')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()

plt.show()