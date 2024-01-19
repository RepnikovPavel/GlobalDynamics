# Previous formulation  

**total population dynamics**  

$$
\frac{d}{dt}N  = \frac{1}{C_0}N^2
$$  


**the results of previous works by other authors**

$$
u_t = \alpha u^2 + (D_x u_x)_x + (D_y u_y)_y
$$

**my suggestion is to model global dynamics taking into account spatial distribution, taking into account the hyperbolic law for the total number of people on the globe**  


$$  
u_t = (\phi(t,x,y) u_x)_x + (\psi(t,x,y) u_y)_y + \frac{1}{C_0}u(t,x,y)\int_{Globe}{u(t,x,y)dS}  
$$  


$$
u|_{water} = 0 
$$  

$$
\frac{\partial}{\partial \vec{n}}u|_{coastline} = 0
$$  

$$
\frac{\partial}{\partial \vec{n}}u|_{borders \ between \ countries} = what \ you \ want
$$  


# The results  

<!-- https://drive.google.com/file/d/1webG3ZLsH5JWXA3gQHQQH8GpWVwWU6cd/view?usp=sharing -->

<!-- ![color picker](https://bobbyhadz.com/images/blog/change-vscode-integrated-terminal-colors/hover-over-color.gif) -->

<!-- ![Alt text](https://drive.google.com/uc?id=1XJO49BaIYPR2dbFGyG2k9ABKFbq_Ec3Z)   -->

<a href="https://drive.google.com/uc?export=view&id=1webG3ZLsH5JWXA3gQHQQH8GpWVwWU6cd"><img src="https://drive.google.com/uc?export=view&id=1webG3ZLsH5JWXA3gQHQQH8GpWVwWU6cd" style="width: 900px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


<!-- https://drive.google.com/drive/u/1/folders/1XJO49BaIYPR2dbFGyG2k9ABKFbq_Ec3Z -->

# Options for future research  

Simulation of transport through air or by water by modeling additional source functions



