#!/usr/bin/env python
# coding: utf-8

# y = Theta
# 
# x = time
# 
# z = omega(angular velocity)
# 
# a = acceleration
# 
# g(x,y) is the derivative dy/dx
# 
# h is the stepsize
# 
# f_i is the value of y(x_i), so
# 
# f_ipo is the value of y(x_ipo).

# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# # Define our coupled derivatives to integrate

# In[93]:


def dydx(x, y): #A, B, C, D):
    
    # Set the derivatives
    
    # Our equation is d^2y/dx^2 = -A*sin(y)-B*z+C*sin(D*x)
    
    # So we can write
    #z = dydx
    #dzdx = -A*sin(y)-B*z+C*sin(D*x) = a
    
    # We will set a = y[0]
    # we will set z = y[1]
    
    # Declare an array
    y_derivs = np.zeros(2)
    
    # Set dydx = z
    y_derivs[0] = y[1]
    
    # Set dy^2dx^2 = a
    y_derivs[1] = -1*np.sin(y[0]) -0*y[1] + 2/3*np.sin(0*x)
    
    # Here we have to return an array
    return y_derivs


# x is an array, how do we calculate y_derivs[1] for each value of x? do we have to set a loop? wouldn't that make y_derivs an array of len > 2?

# ### Define the 4th order RK method

# In[94]:


def rk4_mv_core(dydx, xi, yi, nv, h):
    
    # Declare k? arrays
    k1 = np.zeros(nv)
    k2 = np.zeros(nv)
    k3 = np.zeros(nv)
    k4 = np.zeros(nv)
    
    # Define x at 1/2 step
    x_ipoh = xi +0.5*h
    
    # Define x at 1 step
    x_ipo = xi + h
    
    # Declare a temporary arrayy 
    y_temp = np.zeros (nv)
    
    # Get k1 values
    y_derivs = dydx(xi, yi)
    k1[:] = h*y_derivs[:]
    
    # Get k2 values
    y_temp[:] = yi[:] + 0.5*k1[:]
    y_derivs = dydx(x_ipoh, y_temp)
    k2[:] = h*y_derivs[:]
    
    # Get k3 values
    y_temp[:] = yi[:] + 0.5*k2[:]
    y_derivs = dydx(x_ipoh, y_temp)
    k3[:] = h*y_derivs[:]
    
    # Get k4 values
    y_temp[:] = yi[:] + k3[:]
    y_derivs = dydx(x_ipo, y_temp) #Why ipo and not ipoh?
    k4[:] = h*y_derivs[:]
    
    # Advance y by a step h
    yipo = yi + (k1 +2*k2 + 2*k3 + k4)/6.
    
    return yipo


# This script will be able to create its own pairs to compare their values, and to keep updating the size of its steps, to get a more accurate approximation
# 
# For a 1d array, : means "all the elements", so we don't ned a for loop to go through each f them.

# ### Define an adaptive step size driver for RK4

# In[95]:


def rk4_mv_ad(dydx, x_i, y_i, nv, h, tol):

    #define safetyscale
    SAFETY = .9
    H_NEW_FAC = 2.
    
    #Set a maximum number of iterations
    imax = 10000
    
    #Set an iteration varale
    i = 0
    
    #Create an error
    Delta = np.full(nv, 2*tol)
    
    #Remember the step
    h_step = h
    
    #Adjust the step
    while (Delta.max()/tol > 1.0):
        # Estimate our error by taking one step of size  h
        #vs. two steps of the size h/2
        y_2 = rk4_mv_core(dydx, x_i, y_i, nv, h_step)
        y_1 = rk4_mv_core(dydx, x_i, y_i, nv, 0.5*h_step)
        y_11 = rk4_mv_core(dydx, x_i+0.5*h_step, y_1, nv, 0.5*h_step)
        
        
        # Always try to make every step bigger than the other,
        #but they are always correlated, not just an array of
        #numbers together
        
        #Compute an error
        Delta = np.fabs(y_2 - y_11)
        
        # If the error is too large, take a smaller step
        if (Delta.max()/tol > 1.0):
            #our error is too large, degrease the step
            h_step *= SAFETY * (Delta.max()/tol)**(-0.25)
         
        #Check iteration
        if(i >= imax):
            print ('Too may iteratctions in rk4_mv_ad()')
            raise StopIteration ("Ending after i =", i)
            
        #iterate
        i += 1
        
    #next time, try to take a bigger step
    h_new = np.fmin(h_step * (Delta.max()/tol)**(-0.9), h_step*H_NEW_FAC)
    
    #Return the answer, a new stap, and the step we actually took
    return y_2, h_new, h_step


# Slide 16: instead of dfdx, is dydx
# 
# h_new can never be too big, the while loop keeps taking the minimum, or a apropriate value
# 
# h serves for increasing integration, or decreasing. if a>b, it regresses, giving negative values, So h changes appropriately to avoid oberflowig?

# ### Define a wrapper for RK4

# rk4_mv returns two arrays: x and y, with the respective values obtained from the "integration"

# In[96]:


def rk4_mv(dydx, a, b, y_a, tol):
    
    #dydx is the derivative wrt x 
    #a is the lower bound
    #b is the upper bound
    #y_a are the boundary conditions
    #tol is the tolerance for integrating y
    
    #define our starting step
    xi = a
    yi = y_a.copy()
    
    #an initial step size == make very small
    h = 1.0e-4 * (b-a)
    
    #set max number of iterations
    imax = 10000
    
    #Set an iteration variable
    i = 0
    
    #Set the number of coupled odes to the
    #size of y_a
    nv = len(y_a)
    
    #Set the initial conditions
    x = np.full(1,a)
    y = np.full((1, nv), y_a)
    
    #Set a flag
    flag = 1
    
    #Loop until we reach the right side
    while(flag):
        #caluclate y_i + 1
        yi_new, h_new, h_step = rk4_mv_ad(dydx, xi, yi, nv, h, tol)
        
        #Update the step
        h = h_new
        
        #Prevent an overshoot
        if(xi+h_step > b):
            
            #take a smaller step
            h = b-xi
            
            #recalculate y_i+1
            yi_new, h_new, h_step = rk4_mv_ad(dydx, xi, yi, nv, h, tol)
            
            #break
            flag = 0
            
        #Update values
        xi += h_step
        yi[:] = yi_new[:]
        
        #add the step to the arrays
        x = np. append(x, xi)
        y_new = np.zeros((len(x), nv))
        y_new[0:len(x) - 1, :] = y
        y_new[-1, :] = yi[:]
        del y
        y = y_new
        
        #Prevent too many interactions
        if(i >= imax):
            print("MAximum iterations reachd.")
            raise StopIteration("Iteration number = ", i)
            
        #iterate
        i += 1
        
        #output some information
        s = "i = %3d\tx = %9.8f\th = %9.8f\tb=%9.8f" % (i, xi, h_step, b)
        print(s)
        
        #Break if new xi is == b
        if (xi == b):
            flag = 0
            
    #return the answer
    return x, y


# ### Perform the integration

# In[97]:


a = 0.0
b = 100.

#A
#B
#C
#D


y_0 = np.zeros(2)
y_0[0] = 0.0
y_0[1] = 1.0
nv = 2

tolerance = 1.0e-6

#Perform the integration
x, y = rk4_mv(dydx, a, b, y_0, tolerance)


# ### Plot the results

# In[102]:


f, axarr = plt.subplots(3,1, figsize=(10,12))

axarr[0].plot(x, y[:,0], '-', label='y(x)')

axarr[0].set_xlabel('x')
axarr[0].set_ylabel('y, dy/dx')
axarr[0].set_title('angle v. time', color='C0')
axarr[0].axis()
#axarr[0].set_aspect(100)

axarr[1].plot(x, y[:,1], '-', color='orange', label='dydx(x)')
axarr[1].set_xlabel('x')
axarr[1].set_ylabel('dy^2/dx^2')
axarr[1].set_title('speed v. time', color='C1')

axarr[2].plot(y[:,1], y[:,0], '-', color='green', label='omega(theta)')

axarr[2].set_xlabel('theta')
axarr[2].set_ylabel('omega, dy/dx')
axarr[2].set_title('speed v. angle', color='C2')

f.subplots_adjust(wspace = 1.0)
#fig = plt.figure(figsize = (6,6))

'''Aesthetic edits to the plots'''

#change plots border width
plt.rcParams['axes.linewidth'] = 1 #set the value globally
#change distance between plots
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)


# ### Plot the error

# Notice that the errors will actually exceed our "tolerance".

# In[99]:


sine = np.sin(x)
cosine = np.cos(x)

y_error = (y[:,0] - sine)
dydx_error = (y[:,1] - cosine)

plt.plot(x, y_error, label='y(x) Error')
plt.plot(x, dydx_error, label="dydx(x) Error")
plt.legend(frameon=False)


# Below is the result that we are looking for, except we need it with our data, in a 2x2 panel of plots, and an illustration of the actual pendulum with time in the 4th panel.

# Not sure how to get it to play in the jupyter instead of forcing to save and play on the desktop as html

# In[120]:


"""
A simple example of an animated plot

Found at:
https://matplotlib.org/examples/animation/simple_anim.html
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)
ani.save('simple_example.html', fps=15)
plt.show()


# Found this Double Pendulum example as well. This is the 4th plot that we are looking for

# In[ ]:


"""
===========================
The double pendulum problem

found at:
https://matplotlib.org/examples/animation/double_pendulum_animated.html
===========================

This animation illustrates the double pendulum problem.
"""

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()

