import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(xvo, ti, p):
	x, v = xvo
	m, k, xeq = p

	return [v, a.subs({M:m, K:k, XEQ:xeq, X:x})]


M, K, XEQ, t = sp.symbols('M K XEQ t')
X = dynamicsymbols('X')

T = sp.Rational(1, 2) * M * X.diff(t, 1)**2
V = sp.Rational(1, 2) * K * (X - XEQ)**2

L = T - V

dLdx = L.diff(X, 1)
dLdxdot = L.diff(X.diff(t, 1), 1)
ddtdLdxdot = dLdxdot.diff(t, 1)

dL = ddtdLdxdot - dLdx

aa = sp.solve(dL,X.diff(t, 2))

a = sp.simplify(aa[0])

#--------------------

m = 1
k = 1
xo = 2
vo = 0
xeq = 1

p = m, k, xeq
xv_o = xo, vo

tf = 60 
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

xv = odeint(integrate, xv_o, ta, args=(p,))


x = xv[:,0]
v = xv[:,1]

ke = np.asarray([T.subs({M:m, X.diff(t,1):i}) for i in v])
pe = np.asarray([V.subs({K:k, XEQ:xeq, X:i}) for i in x])
E = ke + pe

fig, a=plt.subplots()

rad = 0.25
yline = 0
xmax = max(x)+0.5
xmin = min(x)-rad-0.5
ymax = yline + 2*rad
ymin = yline - 2*rad
nl = int(np.ceil((xo+rad)/(2*rad)))
xl = np.zeros((nl,nframes))
yl = np.zeros((nl,nframes))
for i in range(nframes):
	l = (x[i]/nl)
	xl[0][i] = x[i] - rad - 0.5*l
	for j in range(1,nl):
		xl[j][i] = xl[j-1][i] - l
	for j in range(nl):
		yl[j][i] = yline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l)**2))

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x[frame],yline),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([x[frame]-rad,xl[0][frame]],[yline,yl[0][frame]],'xkcd:cerulean')
	plt.plot([xl[nl-1][frame],-rad],[yl[nl-1][frame],yline],'xkcd:cerulean')
	for i in range(nl-1):
		plt.plot([xl[i][frame],xl[i+1][frame]],[yl[i][frame],yl[i+1][frame]],'xkcd:cerulean')
	plt.title("A Simple Horizontal Spring")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('spring.mp4', writer=writervideo)
plt.show()

