import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile
SR=44100; D=10.0; N=int(SR*D); t=np.arange(N)/SR
rng=np.random.default_rng(7)
L=np.zeros(N); R=np.zeros(N)
def hz(n): return 440*2**((n-69)/12)
def env(a,b,att,rel):
    e=np.zeros(N); i0,i1=int(a*SR),int(min(b,D)*SR); n=i1-i0
    if n<=0: return e
    x=np.arange(n)/SR; e[i0:i1]=np.minimum(1,x/att)*np.minimum(1,(n/SR-x)/rel); return e
def lp(x,fc): return sosfilt(butter(2,fc/(SR/2),output='sos'),x)
def saw(f,ph=0): return 2*((t*f+ph)%1)-1
def strings(notes,a,b,vol,fc=1800,att=1.2,rel=1.0,pan=0):
    global L,R
    e=env(a,b,att,rel); s=np.zeros(N)
    for n in notes:
        for d in (-7,0,6):
            s+=saw(hz(n)*2**(d/1200),rng.random())
    s=lp(s,fc)*e*vol*(1+0.04*np.sin(2*np.pi*5.2*t))
    L+=s*(1-pan)*.9; R+=s*(1+pan)*.9
def piano(n,at,vol=.25,pan=0):
    global L,R
    i0=int(at*SR); n_=N-i0; x=np.arange(n_)/SR; f=hz(n)
    s=sum(np.sin(2*np.pi*f*k*x)*(.6**k) for k in range(1,6))*np.exp(-x*2.2)*np.minimum(1,x/.004)
    L[i0:]+=s*vol*(1-pan); R[i0:]+=s*vol*(1+pan)
def boom(at,vol=.9,f0=70):
    global L,R
    i0=int(at*SR); n_=min(N-i0,int(2.5*SR)); x=np.arange(n_)/SR
    f=f0*(1+1.2*np.exp(-x*18)); ph=2*np.pi*np.cumsum(f)/SR
    s=np.sin(ph)*np.exp(-x*2.8)+lp(rng.standard_normal(n_),900)*np.exp(-x*25)*.6
    L[i0:i0+n_]+=s*vol; R[i0:i0+n_]+=s*vol
def swell(a,b,vol):
    global L,R
    e=np.zeros(N); i0,i1=int(a*SR),int(b*SR); e[i0:i1]=np.linspace(0,1,i1-i0)**2
    s=lp(rng.standard_normal(N),2500)*e*vol; L+=s; R+=s
# Act 1 (0-4s) night: A minor, low drone, piano arpeggio
strings([33,45],0,4.3,.05,fc=500,att=1.5,rel=.8)
strings([57,60,64],0.3,4.3,.035,fc=1400,att=1.8,rel=.9,pan=-.2)
for i,(n,at) in enumerate([(69,.4),(72,.9),(76,1.4),(81,1.9),(76,2.4),(72,2.9),(69,3.3),(64,3.7)]): piano(n,at,.16,pan=(-.3 if i%2 else .3))
boom(0.05,.6,55)
# Act 2 (4-7.2s) sunrise: F -> C, swelling
boom(4.0,.55,58)
strings([29,41],4,5.7,.06,fc=600,att=.4,rel=.4)
strings([53,57,60,65],4,5.8,.04,fc=2000,att=.6,rel=.5,pan=.15)
strings([36,48],5.6,7.4,.06,fc=650,att=.4,rel=.3)
strings([55,60,64,67],5.6,7.4,.045,fc=2600,att=.5,rel=.3,pan=-.15)
for i,(n,at) in enumerate([(77,4.1),(81,4.5),(84,4.9),(81,5.3),(79,5.65),(84,6.0),(88,6.4)]): piano(n,at,.14,pan=(.25 if i%2 else -.25))
swell(5.9,7.2,.12)
# Act 3 (7.2-10s) fireworks finale: G -> C major, big hits
strings([31,43],7.2,8.6,.08,fc=900,att=.05,rel=.3)
strings([55,59,62,67,71],7.2,8.6,.05,fc=3800,att=.08,rel=.3)
strings([36,48],8.5,10,.09,fc=900,att=.15,rel=1.2)
strings([60,64,67,72,76],8.5,10,.055,fc=4200,att=.2,rel=1.2)
for at,f in [(7.2,52),(7.3,62),(7.7,58),(8.1,64),(8.5,48),(8.6,60),(8.9,62)]: boom(at,.75 if at in (7.2,8.5) else .45,f)
for n,at in [(84,8.55),(88,8.65),(91,8.75),(96,8.85)]: piano(n,at,.12)
# reverb
ir_n=int(2.0*SR); x=np.arange(ir_n)/SR
irL=rng.standard_normal(ir_n)*np.exp(-x*3.2); irR=rng.standard_normal(ir_n)*np.exp(-x*3.2)
irL=lp(irL,5000); irR=lp(irR,5000); irL/=np.sqrt((irL**2).sum()); irR/=np.sqrt((irR**2).sum())
wL=L+fftconvolve(L,irL)[:N]*.35; wR=R+fftconvolve(R,irR)[:N]*.35
fade=np.minimum(1,np.minimum(t/.15,(D-t)/.8)); wL*=fade; wR*=fade
m=max(np.abs(wL).max(),np.abs(wR).max()); out=np.stack([wL,wR],1)/m*.89
wavfile.write('/tmp/mus/score.wav',SR,(out*32767).astype(np.int16)); print('ok')
