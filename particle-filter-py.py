#!/usr/bin/env python3
"""Particle filter (Sequential Monte Carlo)."""
import sys,random,math

class ParticleFilter:
    def __init__(self,n_particles=500,process_noise=0.1,measure_noise=1.0,seed=42):
        self.n=n_particles;self.pn=process_noise;self.mn=measure_noise
        self.rng=random.Random(seed)
        self.particles=[self.rng.gauss(0,1) for _ in range(self.n)]
        self.weights=[1.0/self.n]*self.n
    def predict(self):
        self.particles=[p+self.rng.gauss(0,self.pn) for p in self.particles]
    def update(self,z):
        for i in range(self.n):
            d=z-self.particles[i]
            self.weights[i]=math.exp(-0.5*(d/self.mn)**2)
        total=sum(self.weights)+1e-300
        self.weights=[w/total for w in self.weights]
    def resample(self):
        cumsum=[];s=0
        for w in self.weights:s+=w;cumsum.append(s)
        new=[]
        for _ in range(self.n):
            r=self.rng.random()
            for i,c in enumerate(cumsum):
                if r<=c:new.append(self.particles[i]);break
        self.particles=new;self.weights=[1.0/self.n]*self.n
    def estimate(self):return sum(p*w for p,w in zip(self.particles,self.weights))
    def step(self,z):self.predict();self.update(z);est=self.estimate();self.resample();return est

def main():
    if len(sys.argv)>1 and sys.argv[1]=="--test":
        pf=ParticleFilter(1000,0.01,0.5,seed=42)
        true_val=5.0;rng=random.Random(42)
        for _ in range(50):
            z=true_val+rng.gauss(0,0.5);pf.step(z)
        est=pf.estimate()
        assert abs(est-true_val)<1.0,f"Bad estimate: {est}"
        print("All tests passed!")
    else:
        pf=ParticleFilter();rng=random.Random(0)
        for i in range(20):
            z=10+rng.gauss(0,1);est=pf.step(z)
            print(f"  z={z:.2f} est={est:.2f}")
if __name__=="__main__":main()
