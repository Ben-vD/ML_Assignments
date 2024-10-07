import numpy as np
import ObjectiveFunctions as of
from tqdm import tqdm
import matplotlib.pyplot as plt

class PSO:

    def __init__(self, n_prts, n_itr, min, max, dim, objFunc, w, c1, c2):
        
        self.dim = dim

        self.objFunc = objFunc
        self.n_itr = n_itr 
        self.n_prts = n_prts

        self.min = min
        self.max = max

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.pos = np.random.uniform(min, max, (n_prts, dim))
        self.vel = np.zeros((n_prts, dim))

        self.fit = np.zeros(n_prts)
        for i, p in enumerate(self.pos):
            self.fit[i] = self.objFunc(p)

        self.pBestFit = self.fit.copy()
        self.pBestPos = self.pos.copy()

        minIdx = np.argmin(self.pBestFit)
        self.gBestFit = self.pBestFit[minIdx]
        self.gBestPos = self.pBestPos[minIdx].copy()

    def run(self):

        diversity = []
        diversity.append(self.diversity())

        fitness = []
        fitness.append(self.avgFit())

        for i in tqdm(range(1, self.n_itr + 1), desc="Iterations"):
            
            self.evalFitness()
            self.updateVel()
            self.updatePos()

            fitness.append(self.avgFit())
            diversity.append(self.diversity())

        return self.gBestPos, fitness, diversity

    def updatePos(self):
        self.pos = self.pos + self.vel

        mask = (self.pos < self.min) | (self.pos > self.max)
        random_values = np.random.uniform(self.min, self.max, size=np.sum(mask))
        self.pos[mask] = random_values

        self.vel[mask] = 0

    def updateVel(self):
        
        r1 = np.random.uniform(0, 1, self.vel.shape)
        r2 = np.random.uniform(0, 1, self.vel.shape)

        term1 = self.w * self.vel
        term2 = self.c1 * r1 * (self.pBestPos - self.pos)
        term3 = self.c2 * r2 * (self.gBestPos - self.pos)

        self.vel = term1 + term2 + term3

    def evalFitness(self):
        
        for i, p in enumerate(self.pos):
            self.fit[i] = self.objFunc(p)

            if (self.fit[i] < self.pBestFit[i]):
                self.pBestFit[i] = self.fit[i]
                self.pBestPos[i] = p.copy()

        if (np.min(self.pBestFit) < self.gBestFit):
            minIdx = np.argmin(self.pBestFit)
            self.gBestFit = self.pBestFit[minIdx]
            self.gBestPos = self.pBestPos[minIdx].copy()


    def diversity(self):
        return (1 / self.n_prts) * np.sum(np.sqrt(np.sum(np.square(self.pos - np.mean(self.pos, axis = 0)), axis = 1)))
    
    def avgFit(self):
        return np.mean(self.fit)
        

    def plot(self):
        x = np.linspace(self.min, self.max, 100)
        y = np.linspace(self.min, self.max, 100)

        X, Y = np.meshgrid(x, y)
        Z = np.apply_along_axis(self.objFunc, axis = 0, arr = [X, Y])

        # Creating a filled contour plot
        plt.contourf(X, Y, Z)
        plt.scatter(self.pos[:, 0], self.pos[:, 1])
        
        plt.xlim(self.min, self.max)
        plt.ylim(self.min, self.max)

        plt.draw()  # Update the plot without blocking
        plt.pause(0.001)  # Pause briefly to allow plot update
        plt.clf()  # Clear the figure for the next plot
