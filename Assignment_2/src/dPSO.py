import numpy as np
import UtilFunctions as uf
from tqdm import tqdm

class dPSO:

    def __init__(self, n_prts, n_itr, min, max, dim, objFunc, w, c1, c2, nr):

        self.n_prts = n_prts
        self.n_itr = n_itr
        self.min = min
        self.max = max
        
        self.dim = dim
        self.dimsArr = np.arange(dim)
        np.random.shuffle(self.dimsArr)

        self.objFunc = objFunc
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.nr = nr
        self.nf = int((n_itr) / (1 + ((np.log(dim)) / (np.log(nr)))))

        self.subSwarmSizes = np.array([dim])
        self.subSwarmDimsIdxs = np.array([0, dim], dtype = int)
        self.n_subSwarms = 1

        self.pos = np.random.uniform(min, max, (n_prts, dim))
        self.vel = np.zeros((n_prts, dim))
        
        self.fit = np.zeros(n_prts)
        for i, p in enumerate(self.pos):
            self.fit[i] = objFunc(p)

        self.pBestPos = self.pos.copy()
        self.pBestFit = self.fit.copy()
        
        minIdx = np.argmin(self.pBestFit)
        self.gBestPos = self.pBestPos[minIdx].copy()
        self.gBestFit = self.pBestFit[minIdx]

        self.contextVector = self.gBestPos.copy()

        self.subSwarms = np.array([subSwarm(self.pos[:, self.dimsArr], self.vel, self.pBestPos[:, self.dimsArr], self.gBestPos, self.fit, 
                                      self.pBestFit, self.gBestFit, self.w, self.c1, self.c2, self.min, self.max)])

    def run(self):

        decomp = False

        diversity = []
        diversity.append(self.subSwarmDivs())

        fitness = []
        fitness.append(self.subSwarmFit())

        for i in tqdm(range(1, self.n_itr + 1), desc="Iterations"):

            self.evalFitness(decomp)
            decomp = False
            for j, ss in enumerate(self.subSwarms):
                ss.updateVelPos()

            fitness.append(self.subSwarmFit())
            diversity.append(self.subSwarmDivs())

            if (i % self.nf == 0):
                self.decompose()
                decomp = True
        
        self.createContextVector()
        return self.contextVector, fitness, diversity


    def decompose(self):
        
        # Decompose into new subswarm indeces
        newSubSwarmSizes = np.array([], dtype = int)
        subSwarmSplitCounts = np.zeros(self.n_subSwarms, dtype = int)
        for i, s in enumerate(self.subSwarmSizes):
            if (s < self.nr):
                newSubSwarmSizes = np.append(newSubSwarmSizes, uf.random_ints_sum_to_n(s, s))
                subSwarmSplitCounts[i] = s
            else:
                newSubSwarmSizes = np.append(newSubSwarmSizes, uf.random_ints_sum_to_n(s, self.nr))
                subSwarmSplitCounts[i] = self.nr


        self.subSwarmSizes = newSubSwarmSizes.copy()
        self.n_subSwarms = len(newSubSwarmSizes)

        self.getSubSwarmDimsIdxs()

        # Create new sub swarms
        newSubSwarms = np.empty(self.n_subSwarms, dtype = object)
        idx = 0
        for i, subSwarmSplitCount in enumerate(subSwarmSplitCounts):

            subSwarmPos = self.subSwarms[i].pos
            subSwarmVel = self.subSwarms[i].vel
            subSwarmFit = self.subSwarms[i].fit
            
            subSwarm_pBestFit = self.subSwarms[i].pBestFit
            subSwarm_gBestFit = self.subSwarms[i].gBestFit
            
            subSwarm_pBestPos = self.subSwarms[i].pBestPos
            subSwarm_gBestPos = self.subSwarms[i].gBestPos

            s = 0

            for j in range(subSwarmSplitCount):

                e = s + self.subSwarmSizes[idx]

                newSubSwarms[idx] = subSwarm(subSwarmPos[:, s:e], subSwarmVel[:, s:e],
                                             subSwarm_pBestPos[:, s:e], subSwarm_gBestPos[s:e],
                                             subSwarmFit, subSwarm_pBestFit, subSwarm_gBestFit,
                                             self.w, self.c1, self.c2, self.min, self.max)
                s = e
                idx += 1

        self.subSwarms = newSubSwarms.copy()                

    def getSubSwarmDimsIdxs(self):

        newSubSwarmDimsIdxs = np.zeros(self.n_subSwarms + 1, dtype = int)
        for i, subSwarmSize in enumerate(self.subSwarmSizes):
            newSubSwarmDimsIdxs[i + 1] = subSwarmSize + newSubSwarmDimsIdxs[i]

        self.subSwarmDimsIdxs = newSubSwarmDimsIdxs.copy()

    def createContextVector(self):
        for i, ss in enumerate(self.subSwarms):
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]
            self.contextVector[subSwarmDims] = ss.gBestPos


    def evalFitness(self, merged):
        
        self.createContextVector()
        #print(self.contextVector, self.objFunc(self.contextVector), np.mean(self.subSwarmDivs()))

        # If merged evaluate merged personal and best pos
        if (merged):
            for i, ss in enumerate(self.subSwarms):
                subSwarmpBestPos = ss.pBestPos
                fitnessVals = np.zeros(self.n_prts)
                subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]

                self.contextVector[subSwarmDims] = ss.gBestPos
                ss.gBestFit = self.objFunc(self.contextVector)

                for j, p in enumerate(subSwarmpBestPos):
                    self.contextVector[subSwarmDims] = p
                    fitnessVals[j] = self.objFunc(self.contextVector)
                
                ss.pBestFit = fitnessVals.copy()
                self.contextVector[subSwarmDims] = ss.gBestPos

        for i, ss in enumerate(self.subSwarms):

            subSwarmPos = ss.pos
            fitnessVals = np.zeros(self.n_prts)
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]

            for j, p in enumerate(subSwarmPos):
                self.contextVector[subSwarmDims] = p
                fitnessVals[j] = self.objFunc(self.contextVector)

            ss.updateFit(fitnessVals)
            self.contextVector[subSwarmDims] = ss.gBestPos

    def subSwarmDivs(self):

        divs = np.zeros(self.n_subSwarms)
        for i, ss in enumerate(self.subSwarms):
            divs[i] = ss.diversity()
        return np.mean(divs)

    def subSwarmFit(self):

        avgFit = np.zeros(self.n_subSwarms)
        for i, ss in enumerate(self.subSwarms):
            avgFit[i] = ss.avgFit()
        return np.mean(avgFit)

class subSwarm:
    def __init__(self, pos, vel, pBestPos, gBestPos, fit, pBestFit, gBestFit, w, c1, c2, min, max):
        
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.pBestPos = pBestPos.copy()
        self.gBestPos = gBestPos.copy()
        self.fit = fit.copy()
        self.pBestFit = pBestFit.copy()
        self.gBestFit = gBestFit

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.min = min
        self.max = max

        self.dim = pos.shape[1]

    def updateVelPos(self):
        self.updateVel()
        self.updatePos()
    
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

    def updateFit(self, fitArr):
        self.fit = fitArr.copy()
        for i, p in enumerate(self.pos):
            if (self.fit[i] < self.pBestFit[i]):
                self.pBestFit[i] = self.fit[i]
                self.pBestPos[i] = p.copy()

        if (np.min(self.pBestFit) < self.gBestFit):
            minIdx = np.argmin(self.pBestFit)
            self.gBestFit = self.pBestFit[minIdx]
            self.gBestPos = self.pBestPos[minIdx].copy()
        
    def diversity(self):
        return (1 / self.pos.shape[0]) * np.sum(np.sqrt(np.sum(np.square(self.pos - np.mean(self.pos, axis = 0)), axis = 1)))
    
    def avgFit(self):
        return np.mean(self.fit)