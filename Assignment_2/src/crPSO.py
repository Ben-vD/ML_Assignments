import numpy as np
import UtilFunctions as uf
from tqdm import tqdm

class crPSO:

    def __init__(self, n_prts, n_itr, min, max, dim, objFunc, w, c1, c2):
        
        self.n_prts = n_prts
        self.n_itr = n_itr
        self.min = min
        self.max = max
        self.dim = dim
        self.objFunc = objFunc
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.contextVector = np.zeros(dim)

        self.initSubswarms()
        self.initFit()

    def initSubswarms(self):

        self.n_subSwarms = np.random.randint(1, self.dim + 1)
        self.subSwarmSizes = uf.random_ints_sum_to_n(self.dim, self.n_subSwarms)

        self.dimsArr = np.arange(self.dim)
        np.random.shuffle(self.dimsArr)

        self.subSwarmDimsIdxs = np.zeros(self.n_subSwarms + 1, dtype = int)
        for i in range(1, self.n_subSwarms + 1):
            self.subSwarmDimsIdxs[i] = self.subSwarmSizes[i - 1] + self.subSwarmDimsIdxs[i - 1]

        self.subSwarms = np.empty(self.n_subSwarms, dtype = object)
        for s in range(self.n_subSwarms):
            self.subSwarms[s] = subSwarm(self.n_prts, self.min, self.max, self.subSwarmSizes[s] , self.w, self.c1, self.c2)

    def initFit(self):
        part_idxs = np.zeros(self.n_subSwarms, dtype = int)
        
        # Get random particles from sub swarm to set first context vector
        for i, ss in enumerate(self.subSwarms):
            part_idx, p = ss.getRandomPart()
            part_idxs[i] = part_idx
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]
            self.contextVector[subSwarmDims] = p

        # Set sub swarm initial position paramenters
        fitness = self.objFunc(self.contextVector)
        for i, ss in enumerate(self.subSwarms):
            ss.fit[part_idxs[i]] = fitness
            
            ss.pBestFit[part_idxs[i]] = fitness
            ss.pBestPos[part_idxs[i]] = ss.pos[part_idxs[i]].copy()

            ss.gBestFit = fitness
            ss.gBestPos = ss.pos[part_idxs[i]].copy()

        # Evaluate all particles initial values
        for i, ss in enumerate(self.subSwarms):
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]

            for j, p in enumerate(ss.pos):

                self.contextVector[subSwarmDims] = p
                fitnessVal = self.objFunc(self.contextVector)
                ss.fit[j] = fitnessVal
                ss.pBestFit[j] = fitnessVal

                if (fitnessVal < ss.gBestFit):
                    ss.gBestFit = fitnessVal
                    ss.gBestPos = p.copy()

            self.contextVector[subSwarmDims] = ss.gBestPos

    def createContextVector(self):
        for i, ss in enumerate(self.subSwarms):
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]
            self.contextVector[subSwarmDims] = ss.gBestPos

    def evalFitness(self):
        
        self.createContextVector()
        #print(self.contextVector, self.objFunc(self.contextVector), np.mean(self.subSwarmDivs()))

        for i, ss in enumerate(self.subSwarms):

            subSwarmPos = ss.pos
            fitnessVals = np.zeros(self.n_prts)
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]

            self.contextVector[subSwarmDims] = ss.gBestPos
            ss.gBestFit = self.objFunc(self.contextVector)

            for j, p in enumerate(subSwarmPos):

                self.contextVector[subSwarmDims] = ss.pBestPos[j]
                ss.pBestFit[j] = self.objFunc(self.contextVector)

                self.contextVector[subSwarmDims] = p
                fitnessVals[j] = self.objFunc(self.contextVector)

            ss.updateFit(fitnessVals)
            self.contextVector[subSwarmDims] = ss.gBestPos

    def regroup(self):
        

        pos = np.zeros((self.n_prts, self.dim))
        vel = np.zeros((self.n_prts, self.dim))

        pBestPos = np.zeros((self.n_prts, self.dim))
        gBestPof = np.zeros(self.dim)

        for i, ss in enumerate(self.subSwarms):
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]
            pos[:, subSwarmDims] = ss.pos
            vel[:, subSwarmDims] = ss.vel
            pBestPos[:, subSwarmDims] = ss.pBestPos
            gBestPof[subSwarmDims] = ss.gBestPos

        self.subSwarmSizes = uf.random_ints_sum_to_n(self.dim, self.n_subSwarms)
        np.random.shuffle(self.dimsArr)

        self.subSwarmDimsIdxs = np.zeros(self.n_subSwarms + 1, dtype = int)
        for i in range(1, self.n_subSwarms + 1):
            self.subSwarmDimsIdxs[i] = self.subSwarmSizes[i - 1] + self.subSwarmDimsIdxs[i - 1]

        self.subSwarms = np.empty(self.n_subSwarms, dtype = object)
        for s in range(self.n_subSwarms):
            self.subSwarms[s] = subSwarm(self.n_prts, self.min, self.max, self.subSwarmSizes[s] , self.w, self.c1, self.c2)

        for i, ss in enumerate(self.subSwarms):
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]
            ss.pos = pos[:, subSwarmDims]
            ss.vel = vel[:, subSwarmDims]
            ss.pBestPos = pBestPos[:, subSwarmDims]
            ss.gBestPos = gBestPof[subSwarmDims]

    def run(self):

        diversity = []
        diversity.append(self.subSwarmDivs())

        fitness = []
        fitness.append(self.subSwarmFit())

        for i in tqdm(range(1, self.n_itr + 1), desc="Iterations"):
            self.evalFitness()

            for j, ss in enumerate(self.subSwarms):
                ss.updateVelPos()

            fitness.append(self.subSwarmFit())
            diversity.append(self.subSwarmDivs())

            self.regroup()

        self.createContextVector()
        return self.contextVector, fitness, diversity

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

    def __init__(self, n_prts, min, max, dim, w, c1, c2):

        self.n_prts = n_prts
        self.min = min
        self.max = max
        self.dim = dim
        self.w = w
        self.c1 = c1 
        self.c2 = c2

        self.pos = np.random.uniform(min, max, (n_prts, dim))
        self.vel = np.zeros((n_prts, dim))
        self.fit = np.zeros(n_prts)

        self.pBestFit = np.zeros(n_prts)
        self.pBestPos = self.pos.copy()
        
        self.gBestFit = 0
        self.gBestPos = np.zeros(n_prts, dim)

    def getRandomPart(self):
        idx = np.random.randint(0, self.n_prts)
        return idx, self.pos[idx]

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

    def diversity(self):
        return (1 / self.n_prts) * np.sum(np.sqrt(np.sum(np.square(self.pos - np.mean(self.pos, axis = 0)), axis = 1)))
    
    def avgFit(self):
        return np.mean(self.fit)