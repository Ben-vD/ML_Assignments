import numpy as np
from tqdm import tqdm

class mPSO:

    def __init__(self, n_prts, n_itr, min, max, dim, objFunc, w, c1, c2, nr):

        self.n_prts = n_prts
        self.n_itr = n_itr
        self.min = min
        self.max = max
        self.dim = dim
        self.objFunc = objFunc

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.nr = nr

        self.nf = int((n_itr) / (1 + ((np.log(dim)) / (np.log(nr)))))

        self.dimsArr = np.arange(dim)
        np.random.shuffle(self.dimsArr)

        self.n_subSwarms = dim
        self.subSwarmDimsIdxs = np.arange(dim + 1)
        self.subSwarmSizes = np.ones(dim)
        self.subSwarms = np.empty(dim, dtype = object)

        for i in range(dim):
            self.subSwarms[i] = subSwarm(n_prts, min, max, 1, w, c1, c2)

        self.contextVector = np.zeros(dim)
        self.initFit()

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

    def run(self):

        merged = False

        diversity = []
        diversity.append(self.subSwarmDivs())

        fitness = []
        fitness.append(self.subSwarmFit())

        for i in tqdm(range(1, self.n_itr + 1), desc="Iterations"):

            self.evalFitness(merged)
            merged = False
            for j, ss in enumerate(self.subSwarms):
                ss.updateVelPos()

            fitness.append(self.subSwarmFit())
            diversity.append(self.subSwarmDivs())

            if (i % self.nf == 0):
                self.merge()
                merged = True
        
        self.createContextVector()
        return self.contextVector, fitness, diversity
    
    def merge(self):
        
        new_nSubSwarms = self.n_subSwarms // self.nr
        remainder = self.n_subSwarms % self.nr
        if (remainder > 0):
            new_nSubSwarms += 1

        subSwarmMergeIdxs = np.ones(new_nSubSwarms, dtype = int) * self.nr
        if (remainder > 0):
            subSwarmMergeIdxs[np.random.randint(0, len(subSwarmMergeIdxs))] = remainder

        new_subSwarmSizes = np.zeros(new_nSubSwarms, dtype = int)
        idx = 0
        for i, mergingCount in enumerate(subSwarmMergeIdxs):
            for j in range(mergingCount):
                new_subSwarmSizes[i] += self.subSwarmSizes[idx]
                idx += 1

        newSubSwarms = np.empty(new_nSubSwarms, dtype = object)
        idx = -1
        for i, newSubSwarmSize in enumerate(new_subSwarmSizes):

            idx += 1

            subSwarmPos = self.subSwarms[idx].pos.copy()
            subSwarmVel = self.subSwarms[idx].vel.copy()
            subSwarmFit = self.subSwarms[idx].fit.copy()

            subSwarm_pBestPos = self.subSwarms[idx].pBestPos.copy()
            subSwarm_pBestFit = self.subSwarms[idx].pBestFit.copy()
            subSwarm_gBestPos = self.subSwarms[idx].gBestPos.copy()
            subSwarm_gBestFit = self.subSwarms[idx].gBestFit.copy()

            newSubSwarms[i] = subSwarm(self.n_prts, self.min, self.max, newSubSwarmSize, self.w, self.c1, self.c2)

            for j in range(subSwarmMergeIdxs[i] - 1):
                
                idx += 1
                subSwarmPos = np.concatenate([subSwarmPos, self.subSwarms[idx].pos.copy()], axis = 1)
                subSwarmVel = np.concatenate([subSwarmVel, self.subSwarms[idx].vel.copy()], axis = 1)
                subSwarm_pBestPos = np.concatenate([subSwarm_pBestPos, self.subSwarms[idx].pBestPos.copy()], axis = 1)
                subSwarm_gBestPos = np.concatenate([subSwarm_gBestPos, self.subSwarms[idx].gBestPos.copy()])
                
                subSwarmFit = np.zeros(subSwarmPos.shape) #np.maximum(subSwarmFit, self.subSwarms[idx].fit.copy())
                subSwarm_pBestFit = np.zeros(subSwarmPos.shape) #np.maximum(subSwarm_pBestFit, self.subSwarms[idx].pBestFit.copy())
                subSwarm_gBestFit = 0

            newSubSwarms[i].pos = subSwarmPos.copy()
            newSubSwarms[i].vel = subSwarmVel.copy()
            newSubSwarms[i].fit = subSwarmFit.copy()

            newSubSwarms[i].pBestPos = subSwarm_pBestPos.copy()
            newSubSwarms[i].pBestFit = subSwarm_pBestFit.copy()
            newSubSwarms[i].gBestPos = subSwarm_gBestPos.copy()
            newSubSwarms[i].gBestFit = subSwarm_gBestFit

        self.n_subSwarms = new_nSubSwarms
        self.subSwarmSizes = new_subSwarmSizes.copy()
        self.subSwarms = newSubSwarms.copy()

        self.getSubSwarmDimsIdxs()

    def getSubSwarmDimsIdxs(self):

        newSubSwarmDimsIdxs = np.zeros(self.n_subSwarms + 1, dtype = int)
        for i, subSwarmSize in enumerate(self.subSwarmSizes):
            newSubSwarmDimsIdxs[i + 1] = subSwarmSize + newSubSwarmDimsIdxs[i]

        self.subSwarmDimsIdxs = newSubSwarmDimsIdxs.copy()

    def createContextVector(self):
        for i, ss in enumerate(self.subSwarms):
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]
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

        # Evaluate new fitness values
        for i, ss in enumerate(self.subSwarms):

            subSwarmPos = ss.pos
            fitnessVals = np.zeros(self.n_prts)
            subSwarmDims = self.dimsArr[self.subSwarmDimsIdxs[i] : self.subSwarmDimsIdxs[i + 1]]

            for j, p in enumerate(subSwarmPos):
                self.contextVector[subSwarmDims] = p
                fitnessVals[j] = self.objFunc(self.contextVector)

            ss.updateFit(fitnessVals)
            self.contextVector[subSwarmDims] = ss.gBestPos

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

        self.pBestPos = self.pos.copy()
        self.pBestFit = np.zeros((n_prts, dim))

        self.gBestPos = np.zeros(dim)
        self.gBestFit = 0

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
    
    def getRandomPart(self):
        idx = np.random.randint(0, self.n_prts)
        return idx, self.pos[idx]