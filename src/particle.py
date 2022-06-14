import numpy as np
import random
import math
import copy

class Particle:
    def __init__(self):
        #
        self.weightPosition = []
        self.weightVelocity = []
        self.biasPosition = []
        self.biasVelocity = []
        self.parameters = NN_parameters()
        self.cognitiveCoef = 1 # can be changed
        self.socialCoef = 1 # can be changed
        self.informantList = []
        self.informants_best_err = -1
        self.best_err = -1

    def setInformants(self, swarm, informantNum, index):
        banned_index = []
        i = 0
        swarm_buffer = copy.deepcopy(swarm)
        banned_index.append(index)
        while i < informantNum:
            informant_chosen = np.random.randint(0, len(swarm_buffer))
            if informant_chosen in banned_index:
                continue
            self.informantList.append(swarm[informant_chosen])
            banned_index.append(informant_chosen)
            i += 1

    def set_informant_best(self):
        for informer in self.informantList:
            if informer.best_err < self.informants_best_err or self.informants_best_err == -1:
                self.informants_best_err = informer.best_err
                self.informants_best = informer.best_wb

    def update_velocity(self):
        inertia_weight = 1
        for i in range(CONNECTION_NUMBER):
            r1 = np.random.randn(4, self.num_neurons)
            r2 = np.random.randn(4, self.num_neurons)

            vel_cog = self.cognitive * r1 * (self.best_wb[0] - WEIGHT_LIST[0])      # find how to obtain the synapse weights from pytorch
            vel_soc = self.social * r2 * (self.informants_best[0] - WEIGHT_LIST[0]) # find how to obtain the synapse weights from pytorch
            self.velW1 = inertia_weight * self.velW1 + vel_soc + vel_cog

            #Change the velocity values for B1
            r1 = np.random.randn(1, self.num_neurons)
            r2 = np.random.randn(1, self.num_neurons)

            vel_cog = self.cognitive * r1 * (self.best_wb[1] - self.B1)
            vel_soc = self.social * r2 * (self.informants_best[1] - self.B1)
            self.velB1 = inertia_weight * self.velB1 + vel_soc + vel_cog






        CONNECTION_NUMBER = self.parameters.layerNumber - 1 + 2 # -1 hidden connection + 2 input and output connections
        randomVelocity = []

        #Change the velocity values for W1

            

        

        # Change the velocity values for W2
        r1 = np.random.randn(self.num_neurons,1)
        r2 = np.random.randn(self.num_neurons,1)

        vel_cog = self.cognitive * r1 * (self.best_wb[2] - self.W2)
        vel_soc = self.social * r2 * (self.informants_best[2] - self.W2)
        self.velW2 = inertia_weight * self.velW2 + vel_soc + vel_cog

        # Change the velocity values for B2
        r1 = np.random.randn()
        r2 = np.random.randn()

        vel_cog = self.cognitive * r1 * (self.best_wb[3] - self.B2)
        vel_soc = self.social * r2 * (self.informants_best[3] - self.B2)
        self.velB2 = inertia_weight * self.velB2 + vel_soc + vel_cog