import numpy as np
import random
import math
import copy

class Particle:
    def __init__(self):
        self.hidden_dim = 1
        self.hidden_num = 1
        self.parameters = NN_parameters()
        self.cognitiveCoef = 1 # can be changed
        self.socialCoef = 1 # can be changed
        self.informantList = []
        self.informants_best_err = -1
        self.best_err = -1
        self.velocity_hidden_num = random.random()
        self.velocity_hidden_dim = random.random()

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

        r1 = random.random()
        r2 = random.random()

        vel_cog = self.cognitive * r1 * (self.best_wb[0])
        vel_soc = self.social * r2 * (self.informants_best[0])
        self.velocity_hidden_num = inertia_weight * self.velocity_hidden_num + vel_soc + vel_cog

        #Change the velocity values for B1
        r1 = random.random()
        r2 = random.random()

        vel_cog = self.cognitive * r1 * (self.best_wb[1])
        vel_soc = self.social * r2 * (self.informants_best[1])
        self.velocity_hidden_dim = inertia_weight * self.velocity_hidden_dim + vel_soc + vel_cog

    # Update the weights and biases
    def change_wb(self):
            self.velocity_hidden_num = self.velocity_hidden_num + self.hidden_num
            self.velocity_hidden_dim = self.velocity_hidden_dim + self.hidden_dim
            