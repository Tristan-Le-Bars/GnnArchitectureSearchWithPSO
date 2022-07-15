from matplotlib import pyplot as plt
from particle import Particle
from
from tensorboardX import SummaryWriter

class Swarm:
    def __init__(self, informants_number, particle_number):

        self.informants_number = informants_number
        self.swarm = list()
        # self.cost_func = cost_func
        # self.neurons_number = neurons_number
        self.particle_number = particle_number
        # self.input = input 
        # self.label = label

        for i in range(self.particle_number):
            new_particle = Particle()
            self.swarm.append(new_particle)
        
        for j in range(len(self.swarm)):
            self.swarm[j].setInformants(self.swarm, self.informants_number, j)
        

    def Optimise(self):
        #Run Optimisation
        for p in range(0,self.particle_number):
            # Find best informants
            self.swarm[p].set_informant_best()
            # Update velocities
            self.swarm[p].change_vel()
            # Apply velocities to weights and biases
            self.swarm[p].change_wb()

    # For every particle, creates a neural network from the particles weights and biases. And than calculate the output values of the neural network
    def forward_prop(self, dataset, task):
        for p in range(0, int(self.particle_number)):
            writer = SummaryWriter("./log/" + "|hidden_num = "
                                    + str(self.swarm[p].informants_best[0])
                                    +"|hidden_dim = "
                                    + str(self.swarm[p].informants_best[1]) +"|")
            dataset = dataset.shuffle()
            task = 'graph'

            model = train(dataset, task, writer, 2, 32)

    # Return the best error of the entire swarn 
    def get_best(self):
        swarm_best_err = -1
        for p in self.swarm:
            if p.best_err < swarm_best_err or swarm_best_err == -1:
                swarm_best_err = p.best_err

        return swarm_best_err

    def plot(self, y):
        plt.plot(y)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0,100])
        plt.show()