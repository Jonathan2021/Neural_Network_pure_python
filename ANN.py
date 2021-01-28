"""
Pure Python feed forward neural network from scratch. Uses SGD for stepping weights
"""



import random
from math import exp, sin, pow, floor

class Activation:
    def forward(self, x):
        pass
    def derivate(self, x):
        pass
    def __call__(self, x):
        return self.forward(x)

class IdenticalActivation(Activation):
    def forward(self, x):
        return x
    
    def derivate(self, x):
        return 1

class ReLUActivation(Activation):
    def forward(self, x):
        return max(0, x)
    
    def derivate(self, x):
        if x > 0:
            return 1
        else:
            return 0

def mean_reduce(x):
    return sum(x) / len(x)

def sum_reduce(x):
    return sum(x)

class Loss:
    def __init__(self, reducer  = mean_reduce):
        self.reducer = reducer
    def __call__(self, expected, output):
        pass

    def derivate(self):
        pass

class SquarredErrorLoss(Loss):
    def __call__(self, expected, output):
        if not isinstance(expected, list):
            expected = [expected]
        return [(0.5 * ((exp - out) ** 2)) for exp, out in zip(expected, output)]

    def derivate(self, expected, output):
        if not isinstance(expected, list):
            expected = [expected]
        return [(out - exp) for exp, out in zip(expected, output)]

class SigmoidActivation(Activation):
    def forward(self, x):
        return (1.0 / (1.0 + exp(-x)))

    def derivate(self, x):
        return x * (1.0 - x)

class Neuron:
    def __init__(self, noOfInputs, activationFunction):
        self.noOfInputs = noOfInputs
        self.activationFunction = activationFunction
        self.weights = [random.random() for i in range(self.noOfInputs)]
        self.gradients = [0 for i in range(self.noOfInputs)]
        self.output = 0
        
    def setWeights(self, newWeights):
        self.weights = newWeights
    
    def reset_gradients(self):
        self.gradients = [0 for i in range(self.noOfInputs)]

    def fireNeuron(self, inputs):
        z = sum([x*y for x,y in zip(inputs,self.weights)])
        self.output = self.activationFunction(z)
        return(self.output)
        
    def __str__(self):
        return str(self.weights)

class Layer:
    def __init__(self, noOfInputs, activationFunction, noOfNeurons, bias):
        self.noOfNeurons = noOfNeurons
        self.activationFunction = activationFunction
        if bias:
            noOfInputs += 1
        self.neurons = [Neuron(noOfInputs, activationFunction) for i in 
                      range(self.noOfNeurons)]
        self.bias = bias
        
    
    def forward(self, inputs):
        if self.bias:
            inputs.append(1)
        for x in self.neurons:
            x.fireNeuron(inputs)
        return([x.output for x in self.neurons])

    def __str__(self):
        s = ''
        for i in range(self.noOfNeurons):
            s += ' n '+str(i)+' '+str(self.neurons[i])+'\n'
        return s
        
class FirstLayer(Layer):
    def __init__(self, noOfNeurons, bias = False):
        Layer.__init__(self, 1, IdenticalActivation(), noOfNeurons, bias)
        for x in self.neurons:
            x.setWeights([1])
            
    def forward(self, inputs):
        for i in range(self.noOfNeurons):
            self.neurons[i].fireNeuron([inputs[i]])
        return([x.output for x in self.neurons])
        # return inputs
        
class Network:
    def __init__(self, structure, activationFunctions, bias = True):
        self.activationFunctions = activationFunctions
        self.bias = bias
        self.original_structure = structure[:]
        self.structure = structure[:]
        self.noLayers = len(self.structure)
        self.layers = [FirstLayer(self.structure[0])]
        for i in range(1, self.noLayers):
            self.layers = self.layers + [Layer(self.structure[i-1],
                            activationFunctions[i-1], self.structure[i], self.bias)]
        
    def feedForward(self, inputs):
        self.signal = inputs[:]
        for l in self.layers:
            self.signal = l.forward(self.signal)
        return self.signal

    def backPropag(self, loss_der):
        err = loss_der
        delta = []
        currentLayerIndex = self.noLayers-1

        currentLayer = self.layers[currentLayerIndex]
        for i in range(self.structure[currentLayerIndex]):
            delta.append(err[i] * currentLayer.activationFunction.derivate(currentLayer.neurons[i].output))
            
            nbInputs = self.structure[currentLayerIndex-1] + (1 if self.bias else 0)
            for r in range(nbInputs):
                output_prev = (1 if self.bias and r == nbInputs - 1 else self.layers[currentLayerIndex-1].neurons[r].output)
                currentLayer.neurons[i].gradients[r] += delta[i] * output_prev 

        
        #propagate the errors layer by layer except first (which has weights = 1 to copy input)
        for currentLayerIndex in range(self.noLayers-2,0,-1):
            currentLayer = self.layers[currentLayerIndex]
            currentDelta = []
            for i in range(self.structure[currentLayerIndex]):
                currentDelta.append(currentLayer.activationFunction.derivate(currentLayer.neurons[i].output)* sum(
                    [self.layers[currentLayerIndex+1].neurons[j].weights[i]*delta[j] for j in range(self.structure[currentLayerIndex+1])]))
            
            delta = currentDelta [:]
            for i in range(self.structure[currentLayerIndex]):
                nbInputs = self.structure[currentLayerIndex-1] + (1 if self.bias else 0)
                for r in range(nbInputs):
                    output_prev = (1 if self.bias and r == nbInputs - 1 else self.layers[currentLayerIndex-1].neurons[r].output)
                    currentLayer.neurons[i].gradients[r] += delta[i]* output_prev
        
    def __str__(self):
        s = ''
        for i in range(self.noLayers):
            s += ' l  '+str(i)+' :\n'+str(self.layers[i])
        return s

class schedule_learning:
    def __init__(self, initial):
        self._epoch = 0
        self._initial = initial
        self._learning_rate = initial

    def next(self):
        self._epoch += 1

    def get(self):
        return self._learning_rate

class constantLearningRate(schedule_learning):
    pass

class step_decay(schedule_learning):
    def __init__(self, initial, drop = 0.5, epoch_drop = 100):
        super().__init__(initial)
        self._drop = drop
        self._epoch_drop = epoch_drop

    def next(self):
        super().next()
        self._learning_rate = self._initial * pow(self._drop,  
           floor((self._epoch)/self._epoch_drop))


class Learner:
    def __init__(self, training_data, validation_data, network, lossFunc, nb_epoch, batch_size = 32, learning_rate_scheduler=constantLearningRate(0.001), verbose = True):
        self.training_data = training_data
        self.validation_data = validation_data
        self.network = network
        self.lossFunc = lossFunc
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate_scheduler = learning_rate_scheduler
        self.verbose = verbose

    def epoch(self, cur_epoch):
        losses_train = []
        if self.verbose:
            print("******** ", cur_epoch, " / ", self.nb_epoch, " *********")

        batch_fill = 0
        losses_train = None

        for count, (features, targets) in enumerate(self.training_data):
            outputs = self.network.feedForward(features)
            loss = self.lossFunc(targets, outputs)
            deriv = self.lossFunc.derivate(targets, outputs)
            self.network.backPropag(deriv)

            if losses_train is None:
                losses_train = [[el] for el in loss]
            else:
                for i, item in enumerate(loss):
                    losses_train[i].append(item)

            batch_fill += 1
            
            if batch_fill == self.batch_size or count + 1 == len(self.training_data):
                batch_fill = 0
                self.SGD()


        if self.verbose:
            avg_loss_train = [mean_reduce(losses) for losses in losses_train]
            print(f"Avergage loss training: {avg_loss_train}")
            losses_valid = self.get_losses(self.validation_data)
            avg_loss_val = [mean_reduce(losses) for losses in losses_valid]
            print(f"Avergage loss validation: {avg_loss_val}")

    def SGD(self):
        lr = self.learning_rate_scheduler.get()
        for layer in self.network.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] -= lr * neuron.gradients[i] / self.batch_size
                neuron.reset_gradients()

    

    def get_losses(self, data):
        losses = None
        for features, targets in data:
            loss = self.lossFunc(targets, self.network.feedForward(features))
            if losses is None:
                losses = [[el]for el in loss]
            else:
                for i, item in enumerate(loss):
                    losses[i].append(item)
        return losses

    def train(self):
        cur_epoch = 1
        while cur_epoch <= self.nb_epoch:
            self.epoch(cur_epoch)
            self.learning_rate_scheduler.next()
            random.shuffle(self.training_data) # Prevents from training on same batches each time
            cur_epoch += 1

"""
The following is application specific, just a test case
"""

def get_features_and_target_line(line):
    res = []
    entries = [float(x) for x in line.split(' ')]
    return entries[:-1], entries[-1]

def get_list_features_target_tuples_from_file(path):
    data = []
    with open(path, "r") as data_file:
        for line in data_file:
            features, targets = get_features_and_target_line(line)
            data.append((features, targets))
    return data

def split_list_in_two(arr, ratio):
    cut_point = int(len(arr) * ratio)
    return arr[:cut_point], arr[cut_point:]

def get_path_from_user(default):
    print("Enter the path to read data from.")
    print(f"Default is '{default}', press Enter to keep")
    entry = input()
    return entry if entry else default

def main():
    path = get_path_from_user('dataset.txt')
    print("getting file from path")
    data_from_file = get_list_features_target_tuples_from_file(path)
    print("splitting in half for building and testing model")
    training_data, testing_data = split_list_in_two(data_from_file, 0.8)
    training_data, validation_data = split_list_in_two(training_data, 0.7)
    print("Building Network and Learner")
    network = Network([len(training_data[0][0]), 4, 1], [ReLUActivation(), IdenticalActivation()])
    learner = Learner(training_data, validation_data, network, SquarredErrorLoss(), 500, learning_rate_scheduler = constantLearningRate(0.0001), batch_size=32)
    print("Training")
    learner.train()
    print("testing")
    test_losses = learner.get_losses(testing_data)
    avg_test_loss = [mean_reduce(losses) for losses in test_losses]
    print("\n****** PRINTING NETWORK *******\n")
    print(network)
    print(f"\nAverage Test loss: {avg_test_loss}")

if __name__ == '__main__':
    main()
