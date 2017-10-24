# mlp.py
# -------------

# mlp implementation
import util


import math # used for sigmoid function
def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x/10))

def deriveSigmoid(output):
  return output*(1-output)

PRINT = True

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations, length_L2, lr):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
    self.length_L2 = length_L2
    self.learningrate = lr
    self.weights_L2 = {}
    self.weights_op = {}
    for i in range(0,length_L2):
      self.weights_L2[i] = util.Counter()
    for label in legalLabels:
      self.weights_op[label] = util.Counter()
      
  def setWeights(self, w_L2, w_op):
    assert len(w_op) == len(self.legalLabels);
    assert len(w_L2) == self.length_L2;
    self.weights_op = w_op;
    self.weights_L2 = w_L2;
  
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      
      guesses, values_L2, values_op = MLPClassifier.classify(self, trainingData)
      
      print guesses
    
      MLPClassifier.backpropogation(self, trainingData, trainingLabels, guesses, values_L2, values_op)
    
      for i in range(len(trainingData)):
          "*** YOUR CODE HERE ***"
          ## Clasify Trainging data
          ## backpropogation
          
          #util.raiseNotDefined()
    
  def classify(self, data):
    guesses = []
    values_L2 = []
    values_op = []
    for n in range(0, len(data)):
      "*** YOUR CODE HERE ***"
      ## - Kevin, Ely : fill predictions in the guesses list
      '''
      # Input -> First Hidden Layer
      vectors_L1 = util.Counter()
      for i in range(0, length_w1):                       # ***** Need to define length_w1 *****
        vectors_L1[i] = self.weights_L1[i] * datum        # each neuron in L1 has a set of weights; dot product operation
        vectors_L1[i] = sigmoid(vectors_L1[i])            # sigmoid function
      '''      # Input Layer -> Second Hidden Layer
      vectors_L2 = util.Counter()
      for i in range(0, self.length_L2):
        vectors_L2[i] = self.weights_L2[i] * data[n]       
        vectors_L2[i] = sigmoid(vectors_L2[i])
        
      values_L2.append(vectors_L2)
        
      # Second Hidden Layer -> Output Layer
      vectors_op = util.Counter()
      for i in range(0, 10):
        vectors_op[i] = self.weights_op[i] * vectors_L2
        vectors_op[i] = sigmoid(vectors_op[i])
      
      values_op.append(vectors_op)
            
      guesses.append(vectors_op.argMax())
      ## - End Kevin, Ely      
      ## util.raiseNotDefined()
    return guesses, values_L2, values_op

  
  def backpropogation (self, trainingdata, trainingLabels, guesses, vectors_L2, vectors_op):
    
    for n in range(0, len(trainingdata)):
      if guesses[n] == trainingLabels[n]:
        pass
      else:
        target = []
        for i in range(0, 10):
          if (trainingLabels == i):
            target.append(1)
          else:
            target.append(0)
        ## calculate ERROR
        for i in range(0, 10): # for each output neuron
          #print("Before: ", self.weights_op[i][1])
          for j in range(0, self.length_L2): # for each neuron in hidden layer
            self.weights_op[i][j] -= self.learningrate * (-(target[i]-vectors_op[n][i]) * vectors_op[n][i]*(1-vectors_op[n][i]) * vectors_L2[n][j]) 
          #print("After: ", self.weights_op[i][1])
        
        for i in range(0, self.length_L2):
          for j in range(0, len(trainingdata)):
            
            d_err_outputL2 = 0
            
            for k in range(0, 10):
              d_err_outputL2 += (vectors_op[n][k]-target[k]) * (vectors_op[n][k] * (1-vectors_op[n][k])) * self.weights_op[k][i]
            
            self.weights_L2[i][j] -= self.learningrate * d_err_outputL2 * deriveSigmoid(vectors_L2[n][i]) * trainingdata[n][j]

          
        
        
## ----- UNIT TEST ----- ##

## Take and Make Input Flat
from samples import loadDataFile
from samples import loadLabelsFile
from samples import Datum
import random
def FlatInput(n, items):
  flat_items = [[] for _ in range(n)]
  for m in range(0, n): 
    for i in range(0, 28):
      for j in range(0, 28):
        flat_items[m].append(items[m].getPixel(i, j))
  return flat_items
       
def callData():
    n = 100
    hidden_neurons = 25
    #items = loadDataFile("data/digitdata/trainingimages", n,28,28)
    #for item in items:
      #items = util.Counter()
    
    items = loadDataFile("data/digitdata/trainingimages", n,28,28)
    flat_item = FlatInput(n, items)

    
    trainingData = {}
    for i in range(len(flat_item)):
      trainingData[i] = util.Counter()
      for j in range(len(flat_item[i])):
        trainingData[i][j] = flat_item[i][j]
        
    
    labels = loadLabelsFile("data/digitdata/traininglabels", n)
    
    val_items = loadDataFile("data/digitdata/validationimages", n,28,28)
    flat_val = FlatInput(n, val_items)
    
    validationData = {}
    for i in range(len(flat_val)):
      validationData[i] = util.Counter()
      for j in range(len(flat_val[i])):
        validationData[i][j] = flat_val[i][j]
    
    val_labels = loadLabelsFile("data/digitdata/validationlabels", n)
    data = MLPClassifier( [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 50, hidden_neurons, .5)
    
    weights_op = {}
    for w in range(0, 10): 
      weights_op[w] = util.Counter()
      for i in range(0,784):
        weights_op[w][i] = random.random()
    
    weights_L2 = {}
    for w in range(0, hidden_neurons):
      weights_L2[w] = util.Counter()
      for i in range(0, 784):
        weights_L2[w][i] = random.random()
            
    data.setWeights(weights_L2, weights_op)
    
    data.train(trainingData, labels, validationData, val_labels)

## Unit Test
callData()
      
      