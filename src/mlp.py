# mlp.py
# -------------

# mlp implementation
import util


import math # used for sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def deriveSigmoid(self, output):
  return output*(1-output)

PRINT = True

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):
          "*** YOUR CODE HERE ***"
          ## Clasify Trainging data
          ## backpropogation
          
          util.raiseNotDefined()
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      
      
      "*** YOUR CODE HERE ***"
      ## - Kevin, Ely : fill predictions in the guesses list
      
      '''
      # Input -> First Hidden Layer
      vectors_L1 = util.Counter()
      for i in range(0, length_w1):                       # ***** Need to define length_w1 *****
        vectors_L1[i] = self.weights_L1[i] * datum        # each neuron in L1 has a set of weights; dot product operation
        vectors_L1[i] = sigmoid(vectors_L1[i])            # sigmoid function
      '''  
        
      # First Hidden Layer -> Second Hidden Layer
      vectors_L2 = util.Counter()
      for i in range(0, length_w2):                       # ***** Need to define length_w2 *****
        vectors_L2[i] = self.weights_L2[i] * vectors_L1
        vectors_L2[i] = sigmoid(vectors_L2[i])
        
      # Second Hidden Layer -> Output Layer
      vectors_op = util.Counter()
      for i in range(0, 10):
        vectors_op[i] = self.weights_op[i] * vectors_L2
        vectors_op[i] = sigmoid(vectors_op)
      
      guesses.append(vectors_op.argMax())
      ## - End Kevin, Ely      
      util.raiseNotDefined()
    return guesses, vectors_L1, vectors_L2, vectors_op
  
  
  
  
  def backpropogation (self, trainingdata, trainingLabels, vectors_L2, vectors_op, weights_op, LR, weights_L2):
    target = []
    for i in range(0, 10):
      if (trainingLabels == i):
        target.append(1)
      else:
        target.append(0)
    ## calculate ERROR
    
    for i in range(0, vectors_L2.length):
      for j in range(0, trainingdata.length):
        d_err_outputL2 = 0
        for k in range(0, 10):
          d_err_outputL2 += (vectors_op[k]-target[k])*(vectors_op[k]*(1-vectors_op[k]))*weights_op[k][i]
        weights_L2[i][j] = weights_L2[i][j] - LR * d_err_outputL2 * deriveSigmoid(vectors_L2[i])*trainingdata[j]
        
    
      
      
      