# mlp.py
# -------------

# mlp implementation
import util


import math # used for sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


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
      
      # Input -> First Hidden Layer
      vectors_L1 = util.Counter()
      for i in range(0, length_w1):                       # ***** Need to define length_w1 *****
        vectors_L1[i] = self.weights_L1[i] * datum        # each neuron in L1 has a set of weights; dot product operation
        vectors_L1[i] = sigmoid(vectors_L1[i])            # sigmoid function
        
        
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
  
  def backpropogation (self, data, trainingLabels, vectors_L1, vectors_L2, vectors_op, ):
    ## calculate ERROR
    output_error = []
    
    Etot = 0
    for i in range(0, 10):
      if trainingLabels == i:
        Etot = Etot + .5*(1 - vectors_op[i])^2
      else
        Etot = Etot + .5*(vectors_op[i])^2
    
    partialEtot_out = []
    partialOuti_net = []
    partialNeti_Wn = []
    partialEtot_Wn = []
    
    for i in range(0, 10):
      partialEtot_out.append(vectors_op[i] - trainingLabels) 
      partialOuti_net.append(vectors_op[i]*(1 - vectors_op[i])) 
      partialNeti_Wn.append()
      partialEtot_Wn[i] = partialEtot_out[i] * partialOuti_net[i] * partialNeti_Wn[i]
    
      
      