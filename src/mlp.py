# mlp.py
# -------------

# mlp implementation
import util
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
          util.raiseNotDefined()
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      
      
      "*** YOUR CODE HERE ***"
      ## - Kevin, Ely : fill predictions in the guesses list
      
      # Input -> First Hidden Layer
      vectors_L1 = util.Counter()
      for l in range(0, length_w1):             # 
        vectors_L1[l] = self.weights[l] * datum # weights is multidimensional 
        
      # First Hidden Layer -> Second Hidden Layer
      vectors_L2 = util.Counter()
      
      
      ## - End Kevin, Ely
      util.raiseNotDefined()
    return guesses