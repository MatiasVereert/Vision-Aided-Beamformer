# in this code i will translate the functicional wpe to objet oriented aproach



class WPE_online:

  def __init__( self, taps, delay, delta ):
    self.K = taps
    self.delay = delay
    self.delta = delta 

    self.buffer = np.zeros()
    self.G
    
  