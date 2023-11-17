class Outcome:
    def __init__(self):
        pass

class Observation:
    def __init__(self):
        pass

class Counterfactual:
    def __init__(self,decision_maker,changes={}):
        pass

    def outcome(self,real_data):
        '''
        Return the decision maker's outcome for this counterfactual
        '''
        raise NotImplementedError
    
class CounterfactualExplainer:
    def __init__(self,true_observation,true_outcome,counterfactual):
        self.true_observation = true_observation
        self.true_outcome = true_outcome
        self.CF = counterfactual

    def explain(self):
        pass