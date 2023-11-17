class Outcome:
    def __init__(self):
        pass

class Observation:
    def __init__(self):
        pass

class Counterfactual:
    def __init__(self,decision_maker,changes={}):
        pass

    def copy(self):
        raise NotImplementedError

    def outcome(self,real_data):
        '''
        Return the decision maker's outcome for this counterfactual
        '''
        raise NotImplementedError
    
class Explanation:
    def __init__(self,potential_influences={},critical_influence=None):
        self.potential_influences = potential_influences
        self.critical_influence = critical_influence
    
class CounterfactualExplainer:
    def __init__(self,true_observation,true_outcome,counterfactual,decision_maker):
        self.true_observation = true_observation
        self.true_outcome = true_outcome
        self.CF = counterfactual
        self.decision_maker = decision_maker

    def explain(self):
        influences = self.true_observation.get_influences()

        complete_explanations,partial_explanations = self.explain_case(influences)

    def explain_case(self,influences):
        # BIG TODO
        print(self.true_observation.get_state())

        return None,None




        
