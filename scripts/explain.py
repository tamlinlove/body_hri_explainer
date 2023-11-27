import pandas as pd
# Hide that one annoying pandas warning
pd.options.mode.chained_assignment = None

import numpy as np
import argparse
import copy
import itertools
import networkx as nx

from hriri.decision_maker import DecisionMaker,ACTION_NAMES
from hriri.msg import EngagementLevel
from body_hri_explainer.CounterfactualExplanation import Counterfactual,CounterfactualExplainer,Observation,Outcome

class HRIBodyOutcome(Outcome):
    def __init__(self,target,decision):
        self.target = target
        self.decision = decision

    def valid_outcome(self,outcome,why_not):
        target = outcome[1]
        decision = outcome[2]

        if why_not is None or why_not == [None,None]:
            # Valid if different from the true outcome
            return target!=self.target or decision!=self.decision
        else:
            if why_not[0] is None:
                # We care about the decision, not the target
                return decision==why_not[1]
            elif why_not[1] is None:
                # We care about the target, not the decision
                return target==why_not[0]
            else:
                # We care about both
                return decision==why_not[1] and target==why_not[0]


class HRIBodyObservation(Observation):
    general_influences = {
        "Waiting":[False,True]
        }
    robot_influences = {
        "G":[False,True],
        "GC":[0,1,2,3]
    }
    body_influences = {
        "G":[False,True],
        "GC":[0,1,2,3],
        "A":[0,1,2,3],
        "AC":[0,1,2,3],
        "EL":[0,1,2,3,4],
        "ELC":[0,1,2,3],
        "GWR":[False,True],
        "MG":[0,1,2,3],
        "EV":[0,1,2,3],
        "D":[0.1,0.5,1,1.5,2,2.5,3,4,5,6,7,8,9],
    }

    general_influence_types = {
        "Waiting":"Categorical"
    }

    robot_influence_types = {
        "G":"Categorical",
        "GC":"Continuous"
    }

    body_influence_types = {
        "G":"Categorical",
        "GC":"Continuous",
        "A":"Categorical",
        "AC":"Continuous",
        "EL":"Categorical",
        "ELC":"Continuous",
        "GWR":"Categorical",
        "MG":"Continuous",
        "EV":"Continuous",
        "D":"Continuous",
    }


    def __init__(self,body_df,waiting,bodies):
        self.body_df = body_df
        self.bodies = bodies
        self.detected_bodies = [x for x in list(body_df["Body"].unique()) if x!="ROBOT"]
        self.waiting = waiting
        self.body_state = self.get_body_state()
        self.influences  = self.get_influences()
        self.influence_types = self.get_influence_types()
        self.state = self.get_state()
        self.cards = self.get_cards()
        self.causal_graph = self.get_causal_graph()

    def print(self):
        print(self.body_df)
        print("WAITING: {}".format(self.waiting))

    def get_influences(self):
        influences = list(self.general_influences.keys())
        for body in self.detected_bodies:
            # Ignore undetected bodies
            new_infs = ["{}_{}".format(body,x) for x in self.body_influences.keys()]
            influences += new_infs
        influences += ["ROBOT_{}".format(x) for x in self.robot_influences.keys()]
        return influences
    
    def get_influence_types(self):
        influence_types = {}
        for influence in self.influences:
            if influence in self.general_influences:
                influence_types[influence] = self.general_influence_types[influence]
            else:
                inf_list = influence.split("_")
                if inf_list[0] == "ROBOT":
                    influence_types[influence] = self.robot_influence_types[inf_list[1]]
                else:
                    influence_types[influence] = self.body_influence_types[inf_list[1]]
        return influence_types
    
    def get_cards(self):
        cards = {}
        for influence in self.influences:
            if influence in self.general_influences:
                cards[influence] = self.general_influences[influence]
            else:
                inf_list = influence.split("_")
                if inf_list[0] == "ROBOT":
                    cards[influence] = self.robot_influences[inf_list[1]]
                else:
                    cards[influence] = self.body_influences[inf_list[1]]
        return cards


    
    def get_body_state(self):
        state = {"Waiting":self.waiting}
        for body in self.detected_bodies:
            state[body] = {}
            body_row_dict = self.body_df.loc[self.body_df["Body"]==body].reset_index().to_dict()
            for var in self.body_influences:
                state[body][var] = body_row_dict[var][0]
        state["ROBOT"] = {}
        robot_row_dict = self.body_df.loc[self.body_df["Body"]=="ROBOT"].reset_index().to_dict()
        for var in self.robot_influences:
            state["ROBOT"][var] = robot_row_dict[var][0]
        return state
    
    def get_state(self):
        influence_state = {}
        for var in self.influences:
            if var in self.general_influences:
                influence_state[var] = self.body_state[var]
            else:
                var_list = var.split("_")
                influence_state[var] = self.body_state[var_list[0]][var_list[1]]
        return influence_state
    
    def get_state_from_assignment(self,assignment):
        state = {}
        for var,ass_val in zip(self.influences,assignment):
            if var in self.general_influences:
                state[var] = ass_val
            else:
                var_list = var.split("_")
                if var_list[0] not in state:
                    state[var_list[0]] = {}
                state[var_list[0]][var_list[1]] = ass_val
        return state

    def critical_cards(self,var):
        # Same as observation except for var, which is everything other than observation
        
        card_ranges = []
        for ivar in self.influences:
            if ivar == var:
                clist = self.cards[ivar]
                clist.remove(self.state[ivar])
                card_ranges.append(clist)
            else:
                card_ranges.append([self.state[ivar]])
        return card_ranges
    
    def critical_interventions(self,interventions,var):
        intervention_list = []
        clist = self.cards[var].copy()
        clist.remove(self.state[var])
        for changed_val in clist:
            new_intervention = interventions.copy()
            if var in self.general_influences:
                new_intervention[var] = changed_val
            else:
                var_list = var.split("_")
                if var_list[0] not in new_intervention:
                    new_intervention[var_list[0]] = {}
                new_intervention[var_list[0]][var_list[1]] = changed_val
            intervention_list.append(new_intervention)
        return intervention_list
    
    def potential_interventions(self,interventions,var):
        pass

    def critical_interventions_multi(self,interventions,vars):
        intervention_list = []
        clists = []
        for var in vars:
            clists.append(self.cards[var].copy())
            clists[-1].remove(self.state[var])
        asses = list(itertools.product(*clists))
        
        for ass in asses:
            new_intervention = interventions.copy()
            for var,var_val in zip(vars,ass):
                if var in self.general_influences:
                    new_intervention[var] = var_val
                else:
                    var_list = var.split("_")
                    if var_list[0] not in new_intervention:
                        new_intervention[var_list[0]] = {}
                    new_intervention[var_list[0]][var_list[1]] = var_val
            intervention_list.append(new_intervention)
        return intervention_list
    
    def get_causal_graph(self):
        G = nx.DiGraph()
        for body in self.bodies:
            if body != "ROBOT":
                edges = [
                    ("{}_MG".format(body),"{}_EV".format(body)),
                    ("{}_D".format(body),"{}_EV".format(body)),
                    ("{}_EV".format(body),"{}_EL".format(body))
                ]
                G.add_edges_from(edges)
        return G
    
    def get_causal_effect(self,u,v,changes,dm):
        u_list = u.split("_")
        v_list = v.split("_")
        if u_list[0] in self.bodies and v_list[0] == u_list[0]:
            vars = ["D","MG","EV","EL"]
            #var_vals = {k:self.state[k] for k in ["{}_{}".format(u_list[0],v) for v in vars]}
            var_vals = {k:self.state["{}_{}".format(u_list[0],k)] for k in vars}
            
            # Override with any changes
            if u_list[0] in changes:
                for var in vars:
                    if var in changes[u_list[0]]:
                        var_vals[var] = changes[u_list[0]][var]
            
            # Get causal effect
            if u_list[1] in ["D","MG"] and v_list[1] == "EV":
                new_ev = min(1,(var_vals["MG"]/max(self.body_influences["MG"]))/var_vals["D"] if var_vals["D"] != 0 else 0)
                changes[u_list[0]]["EV"] = dm.float_bucket(new_ev)
            elif u_list[1] == "EV" and v_list[1] == "EL":
                # TODO: Implement something smarter here for estimating engagement level
                ev = var_vals["EV"]
                if ev > 0.75:
                    el = EngagementLevel.ENGAGED
                elif ev > 0.5:
                    el = EngagementLevel.ENGAGING
                elif ev > 0.25:
                    el = EngagementLevel.DISENGAGING
                else:
                    el = EngagementLevel.DISENGAGED
                changes[u_list[0]]["EL"] = el



        return changes

            


        
        

class HRIBodyCounterfactual(Counterfactual):
    def __init__(self,decision_maker,intervention_order=[],interventions={},changes={}):
        '''
        intervention_order = ["waiting","aaaaa_G",etc.]

        interventions:
            {
            "Waiting":...,
            "aaaaa":{
                "G":...,
                "GC"...,
                etc.
            },
            "bbbbb":{
                "G":...,
                "GC"...,
                etc.
            },
            etc.
            }
        '''
        self.decision_maker = decision_maker
        self.intervention_order = intervention_order
        self.interventions = interventions
        self.changes = changes
        self.action_names = ACTION_NAMES

    def copy(self):
        return HRIBodyCounterfactual(self.decision_maker,intervention_order=self.intervention_order,interventions=copy.deepcopy(self.interventions),changes=copy.deepcopy(self.changes))
    
    def full_state(self,observation):
        state = copy.deepcopy(observation.body_state)
        for change in self.changes:
            if change == "Waiting":
                state["Waiting"] = self.changes[change]
            else:
                for var in self.changes[change]:
                    state[change][var] = self.changes[change][var]
        return state
    
    def outcome(self,observation,intervention_order=None,interventions=None):
        counterfactual_df = observation.body_df.copy()
        counterfactual_waiting = observation.waiting

        if intervention_order is None:
            intervention_order = self.intervention_order
        if interventions is None:
            interventions = self.interventions

        changes = self.apply_interventions(observation,intervention_order,interventions)
        for change in changes:
            if change == "Waiting":
                counterfactual_waiting = changes[change]
            else:
                idx_list = counterfactual_df.index[counterfactual_df['Body']==change]
                if idx_list.size == 0:
                    # Body not in body df
                    continue
                else:
                    body_idx = idx_list[0]
                    for body_var in changes[change]:
                        counterfactual_df.at[body_idx,body_var] = changes[change][body_var]

        return self.decision_maker.decide(counterfactual_df,counterfactual_waiting)
    
    def apply_interventions(self,observation,intervention_order,interventions,causal=True):
        changes = {}
        if causal:
            # TODO: Add causal reasoning here
            causal_graph = observation.causal_graph.copy()
            
            for intrv in intervention_order:
                if intrv in causal_graph.nodes:
                    # In the order of interventions, remove the edges from parents of intervened node
                    parents = list(causal_graph.predecessors(intrv))
                    for par in parents:
                        causal_graph.remove_edge(par,intrv)
                    # Apply causal effects
                    changes = self.apply_change(changes,interventions,observation,intrv)
                    changes = self.apply_causal_effects(causal_graph,changes,observation,intrv)
                else:
                    # Just apply regularly
                    changes = self.apply_change(changes,interventions,observation,intrv)
        else:
            for intrv in intervention_order:
                changes = self.apply_change(changes,interventions,observation,intrv)
        return changes
    
    def apply_causal_effects(self,causal_graph,changes,observation,intrv):
        children = causal_graph.successors(intrv)
        for child in children:
            changes = observation.get_causal_effect(intrv,child,changes,self.decision_maker)
        # Recursively apply down the graph
        for child in children:
            changes = self.apply_causal_effects(causal_graph,changes,observation,child)

        return changes

    def apply_change(self,changes,interventions,observation,var):
        if var in observation.general_influences:
            changes[var] = interventions[var]
        else:
            intrv_list = var.split("_")
            if intrv_list[0] not in changes:
                changes[intrv_list[0]] = {}
            changes[intrv_list[0]][intrv_list[1]] = interventions[intrv_list[0]][intrv_list[1]]
        return changes
    
    def in_interventions(self,influence_var):
        if influence_var in self.interventions:
            return True
        
        vlist = influence_var.split("_")
        if len(vlist)==2 and vlist[0] in self.interventions and vlist[1] in self.interventions[vlist[0]]:
            return True
        
        return False




class HRIBodyExplainer:
    csv_dir = "~/catkin_ws/src/hriri/logging/decision_csvs/"

    def __init__(self,csv_file,decision_maker):
        self.data = pd.read_csv(csv_file)
        self.decision_maker = decision_maker

    def explain(self,row_index,why_not=None,display=True):
        row = self.data.iloc[row_index,:]
        body_df,bodies = self.row_to_body_df(row)
        true_decision = row["Decision"]
        true_target = row["Target"]
        waiting = row["Waiting"]

        true_observation = HRIBodyObservation(self.prune_body_df(body_df),waiting,bodies)
        true_outcome = HRIBodyOutcome(true_target,true_decision)

        self.display = display
        if self.display:
            self.display_query(true_observation,true_outcome,why_not)

        trivial,text_explanation = self.handle_trivial_cases(true_observation,true_outcome,why_not)
        if trivial:
            print("Explanation: {}".format(text_explanation))
        else:
            cfx = CounterfactualExplainer(true_observation,true_outcome,HRIBodyCounterfactual,self.decision_maker)
            cfx.explain(why_not)

    '''
    
    Trivial cases
    
    '''

    def handle_trivial_cases(self,observation,true_outcome,why_not):
        self.body_indices = self.get_body_indices(observation.body_df,observation.bodies)
        text_explanation = None
        
        if why_not is not None and why_not != [None,None]:
            if (why_not[0] == true_outcome.target or (why_not[0] is None and np.isnan(true_outcome.target))) and why_not[1] == true_outcome.decision:
                text_explanation = "Your query is exactly the decision the robot made"
            
            if why_not[0] is not None:
                if why_not[0] not in observation.bodies:
                    text_explanation = "The body {} is not recognised".format(why_not[0])
                elif self.body_indices[why_not[0]] is None:
                    text_explanation = "The robot did not select target {0} because {0} was not detected at the time".format(why_not[0])
                elif why_not[0] == "ROBOT":
                    text_explanation = "The robot cannot target itself"

        return text_explanation is not None,text_explanation
    

    '''
    
    Display    
    
    '''

    def display_query(self,observation,true_outcome,why_not):
        print("In the queried instance, the robot observed:")
        observation.print()
        print("The robot decided to take action {} with target {}".format(ACTION_NAMES[int(true_outcome.decision)],true_outcome.target))

        query_text = ""
        if why_not is None or why_not == [None,None]:
            query_text = "why did the robot take action {} on target {}".format(ACTION_NAMES[int(true_outcome.decision)],true_outcome.target)
        else:
            if why_not[0] is None and why_not[1] is not None:
                # Any target, specific decision
                query_text = "why did the robot take action {} instead of action {}".format(ACTION_NAMES[int(true_outcome.decision)],ACTION_NAMES[why_not[1]])
            elif why_not[0] is not None and why_not[1] is None:
                # Specific target, any decision
                query_text = "why did the robot choose {} instead of {}".format(true_outcome.target,why_not[0])
            elif why_not[0] is not None and why_not[1] is not None:
                # Specific both
                query_text = "why did the robot take action {} on target {} and not action {} on target {}".format(ACTION_NAMES[int(true_outcome.decision)],true_outcome.target,ACTION_NAMES[why_not[1]],why_not[0])
        print("Your query: {}".format(query_text))

    '''
    
    Data Processing
    
    '''

    def row_to_body_df(self,row):
        '''
        Convert the row to the body_df format expected by the decision maker
        '''
        bodies = self.get_bodies(row)
        headers = ["Body Time","Dec Time","Body","Group","G","GC","A","AC","EL","ELC","MG","D","EV","GWR"]
        bdf_list = []
        for body in bodies:
            if body != "ROBOT":
                bdf_list.append([
                    row["Body Time"],
                    row["Dec Time"],
                    body,
                    row["{}_Group".format(body)],
                    row["{}_G".format(body)],
                    row["{}_GC".format(body)],
                    row["{}_A".format(body)],
                    row["{}_AC".format(body)],
                    row["{}_EL".format(body)],
                    row["{}_ELC".format(body)],
                    row["{}_MG".format(body)],
                    row["{}_D".format(body)],
                    row["{}_EV".format(body)],
                    row["{}_GWR".format(body)],
                ])
            else:
                bdf_list.append([
                    row["Body Time"],
                    row["Dec Time"],
                    body,
                    row["{}_Group".format(body)],
                    row["{}_G".format(body)] if not pd.isna(row["{}_G".format(body)]) else False,
                    row["{}_GC".format(body)] if not pd.isna(row["{}_GC".format(body)]) else 0,
                    0,
                    3,
                    0,
                    3,
                    0,
                    0,
                    0,
                    True,
                ])
        body_df = pd.DataFrame(bdf_list,columns=headers)
        return body_df,bodies


    def get_bodies(self,row):
        return [b.split("_")[0] for b in row.index if b.endswith("_Group")]
    
    def prune_body_df(self,body_df):
        pruned_df = body_df.copy().loc[((body_df["Group"].notnull()) | (body_df["Body"] == "ROBOT"))]
        return pruned_df
    
    def get_body_indices(self,df,bodies):
        body_indices = {}
        for body in bodies:
            idx_list = np.flatnonzero(df['Body'] == body)
            if idx_list.size == 0:
                body_indices[body] = None
            else:
                body_indices[body] = idx_list[0]

        return body_indices
    
    def get_decision_indices(self,exclude=[]):
        indices = []
        for i in range(len(ACTION_NAMES)):
            if i in exclude:
                indices.append(None)
            else:
                indices.append(self.data.index[self.data['Decision']==i].tolist())
        return indices


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", nargs='?', const=1, type=str, default="UntitledExperiment")
    parser.add_argument("-r", "--row", nargs='?', const=1, type=int, default=0)
    args = vars(parser.parse_args())

    filename = HRIBodyExplainer.csv_dir + args["file"] + ".csv"
    dm = DecisionMaker()
    exp = HRIBodyExplainer(filename,dm)

    print(exp.get_decision_indices(exclude=[1]))

    

    exp.explain(args["row"],why_not=[None,None])
    #exp.explain(args["row"],why_not=[None,5])
    exp.explain(args["row"],why_not=[None,4])
    exp.explain(args["row"],why_not=["vxmre",None])
