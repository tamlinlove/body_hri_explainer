import pandas as pd
# Hide that one annoying pandas warning
pd.options.mode.chained_assignment = None

import numpy as np
import argparse

from hriri.decision_maker import DecisionMaker,ACTION_NAMES
from body_hri_explainer.CounterfactualExplanation import Counterfactual,CounterfactualExplainer,Observation,Outcome

class HRIBodyOutcome(Outcome):
    def __init__(self,target,decision):
        self.target = target
        self.decision = decision

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
        "D":[0.1,0.5,1,1.5,2,2.5,3,4,5,6,7,8,9],
    }

    def __init__(self,body_df,waiting):
        self.body_df = body_df
        self.bodies = list(body_df["Body"].unique())
        self.waiting = waiting

    def print(self):
        print(self.body_df)
        print("WAITING: {}".format(self.waiting))
        

class HRIBodyCounterfactual(Counterfactual):
    def __init__(self,decision_maker,changes={}):
        '''
        Changes:
            {
            "waiting":...,
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
        self.changes = changes

    def outcome(self,observation):
        counterfactual_df = observation.body_df.copy()
        counterfactual_waiting = observation.waiting

        for change in self.changes:
            if change == "Waiting":
                counterfactual_waiting = self.changes[change]
            else:
                idx_list = np.flatnonzero(counterfactual_df['Body'] == change)
                if idx_list.size == 0:
                    # Body not in body df
                    continue
                else:
                    body_idx = idx_list[0]
                    for body_var in self.changes[change]:
                        counterfactual_df.at[body_idx,body_var] = self.changes[change][body_var]
        
        return self.decision_maker.decide(counterfactual_df,counterfactual_waiting)


class HRIBodyExplainer:
    csv_dir = "~/catkin_ws/src/hriri/logging/decision_csvs/"

    def __init__(self,csv_file,decision_maker):
        self.data = pd.read_csv(csv_file)
        self.decision_maker = decision_maker

    def explain(self,row_index,why_not=None,display=True):
        row = self.data.iloc[row_index,:]
        body_df = self.row_to_body_df(row)
        true_decision = row["Decision"]
        true_target = row["Target"]
        waiting = row["Waiting"]

        true_observation = HRIBodyObservation(body_df,waiting)
        true_outcome = HRIBodyOutcome(true_target,true_decision)

        self.display = display
        if self.display:
            self.display_query(true_observation,true_outcome,why_not)

        trivial,text_explanation = self.handle_trivial_cases(true_observation,true_outcome,why_not)
        if trivial:
            print("Explanation: {}".format(text_explanation))
        else:
            cfx = CounterfactualExplainer(true_observation,true_outcome,HRIBodyCounterfactual)
            cfx.explain()

    '''
    
    Trivial cases
    
    '''

    def handle_trivial_cases(self,observation,true_outcome,why_not):
        self.body_indices = self.get_body_indices(self.prune_body_df(observation.body_df),observation.bodies)
        text_explanation = None
        
        if why_not is not None and why_not != [None,None]:
            if why_not[0] == true_outcome.target and why_not[1] == true_outcome.decision:
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
        return body_df


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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", nargs='?', const=1, type=str, default="decision_test.csv")
    parser.add_argument("-r", "--row", nargs='?', const=1, type=int, default=0)
    args = vars(parser.parse_args())

    filename = HRIBodyExplainer.csv_dir + args["file"]
    dm = DecisionMaker()
    exp = HRIBodyExplainer(filename,dm)

    

    exp.explain(args["row"],why_not=[None,None])
