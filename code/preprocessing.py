import pandas as pd
import numpy as np


class ProcessingData():
    def __init__(self,data:pd.DataFrame,split:list=[0.8,0.2],**args):
        self.DataFrame = data
        self.split = split
        self.RandomCrossValidation = args.get("Random","Random")

    def splitdata(self,data,split) -> tuple:
        
        