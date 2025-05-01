import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","churn.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor1.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        CreditScore: int,
        Age: int,
        Tenure:int,
        Balance: int,
        NumOfProducts: int,
        HasCrCard: int,
        IsActiveMember: int,
        EstimatedSalary: int,
        Geography: str,
        Gender: str):

        self.CreditScore = CreditScore

        self.Age= Age

        self.Tenure = Tenure

        self.Balance = Balance

        self.NumOfProducts = NumOfProducts

        self.HasCrCard = HasCrCard

        self.IsActiveMember = IsActiveMember

        self.EstimatedSalary = EstimatedSalary

        self.Geography = Geography

        self.Gender = Gender

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "NumOfProducts": [self.NumOfProducts],
                "Balance": [self.Balance],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

