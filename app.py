import uvicorn
import numpy as np
from fastapi import FastAPI

import pickle

app=FastAPI()



rgModel= pickle.load(open('rf.pkl', 'rb'))


@app.get('/')
def index():
    return {'message': 'APP DEV TEST'}


@app.get("/predictLoan")
def gePredictLoan(Gender: int,Married: int,Dependents: int,Education:int,Self_Employed: int,ApplicantIncome: int,CoapplicantIncome: int,Loan_Amount_Term : int,Credit_History: int,Property_Area:int ):
    prediction=rgModel.predict([[1,1,3,1,1,4009,1777,360,1,2]])
    #prediction=rgModel.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,Loan_Amount_Term,Credit_History,Property_Area]])
    result = ','.join(str(x) for x in prediction)
    return{'Loan': result}





if __name__ == '__main__':
     uvicorn.run(app, port=80, host='0.0.0.0')
    
    