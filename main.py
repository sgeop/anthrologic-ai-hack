import numpy as np
import pandas as pd
import dspy
from typing import Literal

import loader


class Tagged(dspy.Signature):
    body: str = dspy.InputField()
    tag: Literal[
        "Side Effects"
        "Product Issues",
        "Account & Orders",
        "Customer Service",
        "Product Information",
        "System & Administrative"
    ] = dspy.OutputField()
    confidence: float = dspy.OutputField()



def main():
    print("Hello from ai-hack!")

    lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)
    
    classify = dspy.Predict(Tagged)

    # load tickets csv and store first 20 rows in dataframe
    df = loader.get_tickets().to_pandas()[:20]

    # add column to dataframe to save predicted tag values
    df['predicted'] = pd.Series(np.empty(20, dtype=str))
    df['confidence'] = pd.Series(np.empty(20, dtype=np.float64))
    
    for index, row in df.iterrows():
        print(f"running inferece on ticket: {row['ticket_id']}, index: {index}")
        prediction = classify(body=row['body_text'])
        df.loc[index, 'predicted'] = prediction.tag
        df.loc[index, 'confidence'] = prediction.confidence

    print(df)

if __name__ == "__main__":
    main()
