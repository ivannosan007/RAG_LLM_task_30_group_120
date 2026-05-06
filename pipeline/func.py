import pandas as pd
def foo():
    df = pd.read_pickle('C:/Users/Arkadiy/Desktop/python_files/Myfuturejob/MAGA_FKN/RAG_model_30/pipeline/ru_rag_test_dataset_renamed.pkl')
    print(df.keys())

foo()