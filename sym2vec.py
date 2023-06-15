import pickle
import pandas as pd
from tqdm import tqdm
def sym2vec(sym):
    symvec = pd.DataFrame(columns=symgt5.keys())
    for i in tqdm(range(len(sym))):
        symlist = sym[i].strip().split('、')
        for j in symgt5:
            if j in symlist:
                symvec.loc[i, j] = 1
            else:
                symvec.loc[i, j] = 0
    return symvec

if __name__ == '__main__':
    symgt5 = pickle.load(open('./output/symgt5.pkl', 'rb'))
    with open('./output/chatgpt_symptom.txt') as f:
        symptoms = f.readlines()
    sym5 = sym2vec(symptoms)
    sym5.to_csv('./output/sym5.csv', index=False)
