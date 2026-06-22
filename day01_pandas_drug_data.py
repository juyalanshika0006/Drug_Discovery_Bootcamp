import pandas as pd
df=pd.DataFrame({
    "Compound":["CMP1","CMP2","CMP3","CMP4","CMP5"],
    "SMILES":["CCO","CCN","CCC",None,"CCCl"],
    "IC50":[12,45,8,None,20],
    "Toxicity":[2,8,1,3,5],
    "Target":["EGFR","EGFR","TP53","TP53","EGFR"]
})
#print(df.shape,df.info(),df.head())
#print(df.select_dtypes(include='number'))
#print(df.select_dtypes(include='object'))
#df["IC50"] = df["IC50"].fillna(25)#fill missing values
#df["Hit"] = df["IC50"] < 20
#hits=df[df["Hit"]]
#print(df)
print(df.groupby("Target")["IC50"].mean())
print(df["SMILES"].str.contains("N"))