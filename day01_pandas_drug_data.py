import pandas as pd
df=pd.DataFrame({
    "Compound":["CMP1","CMP2","CMP3","CMP4","CMP5"],
    "SMILES":["CCO","CCN","CCC",None,"CCCl"],
    "IC50":[12,45,8,None,20],
    "Toxicity":[2,8,1,3,5],
    "Target":["EGFR","EGFR","TP53","TP53","EGFR"]
})
# To identify the shape of the dataframe
print(df.shape,df.info(),df.head())
# To see which section includes numerical values
print(df.select_dtypes(include='number'))
# To see which section includes the text
print(df.select_dtypes(include='object'))
#fill missing values
df["IC50"] = df["IC50"].fillna(25)
#Creating a subset of Hit target molecule 
df["Hit"] = df["IC50"] < 20
hits=df[df["Hit"]]
print(df)
# Grouping all the Hit  target and obtaining their mean  
print(df.groupby("Target")["IC50"].mean())
#find Nitrogen- containing molecules
print(df["SMILES"].str.contains("N"))