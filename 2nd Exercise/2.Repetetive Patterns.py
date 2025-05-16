import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth, association_rules

#Data Upload
df = pd.read_csv("diabetes_dataset.csv")

#Convert to binary
df_binary = df.copy()
for col in df.columns[:-1]:  # Outcome is not important
    df_binary[col] = df_binary[col].apply(lambda x: 1 if x > df_binary[col].median() else 0)

#Apriori
frequent_itemsets_ap = apriori(df_binary, min_support=0.3, use_colnames=True)

#Rules
rules_ap = association_rules(frequent_itemsets_ap, metric="lift", min_threshold=1.0)

#Results
print("AP Repetetive Results: ")
print(frequent_itemsets_ap)

print("AP Dependency rules: \n")
print(rules_ap)

#Data Upload
df = pd.read_csv("diabetes_dataset.csv")

#Convert to binary
df_binary = df.copy()
for col in df.columns[:-1]:  # Outcome is not important
    df_binary[col] = df_binary[col].apply(lambda x: 1 if x > df_binary[col].median() else 0)

#FP-Growth
frequent_itemsets_fp = fpgrowth(df_binary, min_support=0.3, use_colnames=True)

#Rules
rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.0)

#Results
print("FP Repetetive Results: ")
print(frequent_itemsets_fp)

print("FP Dependency rules: \n")
print(rules_fp)