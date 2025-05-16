import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

#Data Upload
df = pd.read_csv("diabetes_dataset.csv")

#Convert to 0 & 1
df_binary = df.copy()
for col in df.columns[:-1]:  # Outcome را نادیده می‌گیریم
    df_binary[col] = df_binary[col].apply(lambda x: 1 if x > df_binary[col].median() else 0)

#Apriori
frequent_itemsets_apriori = apriori(df_binary, min_support=0.3, use_colnames=True)

#FP-Growth
frequent_itemsets_fp = fpgrowth(df_binary, min_support=0.3, use_colnames=True)

#Apriori Rules
rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1.0)

#FP-Growth Rules
rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.0)

#Result
print("Apriori Rules: \n")
print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("FP-Growth Rules: \n")
print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])