import pandas as pd

material = "MoS2"
df = pd.read_csv(f"./{material}/mass/Mass_TNN_{material}_GGA.dat", sep=",")

print(df)
df["m_rK1"] = round(df["m_hK1"] * df["m_eK1"] / (df["m_hK1"] + df["m_eK1"]), 4)
df["m_rK2"] = round(df["m_hK2"] * df["m_eK2"] / (df["m_hK2"] + df["m_eK2"]), 4)
print(df.head(25))
