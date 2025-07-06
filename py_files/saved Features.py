import pandas as pd
import numpy as np

def save_data(signal1, signal2, signal3, signal4, sheet_name):
    file_path = "D:/FCIS - ASU/Y4S2/Human Computer Interface/Project/Others/Data.xlsx"

    df1 = pd.DataFrame(np.array(signal1).T)
    df2 = pd.DataFrame(np.array(signal2).T)
    df3 = pd.DataFrame(np.array(signal3).T)
    df4 = pd.DataFrame(np.array(signal4).T)

    df_combined = pd.concat([df1, df2, df3, df4], axis=1)

    with pd.ExcelWriter(file_path, engine = 'openpyxl', mode = 'a', if_sheet_exists="overlay") as writer:
        df_combined.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=2)