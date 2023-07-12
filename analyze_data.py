import pandas as pd
import sweetviz as sv

df = pd.read_csv('codorniu.csv')
report = sv.analyze(df)
report.show_html('add.html')