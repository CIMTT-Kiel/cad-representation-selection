#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


results = pd.read_csv('/workspaces/clear-shape/reports/LSTM-full-training_2025-02-28/testset_outputs.csv', index_col=0)


# Volume
# filter out volume outliers
(
    so.Plot(results, x="volume", y="pred_volume")
    .add(so.Dots())
    .add(so.Line(), data={"x": [0, 1e7], "y": [0, 1e7]}, x="x", y="y")
    .label(title="Volume", x="True Volume", y="Predicted Volume")
).show()

(
    so.Plot(results)
    .add(so.Area(color="g"), so.Hist(), x="volume", label="True")
    .add(so.Area(color="b"), so.Hist(), x="pred_volume", label="Predicted")
    .scale(x="log")
    .label(title="Volume (Distribution)")
).show()

# Vertices
(
    so.Plot(results, x="vertices", y="pred_vertices")
    .add(so.Dots())
    .add(so.Line(), data={"x": [0, 500], "y": [0, 500]}, x="x", y="y")
    .label(title="Amount of Vertices", x="True Amount", y="Predicted Amount")
).show()

(
    so.Plot(results)
    .add(so.Area(color="g"), so.Hist(), x="vertices", label="True")
    .add(so.Area(color="b"), so.Hist(), x="pred_vertices", label="Predicted")
    .label(title="Amount of Vertices (Distribution)")
).show()

# Edges
(
    so.Plot(results, x="edges", y="pred_edges")
    .add(so.Dots())
    .add(so.Line(), data={"x": [0, 800], "y": [0, 800]}, x="x", y="y")
    .label(title="Amount of Edges", x="True Amount", y="Predicted Amount")
).show()

(
    so.Plot(results)
    .add(so.Area(color="g"), so.Hist(), x="edges", label="True")
    .add(so.Area(color="b"), so.Hist(), x="pred_edges", label="Predicted")
    .label(title="Amount of Edges (Distribution)")
).show()

# Faces
(
    so.Plot(results, x="faces", y="pred_faces")
    .add(so.Dots())
    .add(so.Line(), data={"x": [0, 250], "y": [0, 250]}, x="x", y="y")
    .label(title="Amount of Faces", x="True Amount", y="Predicted Amount")
).show()

(
    so.Plot(results)
    .add(so.Area(color="g"), so.Hist(), x="faces",label="True")
    .add(so.Area(color="b"), so.Hist(), x="pred_faces",label="Predicted")
    .label(title="Amount of Faces (Distribution)")
).show()

# %%
