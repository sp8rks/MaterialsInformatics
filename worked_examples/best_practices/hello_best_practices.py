import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.DataFrame({
    "a": np.random.randn(100),
    "b": np.random.choice(["cat", "dog", "turtle"], size=100),
})

report = ProfileReport(df, title="Best Practices 2026")
report.to_file("profile.html")
print("Wrote profile.html")
