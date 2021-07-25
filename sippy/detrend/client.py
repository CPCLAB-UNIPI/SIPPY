import pandas as pd
import matplotlib.pyplot as plt
from detrending_filter import DetrendingFilter
import matplotlib.pyplot as plt

# load spteptest data from a CSV file
file = r"data\PC_Data_shifted.csv"
step_test_data = pd.read_csv(file, index_col="Time", parse_dates=True, skiprows=[1, 2])

filter = DetrendingFilter().get_filter("highpass")
filter.apply_filter(step_test_data, 20, 6)
trend = filter.filterdata.data["trend"]
filterddata = filter.filterdata.data["output"]
plt.plot(
    step_test_data.index,
    step_test_data["Temp"],
    step_test_data.index,
    trend["Temp"],
    step_test_data.index,
    filterddata["Temp"],
)
plt.show()
