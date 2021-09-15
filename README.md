# AdapTE

Adaptive pTE

This python code can be used in order to compute causality between time series. It is an improvement of the pTE that can be found here:
https://github.com/riccardosilini/pTE .

# What is AdapTE?

It has been created with the objective of having an agile and user-friendly Python function to compute the pTE (equivalent to Granger causality),
between time series to obtain the causality matrix. 

AdapTE can compute the pTE for a set of embeddings and time lags, 
returning a causality matrix where element (i, j) is the maximum pTE
from time series i to time series j across all different embeddings and time lags. 
This means that it is possible to see the influence of a given time series on another after a given time.
Or, to find the causality matrix containing the maximum value of causality, in the selected range of lags.

AdapTE can also partially compute the causality matrix. 
If the user is interested on the effect of a sub-ensemble of time series, 
or the influence of all time series on a sub-ensemble of them. 
It is possible to select the time series of interest and significantly speed up the computation by avoiding the all-to-all computation.
