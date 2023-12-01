# This project implemented a supervised feature selection method using a density-based feature clustering procedure (SFSDFC).

This project is the extension of "EUFSFC: An efficient unsupervised feature selection procedure through feature clustering".

## Getting Start

We provided two versions of jupternotebooks for the implementation of the SFSDFC algorithm:

1. example_sfsdfc.ipynb for classification problems
2. example_sfsdfc.ipynb for regression problems

Users can directly run the two jupternotebooks depends on the problem.

The FC_v.py file implements the parallel processing for classification problem. The authors can modify it with any customized datasets by changing
* Line 17 `sample = pd.read_csv('Test.csv',header=None)`
* The header option can be set as True if the original csv file has headers

In case that the authors want to increase the number of selected features, the users can change the following line:
* Line 121, `NeiRad = 0.01*max(Dist)`: you can increase the value from 0.01 up to 0.7.
* Line 162, `NeiRad = 0.01*np.max(Dist)`: you can increase the value from 0.01 up to 0.7.

## Dependency
* numpy
* pandas
* scikit-learn

# Note:

More updates will be added gradually to wrap up the code into a python package soon.


## Citation format

For any use or modification of this project, please refer to the following article:

* X Yan, M Sarkar, B Gebru, S Nazmi, and A Homaifar. A supervised feature selection method for mixed-type data using density-based feature clustering. 2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC2021). 

For the continuation of this work, users can refer to the following paper:

* Yan, X., Homaifar, A., Sarkar, M., Lartey, B., & Gupta, K. D. (2022). An Online Unsupervised Streaming Features Selection Through Dynamic Feature Clustering. IEEE Transactions on Artificial Intelligence.
