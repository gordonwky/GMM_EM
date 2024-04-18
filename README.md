# GMM-EM 

Developer Names: Kim Ying WONG

Date of project start: 9 Jan 2024

This project is a library for Gaussian Mixture Model (GMM-EM). Gaussian Mixture model is a probabilistic model that describes the datasets or measurement in a linear combination of some basic distributions. Gaussian mixture model falls into a subset of the mixture model, where Gaussian distribution is used as a basis. In general, almost all continuous density could be approximated as sufficient number of Gaussian mixtures with appropriate mean and covariance. 
Therefore, it can be used in various cases in machine learning, such as clustering and density estimation. The GMM-EM aims at implementation of the Gaussian mixture model for clustering with the Expectation-Maximization algorithm (EM Algorithm). 

The folders and files for this project are as follows:<br />

docs - Documentation for the project <br />

refs - Reference material used for the project, including papers <br />

src - Source code (cpp file) <br />

include - Source code (header file) <br />

test - Test cases <br />

dataset - Test dataset <br />

example - running example code and visualization for result <br />

executable - executable file for example and testing <br />

## Compile

Create a build folder and follows command <br />

cd build , cmake .. , make <br />
or auto build by VS code and follow with make