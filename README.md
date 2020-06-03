# DeepRitz&DeepGalerkin

## Implementation of the Deep Ritz method and the Deep Galerkin method

Four problems are solved using the Deep Ritz method, see 2dpoisson.py, 2dpoisson-hole.py, 10dpoisson-cube.py, and 10dpoisson.py. Four problems are solved using least square functionals, see 2dpoisson-ls.py, 2dpoisson-hole-ls.py, 10dpoisson-cube-ls.py, and 10dpoisson-ls.py.

## Dependencies

* [NumPy](https://numpy.org)
* [PyTorch](https://pytorch.org/)
* [MATLAB](https://www.mathworks.com/products/matlab.html) (for post-processing only)

## References

W E, B Yu. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. <em>Communications in Mathematics and Statistics</em> 2018, 6:1-12. [[journal]](https://link.springer.com/article/10.1007/s40304-018-0127-z)[[arXiv]](https://arxiv.org/abs/1710.00211)  
J Sirignano, K Spiliopoulos. DGM: A deep learning algorithm for solving partial differential equations. <em>Journal of Computational Physics</em> 2018, 375:1339–1364. [[journal]](https://www.sciencedirect.com/science/article/pii/S0021999118305527)[[arXiv]](https://arxiv.org/abs/1708.07469)  
Y Liao, P Ming. Deep Nitsche method: Deep Ritz method with essential boundary conditions. 2019. [[arXiv]](https://arxiv.org/abs/1912.01309)