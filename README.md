# Contributions

Contributions are welcome!


# Requirements
* numpy
* h5py
* scipy
* theano
* biopython
* keras

tensorflow may be additionally required if using Python 2 instead of Python 3.

On Windows, Visual Studio C++ is required.
[Microsoft Visual C++ 14.0](http://landinghub.visualstudio.com/visual-cpp-build-tools)
is required if using Python 3.5 or
[Microsoft Visual C++ 10.0](www.microsoft.com/download/details.aspx?id=8279)
is required is Python 3.4.


# Installation
## Using Anaconda

```
conda create -n phenotypes python=3 numpy h5py scipy theano biopython
source activate phenotypes
pip install keras
```

## Using pip
`pip install -r requirements.txt`

Depending on your environment you may have to pip install numpy before
pip installing other packages. This will be improved if/when a
setup script is created.


# References
Sønderby, S. K., Sønderby, C. K., Nielsen, H. & Winther, O. Convolutional LSTM networks for subcellular localization of proteins. in Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 9199, 68–80 (2015).
<br>https://arxiv.org/abs/1503.01919
