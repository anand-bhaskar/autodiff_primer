# Automatic differentiation examples

[Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is a powerful tool that provides the advantages of numerical and symbolic differentiation. It can be used to compute _exact derivatives_ of a mathematical function implemented in a computer program, while relieving the user from having to implement the symbolic derivative. After installing the dependencies mentioned below, check out the examples in the IPython notebook.

## Dependencies
* IPython, NumPy, SciPy, and matplotlib  
  [IPython](https://ipython.org/install.html) provides an interactive notebook that lets us mix markdown, math, and code.
  The default installation of python on Mac OS X doesn't come with these libraries. It's easiest to download the [Anaconda](https://www.continuum.io/downloads) python distribution (we'll use python 2), which comes with IPython, NumPy, SciPy, and matplotlib and several other useful libraries. If you don't want to use IPython, you can use the python script version of the notebook.

* AD package  
   We'll use the [AD](https://pypi.python.org/pypi/ad/1.3.2) package for python 2, which implements forward-mode automatic differentiation. You can install it using the python package manager pip by running  
```pip install ad```  
   The [AD package documentation](http://pythonhosted.org/ad/user_guide.html) will be very useful.
