

Opto the optimizer
=========

.. image:: https://github.com/robertocalandra/opto/blob/master/logo.png
     :width: 100px
     
This package offer a flexible framework to implement optimizers for Python.
Moreover, some standard algorithms are currently implemented::

* Bayesian optimization

To have more details about the features available on the package have a look at the examples.



============
Installation
============
To install the package clone the repository, and manually install the package using::

	git clone https://github.com/robertocalandra/opto.git 
	cd opto
	python setup.py install
	
Note: the DIRECT algorithm currently is just an interface to a FORTRAN implementation of DIRECT. For this reason it is necessary to install a FORTRAN compiler using::
 
    sudo apt-get install gfortran

============
Publications
============
The following publications make use of Opto::

- Yang, B.; Wang, G.; Calandra, R.; Contreras, D.; Levine, S. & Pister, K. Learning Flexible and Reusable Locomotion Primitives for a Microrobot IEEE Robotics and Automation Letters (RA-L), 2018
