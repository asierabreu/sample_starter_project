# Sample Documentation

This is sample documentation generated with the project. We use Sphinx for automatic document generation.

### Project Structure

The project code and documentation are kept in a GitHub repository with the following structure:
  - code : contains project related Jupyter notebooks or Python code
  - docs : contains project related documentation 
  - figs : project related figures generated as part of the data analysis or modelling process
  - requirements.txt specifies the library dependencies in this project

### Installation Instructions
```{note}
The following assumes you have a local Python or Anaconda installation
```

 1. Checkout the github repository onto a local machine :

 ```console
(base)foo@machine:~$ git checkout https:github.com/path_to_repository.git
```

 2. Create a working environment and install the required dependencies on it :
 
 ```console
(base)foo@machine:~$ python -m venv my_new_localenv 
(base)foo@machine:~$ source my_new_localenv/bin/activate 
(my_new_localenv)foo@machine:~$ pip install -r requirements.txt 
```

 ### Basic Usage

 1. Start-up your preferred environment for Jupyter Notebook execution and load the provided notebooks
    For example:   
 ```console
(my_new_localenv)foo@machine:~$ jupyterlab 
```

