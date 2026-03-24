# Python Guide 

This is the repository relative to the _Python Workflow Guide_, which is available [here](https://hexagonal-giraffe-6f8.notion.site/Python-basics-in-WSL-1a4e8405938380ca9618e921065015aa?pvs=73).

## Clone and test
First clone the repository with
```bash
git clone git@github.com:alessandropedone/python-guide
```
This repository has one package (`src/pkg`) which is meant to perform polynomial regression on a artificial dataset, and select the best polynomial degree using 5-fold cross-validation.
To test the code, first recreate the environment with
```bash
mamba env create -f environment.yml
mamba activate guide
```
and then run `main.py` as a module in this way:
```bash
python -m src.pkg.main
```

## Repository structure
The repository has the basic structure a project should follow, let's summarize it with `tree`:
```bash
python-guide
├── .gitignore
├── README.md
├── docs
│   ├── Makefile
│   ├── _build
│   │   ├── doctrees
│   │   └── html
│   ├── _static
│   ├── _templates
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── pkg.rst
│   └── readme.rst
├── environment.yml
└── src
    └── pkg
        ├── __init__.py
        ├── main.py
        └── polyCV.py
```

`.gitignore` let's us avoid keeping track of the `__pycache__` folder, `README.md` is the Markdown file you are reading at the moment, the `docs` folder contains all the documentation generated using Sphinx, `environment.yml` allows us to rebuild the environment and the `src` folder contains the packages (in this case we find only `pkg`).

## Package 
Inside `src/pkg` you find two files, that are:
- `main.py`: it generates the synthetical data and performs the model selection and training task; but only when you run the file directly like we did at the begging, thank to the check 
    ```python
    if __name__ == "__main__":
        ...
    ```
- `polyCV.py`: it's the implementation of the class that handles the entire set of tasks we want to perform.

## Documentation
The documentation is obtained using Sphinx, and it's available at this [link](https://alessandropedone.github.io/python-guide/).
The important files are of two types:
1. `conf.py`, that contains the information about the compiler. 
2. `.rst` files, that contain the structure and content of our documentation.

I invite you to take a look at them to understand better the meaning of its content, since they are the backbone of the documentation. The content, instead, comes mainly from the comments in the code `main.py` and `polyCV.py`.