# Editing the Notebook

## Clone the Project

```bash
git clone git@github.com:luisca1985/r_squared_article.git
```

## Virtual Enverironment (optional)

To [Create a Python Virtual Environment] run:

```bash
python -m venv .venv
```

To activate run:

```bash
source .venv/bin/activate
```

To deactivate run:

```bash
deactivate
```

## Install packages

To run the notebook first you should [install the packages using the requirements.txt file][install packages using a requirements file]:

```bash
pip install -r requirements.txt
```

## Convert Notebook to README.md

To [convert the notebook to a README.md file][Convert the Jupyter notebook to markdown] run:

```bash
jupyter nbconvert --execute --to markdown edit_notebook/r_squared.ipynb --output-dir . --output README.md
```

[Create a Python Virtual Environment]: https://docs.python.org/3/library/venv.html#creating-virtual-environments

[Install packages using a requirements file]: https://packaging.python.org/en/latest/tutorials/installing-packages/#requirements-files

[Convert the Jupyter notebook to markdown]: https://nbconvert.readthedocs.io/en/latest/usage.html#markdown