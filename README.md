# my_awesome_component

[![codecov](https://codecov.io/gh/ICRAR/dlg-nifty-components/branch/main/graph/badge.svg?token=dlg-nifty-components_token_here)](https://codecov.io/gh/ICRAR/dlg-nifty-components)
[![CI](https://github.com/ICRAR/dlg-nifty-components/actions/workflows/main.yml/badge.svg)](https://github.com/ICRAR/dlg-nifty-components/actions/workflows/main.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Awesome dlg_nifty_components created by ICRAR

## Installation

There are multiple options for the installation, depending on how you are intending to run the DALiuGE engine, directly in a virtual environment (host) or inside a docker container. You can also install it either from PyPI (latest released version).

## Install it from PyPI

### Engine in virtual environment
```bash
pip install dlg_nifty_components
```
### Engine in Docker container
```bash
docker exec -t daliuge-engine bash -c 'pip install --prefix=$DLG_ROOT/code dlg_nifty_components'
```
## Usage
For example the MyComponent component will be available to the engine when you specify 
```
dlg_nifty_components.appComponents.MyComponent
```
in the AppClass field of a Python Branch component. The EAGLE palette associated with these components are also generated and can be loaded directly into EAGLE. In that case all the fields are correctly populated for the respective components.

