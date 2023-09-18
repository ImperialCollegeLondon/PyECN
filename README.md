# PyECN

PyECN is a python-based equivalent circuit network (ECN) framework for modelling lithium-ion batteries.


## Using PyECN

PyECN is run by providing `pyecn` with a configuration file for a simulation, containing details of a cell's geometrical, physical, electrical and thermal properties, as well as operating conditions. An example configuration file for a pouch cell is provided in `pouch.toml`:

```bash
$ python -m pyecn pouch.toml
```

PyECN can also be run in an interactive python session:

```bash
$ python
>>> import pyecn
>>> pyecn.run()
Enter config file name:
pouch.toml
```


## Installing PyECN

<details>
  <summary>Linux/macOS</summary>

  1. Clone the repository and enter the directory:
  ```bash
  git clone https://github.com/ImperialCollegeLondon/PyECN.git
  cd PyECN
  ```

  2. Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

  3. Install the dependencies:
  ```bash
  pip install -U pip
  pip install -r requirements.txt
  ```
</details>

<details>
  <summary>Windows</summary>

  1. Clone the repository and enter the directory:
  ```bat
  git clone https://github.com/ImperialCollegeLondon/PyECN.git
  cd PyECN
  ```

  2. Create and activate a virtual environment:
  ```bat
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

  3. Install the dependencies:
  ```bat
  pip install -U pip
  pip install -r requirements.txt
  ```
</details>

## Citing PyECN

If you use PyECN in your work, please cite our paper

> Li, S., Rawat, S. K., Zhu, T., Offer, G. J., & Marinescu, M. (2023). Python-based Equivalent Circuit Network (PyECN) Model-ling Framework for Lithium-ion Batteries: Next generation open-source battery modelling framework for Lithium-ion batteries. _Engineering Archive_.

You can use the BibTeX

```
@article{lipython,
  title={Python-based Equivalent Circuit Network (PyECN) Model-ling Framework for Lithium-ion Batteries: Next generation open-source battery modelling framework for Lithium-ion batteries},
  author={Li, Shen and Rawat, Sunil Kumar and Zhu, Tao and Offer, Gregory J and Marinescu, Monica},
  publisher={Engineering Archive}
}
```

## Contributing to PyECN

TBC


## License

PyECN is fully open source. For more information about its license, see [LICENSE](https://github.com/ImperialCollegeLondon/PyECN/blob/add_license/LICENSE.md).


## Contributors

- Shen Li: Conceptualisation, methodology, creator and lead developer of PyECN, writing and review;
- Sunil Rawat: Contributor of PyECN, discussion, writing and review;
- Tao Zhu: Contributor of PyECN, discussion, writing and review;
- Gregory J Offer: Conceptualisation, funding acquisition, supervision, writing – review & editing;
- Monica Marinescu: Conceptualisation, funding acquisition, supervision, writing – review & editing;
