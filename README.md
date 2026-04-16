# PHI-finder

[![CI/CD](https://github.com/australian-imaging-service/phi-finder/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/australian-imaging-service/phi-finder/actions/workflows/ci-cd.yml)
[![Codecov](https://codecov.io/gh/australian-imaging-service/phi-finder/branch/main/graph/badge.svg?token=UIS0OGPST7)](https://codecov.io/gh/australian-imaging-service/phi-finder)


## Local testing (docker required)

```bash
conda create -n phi-finder python==3.11
conda activate phi-finder
pip install -e .[dev,test] --no-cache-dir
pytest .
```

## Building

```bash
python -m pip install --upgrade build

python -m build

pip install dist/phi_finder-0.1.11-py3-none-any.whl
```

## Basic usage

```python
import pydicom as dicom
from phi_finder.dicom_tools import anonymise_dicom

dcm = dicom.dcmread("/path/to/some/dicom.dcm")
anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
anonymised_dcm.save_as('/path/to/some/dicom_anon.dcm')


import pydicom as dicom
from phi_finder.dicom_tools import anonymise_dicom

dcm = dicom.dcmread("0012.DCM")
anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
anonymised_dcm.save_as('0012b.DCM')
```
