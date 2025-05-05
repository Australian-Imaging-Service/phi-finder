# ML-Anonymisation

[![CI/CD](https://github.com/australian-imaging-service/ml-anonymisation/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/australian-imaging-service/ml-anonymisation/actions/workflows/ci-cd.yml)
[![Codecov](https://codecov.io/gh/australian-imaging-service/ml-anonymisation/branch/main/graph/badge.svg?token=UIS0OGPST7)](https://codecov.io/gh/australian-imaging-service/ml-anonymisation)

## Building

```bash
python -m pip install --upgrade build

python -m build

pip install dist/ml_anonymisation-2025.5.0-py3-none-any.whl
```

## (Very) Basic usage

```python
import pydicom as dicom
from ml_anonymisation.dicom_tools import anonymise_dicom

dcm = dicom.dcmread("/path/to/some/dicom.dcm")
anonymised_dcm = anonymise_dicom.anonymise_image(dcm)

```
