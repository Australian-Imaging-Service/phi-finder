### ML-Anonymisation

### Building

```
python -m pip install --upgrade build

python -m build 

pip install dist/ml_anonymisation-2025.3.0-py3-none-any.whl
```


### (Very) Basic usage

```python
import pydicom as dicom
from dicom_tools import anonymise_dicom

ds = dicom.dcmread("/path/to/some/dicom.dcm")
ds = anonymise_dicom.anonymise_image(ds)

```