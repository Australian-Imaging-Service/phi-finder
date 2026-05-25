import os
import json
import logging
from datetime import datetime
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

import torch
import torch.ao.quantization as taq
import pydicom as dicom
from pydicom.datadict import add_private_dict_entries
from gliner import GLiNER
from gliner.model import UniEncoderSpanGLiNER

from presidio_image_redactor import DicomImageRedactorEngine
from presidio_anonymizer import AnonymizerEngine
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.valuerep import PersonName
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider


def destroy_pixels(ds: dicom.dataset.FileDataset) -> dicom.dataset.FileDataset:
    """It sets all pixel values to 0.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        The DICOM dataset containing the image data to be destroyed.

    Returns
    -------
    pydicom.dataset.FileDataset
        The modified DICOM dataset with pixel data destroyed.
    """
    if "PixelData" in ds:
        pixel_array = ds.pixel_array
        pixel_array[:] = 0
        data_downsampling = pixel_array[:8, :8]
        ds.PixelData = data_downsampling.tobytes()
        ds.Rows, ds.Columns = data_downsampling.shape
    return ds


def _build_presidio_analyser(score_threshold: float=0.5,
                             spacy_model_name: str="en_core_web_md") -> AnalyzerEngine:
    """Builds and configures a Presidio analyser engine for named entity recognition.

    This function initialises an NLP engine using the SpaCy library and sets up
    various pattern recognisers for different types of entities, including titles,
    correspondence, phone numbers, medical record numbers (MRN), provider numbers,
    dates, street addresses, postcodes, suburbs, states, and institutes. The
    recognisers are configured with specific patterns and deny lists.

    Parameters
    ----------
    score_threshold : float, optional
        The score threshold for entity recognition. Entities with a score below this
        threshold will not be considered for anonymisation. Default is 0.5.
    spacy_model_name : str, optional
        The name of the SpaCy model to use for NLP processing. Default is "en_core_web_md".
        Other options include "en_core_web_sm" and "en_core_web_lg".
        
    Returns
    -------
    AnalyzerEngine
        An instance of the AnalyzerEngine configured with various recognisers for
        named entity recognition.
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model_name}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    title_recognizer = PatternRecognizer(
        supported_entity="TITLE",
        deny_list=[
            "Dr",
            "DR",
            "Prof",
            "PROF",
            "Prof.",
            "Doctor",
            "DOCTOR",
            "Professor",
            "PROFESSOR",
            "Associate Professor",
            "ASSOCIATE PROF",
            "ASSOCIATE PROFESSOR",
            "A/Prof",
            "A/Prof.",
            "A / Prof",
            "A / Professor",
            "A / PROF",
            "Radiation Oncologist",
        ],
    )
    correspondence_recognizer = PatternRecognizer(
        supported_entity="CORRESPONDENCE",
        patterns=[
            Pattern(name="correspondence", regex=r"Dear(\s+)(\w+)(\s+)(\w+)", score=score_threshold)
        ],
    )  # A lower score increases likelihood of capturing the entity but decreases the confidence
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE",
        patterns=[
            Pattern(
                name="phone",
                regex=r"(\(+61\)|\+61|\(0[1-9]\)|0[1-9])?( ?-?[0-9]){8,14}",  # 8 to 14 digits
                score=score_threshold,
            )
        ],
    )
    mrn_recognizer = PatternRecognizer(
        supported_entity="MRN",
        patterns=[
            Pattern(
                name="mrn",
                regex=r"\d{5,9}",  # for numbers between 5-9 digits long
                score=score_threshold,
            )
        ],
    )
    gender_recognizer = PatternRecognizer(
        supported_entity="GENDER",
        patterns=[
            Pattern(
                name="gender",
                regex=r"(?i)(^[fm]$)|(^(male|female)$)",  # Sole string 'M' or 'F'
                score=score_threshold,
            )
        ],
    )
    providernumber_recognizer = PatternRecognizer(
        supported_entity="PROVIDER_NUMBER",
        patterns=[
            Pattern(
                name="provider number",
                regex=r"(\d+(Y))|(\d+(X)|(?:(Provider Number:)+\s+(\d+|\w+)))",
                score=score_threshold,
            )
        ],
    )
    date_recognizer = PatternRecognizer(
        supported_entity="DATE",
        patterns=[
            Pattern(
                name="date",
                regex=r"([0-9]{1,2}(\/|-|.)[0-9]{1,2}(\/|-|.)[0-9]{2,4})|(\b\d{1,2}\D{0,3})?\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)\D?(\d{1,2}\D?)?\D?((19[7-9]\d|20\d{2})|\d{2})",
                score=score_threshold,
            )
        ],
    )
    street_recognizer = PatternRecognizer(
        supported_entity="STREET",
        patterns=[
            Pattern(
                name="street",
                regex=r"((\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE))|(\d+\s+\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce))|(\d+\s+\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE))|(\D+\S+\W+\S(?:ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE)(\s+\w+\s)(?:New South Wales|Victoria|Queensland|Western Australia|South Australia|Tasmania|Australian Capital Territory|Northern Territory|NEW SOUTH WALES|VICTORIA|QUEENSLAND|WESTERN AUSTRALIA|SOUTH AUSTRALIA|TASMANIA|AUSTRALIAN CAPITAL TERRITORY|NORTHERN TERRITORY|NSW|VIC|QLD|WA|SA|TAS|ACT|NT)(\s+\d{4})))",
                score=score_threshold,
            )
        ],
    )
    postcode_recognizer = PatternRecognizer(
        supported_entity="POSTCODE",
        patterns=[
            Pattern(
                name="postcode",
                regex=r"\d{4}",  # for numbers between 4 digits long
                score=score_threshold,
            )
        ],
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Suburbs list from https://github.com/damiankotevski/anonymisation
    suburbs_australia_path = os.path.join(script_dir, "suburbs_australia.txt")
    with open(suburbs_australia_path, "r", encoding='utf8') as f:
        deny_list = f.readlines()
    deny_list = [x.strip() for x in deny_list]
    suburb_recognizer = PatternRecognizer(
        supported_entity="SUBURB",
        deny_list=deny_list,
    )

    state_recognizer = PatternRecognizer(
        supported_entity="STATE",
        deny_list=[
            "NSW",
            "New South Wales",
            "NEW SOUTH WALES",
            "QLD",
            "Queensland",
            "QUEENSLAND",
            "NT",
            "Northern Territory",
            "NORTHERN TERRITORY",
            "WA",
            "Western Australia",
            "WESTERN AUSTRALIA",
            "SA",
            "South Australia",
            "SOUTH AUSTRALIA",
            "VIC",
            "Victoria",
            "VICTORIA",
            "TAS",
            "Tasmania",
            "TASMANIA",
            "ACT",
            "Australian Capital Territory",
            "AUSTRALIAN CAPITAL TERRITORY",
            "Australia",
            "AUSTRALIA",
        ],
    )

    institute_recognizer = PatternRecognizer(
        supported_entity="INSTITUTE",
        patterns=[
            Pattern(
                name="institute",
                regex=r"(\w+\s(Medical Centre|Cancer Centre|Medical Practice))",
                score=score_threshold,
            )
        ],
        deny_list=[
            "Prince of Wales Hospital",
            "Prince of Wales",
            "Prince of Wales Private",
            "POW Private",
            "POWPH",
            "POWH",
            "Nelune Comprehensive Cancer Centre",
            "Bright Building",
            "Liverpool Hospital",
            "Liverpool",
            "Campbelltown Hospital",
            "Campbelltown",
            "Wollongong Hospital",
            "Wollongong",
            "Shoalhaven District Memorial Hospital",
            "Shoalhaven District Memorial",
            "Shoalhaven",
            "St George Hospital",
            "St George",
            "SGH",
            "Royal North Shore Hospital",
            "Royal North Shore",
            "RNSH",
            "Tamworth Hospital",
            "Tamworth",
            "TBH",
            "Calvary",
            "Calvary Mater",
            "Calvary Mater Newcastle",
            "Calvary Mater Newcastle Hospital",
            "Newcastle",
            "CMMN",
            "St Vincents Hospital",
            "St Vincents",
            "GenesisCare",
            "SVH",
            "Macquarie Univerisity",
            "Macquarie University Hospital",
            "Waratah Private Hospital",
            "Hurstville",
            "Mater Sydney",
            "Mater Hospital",
            "Albury Wodonga",
            "Albury",
        ],
    )

    analyzer.registry.add_recognizer(title_recognizer)
    analyzer.registry.add_recognizer(correspondence_recognizer)
    analyzer.registry.add_recognizer(phone_recognizer)
    analyzer.registry.add_recognizer(mrn_recognizer)
    analyzer.registry.add_recognizer(providernumber_recognizer)
    analyzer.registry.add_recognizer(gender_recognizer)
    analyzer.registry.add_recognizer(date_recognizer)
    analyzer.registry.add_recognizer(street_recognizer)
    analyzer.registry.add_recognizer(postcode_recognizer)
    analyzer.registry.add_recognizer(suburb_recognizer)
    analyzer.registry.add_recognizer(state_recognizer)
    analyzer.registry.add_recognizer(institute_recognizer)
    return analyzer


def _build_transformer() -> UniEncoderSpanGLiNER:
    model = GLiNER.from_pretrained("nvidia/gliner-pii", max_length=384)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
        model.compile()
        torch.set_float32_matmul_precision('high')
    else:
        model = taq.quantize_dynamic(
            model,
            {torch.nn.Linear},  # quantize all Linear layers to INT8
            dtype=torch.qint8,
        )
    return model


def _anonymise_with_transformer(model: UniEncoderSpanGLiNER, text: str, threshold: float=0.01) -> str:
    """Anonymises text using a specified named entity recognition (NER) pipeline.

    This function processes the input text through the provided NER pipeline,
    replacing recognised entities of type "PER", "LOC", and "ORG" with the placeholder "[XXXX]".

    Parameters
    ----------
    model : Gliner's UniEncoderSpanGLiNER
        The NER pipeline to use for entity recognition.
    
    text : str
        The input text to be anonymised.

    Returns
    -------
    str
        The anonymised text with specified entities replaced by "[XXXX]".
    """
    LABELS = [
        "age", "profession", "gender",
        "sex", "language", "ethnicity",
        "country", "city", "state", "suburb",
        "location", "person", "organization",
        "phone number", "address", "passport number",
        "email", "social security number", "health insurance id number",
        "date of birth", "mobile phone number",
        "health insurance number",
    ]
    # merged collapses overlapping entity spans into non-overlapping 
    # ones so the slice-replacement at the end doesn't 
    # double-redact or produce corrupted offsets.
    # e.g. "Dr John Smith" might be person (0–13) and profession (0–2).
    try:
        with torch.inference_mode():
            pred_entities = model.predict_entities(text, LABELS, threshold=threshold)
        spans = sorted((e['start'], e['end']) for e in pred_entities)
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in reversed(merged):
            text = text[:start] + 'XXXX' + text[end:]
    except Exception as e:
        print(f"Error occurred while anonymising text: {e}")
    return text


def _anonymise_ds(ds: dicom.dataset.Dataset,
                  analyser: AnalyzerEngine,
                  anonymizer: AnonymizerEngine,
                  score_threshold: float,
                  use_transformers: bool,
                  gliner_pii=None,
                  use_case: str='Standard',
                  anonymised_headers: list | None = None) -> None:
    """Recursively anonymises all elements in a DICOM dataset in-place."""
    if anonymised_headers is None:
        anonymised_headers = []
    for elem in ds:
        if elem.VR == "SQ":
            for sub_ds in elem.value:
                if not isinstance(sub_ds, dicom.dataset.Dataset):
                    continue
                _anonymise_ds(
                    sub_ds, analyser, anonymizer, score_threshold,
                    use_transformers, gliner_pii, use_case,
                    anonymised_headers
                )
        elif elem.VR == "PN" or elem.tag == (0x0010, 0x0010):
            ds[elem.tag].value = PersonName("XXXX")
            anonymised_headers.append({"tag": str(elem.tag), "name": elem.name})
        elif elem.tag == (0x0010, 0x0040):  # Sex unchanged.
            continue
        elif elem.tag == (0x0010, 0x0030):  # Birthdate
            birthdate_str = str(elem.value).strip()
            if birthdate_str == "":
                continue
            year = None
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y"):
                try:
                    year = datetime.strptime(birthdate_str, fmt).year
                    break
                except ValueError:
                    continue
            # Fail-safe: if the format is unrecognised, scrub the value so the
            # original birthdate never survives in the dataset.
            ds[elem.tag].value = f"{year:04d}0101" if year is not None else "19000101"
            anonymised_headers.append({"tag": str(elem.tag), "name": elem.name})
        elif elem.VR in [
            "LO",  # Long String
            "LT",  # Long Text
            #"OW",  # Other Word
            "SH",  # Short String
            "ST",  # Short Text
            "UC",  # Unlimited Characters
            "UT",  # Unlimited Text
            #"DA",  # Date
            "CS",  # Code String
            #"AS",  # Age String
        ]:  # https://dicom.nema.org/medical/dicom/current/output/html/part05.html#table_6.2-1 and https://pydicom.github.io/pydicom/stable/guides/element_value_types.html
            try:
                original = elem.value
                if original is None:
                    continue
                is_multi = isinstance(original, dicom.multival.MultiValue)
                if is_multi and len(original) == 0:
                    continue
                if not is_multi and original == "":
                    continue
                values = [str(v) for v in original] if is_multi else [str(original)]
                new_values = []
                for v in values:
                    analyzer_results = analyser.analyze(text=v, language="en", score_threshold=score_threshold)
                    redacted = anonymizer.anonymize(
                        text=v,
                        analyzer_results=analyzer_results,
                        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "XXXX"})},
                    ).text
                    if use_transformers and len(redacted) > 30:
                        redacted = _anonymise_with_transformer(gliner_pii, redacted, threshold=score_threshold)
                    new_values.append(redacted)
                if new_values != values:
                    anonymised_headers.append({"tag": str(elem.tag), "name": elem.name})
                if is_multi:
                    ds[elem.tag].value = dicom.multival.MultiValue(str, new_values)
                else:
                    ds[elem.tag].value = new_values[0]
            except Exception as e:
                print(f"Skipped {elem.tag} : {type(e).__name__}: {e}")


def anonymise_image(ds: dicom.dataset.FileDataset,
                    analyser: AnalyzerEngine=None,
                    anonymizer: AnonymizerEngine=None,
                    image_redactor: DicomImageRedactorEngine = None,
                    score_threshold: float=0.5,
                    use_transformers: bool=False,
                    gliner_pii: UniEncoderSpanGLiNER=None,
                    use_case: str='Standard') -> dicom.dataset.FileDataset:
    """Anonymises a DICOM image by redacting personal information.

    This function processes the DICOM dataset, redacting personal names and other
    identifiable information based on the specified score threshold. It utilises
    named entity recognition pipelines to identify and replace sensitive information.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        The DICOM dataset containing the image data and metadata to be anonymised.
    
    analyser : AnalyzerEngine, optional
        Presidio analyser engine. Built automatically if not provided.

    anonymizer : AnonymizerEngine, optional
        Presidio anonymizer engine. Built automatically if not provided.

    image_redactor : DicomImageRedactorEngine, optional
        It redacts burned-in PHI from the pixel data.

    score_threshold : float, optional
        The score threshold for entity recognition. Entities with a score below this
        threshold will not be considered for anonymisation. Default is 0.5.

    use_transformers : bool, optional (default False)
        If True, transformers will be used for anonymisation on top of Presidio's output.
    
    gliner_pii: UniEncoderSpanGLiNER, optional (default False)
        If set, the model will be used for anonymisation on top of Presidio's output.

    use_case : str, optional (default 'Standard')
        Standard: some fields are not redacted, only flagged.
        Aggressive: all PII fields are redacted.

    Returns
    -------
    pydicom.dataset.FileDataset
        The anonymised DICOM.
    """
    new_dict_items = {
        0x02091000: ('ST', '1', 'Flagged Headers PHI-Finder')  # Private tag to store the list of anonymised headers,
    }
    add_private_dict_entries(private_creator="phi-finder", new_entries_dict=new_dict_items)

    if image_redactor is not None:
        ds = image_redactor.redact(ds, fill="contrast")  # fill="background")
    if analyser is None:
        analyser = _build_presidio_analyser(score_threshold)
    if anonymizer is None:
        anonymizer = AnonymizerEngine()
    # operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})}

    anonymised_headers = []
    _anonymise_ds(ds, analyser, anonymizer, score_threshold,
                  use_transformers, gliner_pii, use_case, anonymised_headers)
    '''
    Adding a private header with the flagged headers list.
    private_block() reserves a slot (e.g., 0x10) and writes the creator name at (0x0209, 0x0010).
    The actual data then lives at (0x0209, 0x10XX).
    Then, ds.add_new([0x0209, 0x0010], ...) overwrites the Private Creator element itself.
    '''
    flagged_headers = json.dumps(anonymised_headers)
    block = ds.private_block(0x0209, "phi-finder", create=True)
    block.add_new(0x00, 'ST', flagged_headers)  # 0x00 offset within block → maps to (0x0209, 0x1000)
    return ds
