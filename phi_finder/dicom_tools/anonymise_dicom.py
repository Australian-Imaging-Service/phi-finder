import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import torch
import pydicom as dicom
from pydicom.datadict import add_private_dict_entries
from pydicom.tag import Tag
from pydicom.valuerep import PersonName
from gliner import GLiNER
from gliner.model import UniEncoderSpanGLiNER
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_image_redactor import DicomImageRedactorEngine

from phi_finder.dicom_tools import ps3_15

logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Provenance stamped on each flagged-header record, so the de-identification
# report can separate headers whose value was read and scrubbed by the NER
# models from those the PS3.15 action map handled (see ps3_15.SOURCE_PS3_15).
SOURCE_NER = "ner"


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
        # Build the replacement pixels from scratch rather than decoding the
        # originals, so compressed files work without any decode handlers.
        bits = int(ds.get("BitsAllocated", 16) or 16)
        bits = 8 if bits <= 8 else 16 if bits <= 16 else 32
        signed = int(ds.get("PixelRepresentation", 0) or 0) == 1
        zeros = np.zeros((8, 8), dtype=f"{'int' if signed else 'uint'}{bits}")
        ds.PixelData = zeros.tobytes()
        ds.Rows, ds.Columns = zeros.shape
        ds.BitsAllocated = bits
        ds.BitsStored = bits
        ds.HighBit = bits - 1
        ds.PixelRepresentation = 1 if signed else 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        for keyword in ("NumberOfFrames", "PlanarConfiguration"):
            if keyword in ds:
                del ds[keyword]
        if getattr(ds, "file_meta", None) is None:
            ds.file_meta = dicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = dicom.uid.ExplicitVRLittleEndian
        ds.is_implicit_VR = False
        ds.is_little_endian = True
    return ds


def _build_presidio_analyser(score_threshold: float=0.5,
                             spacy_model_name: str="en_core_web_lg") -> AnalyzerEngine:
    """Builds and configures a Presidio analyser engine for named entity recognition.

    Parameters
    ----------
    score_threshold : float, optional
        The score threshold for entity recognition. Entities with a score below this
        threshold will not be considered for anonymisation. Default is 0.5.
    spacy_model_name : str, optional
        The name of the SpaCy model to use for NLP processing. Default is "en_core_web_lg".
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
                regex=r"(\(\+61\)|\+61|\(0[1-9]\)|0[1-9])?( ?-?[0-9]){8,14}",  # 8 to 14 digits
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

    age_recognizer = PatternRecognizer(
        supported_entity="AGE",
        patterns=[
            # DICOM Age String (AS): 057Y / 057 Y / 057D / 012M / 006W — all units, optional OCR space
            Pattern(name="dicom_age", regex=r"\b\d{1,3}\s*[DWMY]\b", score=score_threshold),
            # Labelled: "Age: 57", "Age 057Y", "AGE=89"
            Pattern(name="labelled_age", regex=r"(?i)\bage\b\s*[:=]?\s*\d{1,3}\s*[dwmy]?\b", score=score_threshold),
            # Suffixed: "57 yo", "57y/o", "57 yrs old"
            Pattern(name="age_suffix", regex=r"(?i)\b\d{1,3}\s*(?:yo|y/?o|yrs?|years?\s*old)\b", score=score_threshold),
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
    analyzer.registry.add_recognizer(age_recognizer)
    return analyzer


def _build_transformer() -> UniEncoderSpanGLiNER:
    model = GLiNER.from_pretrained("nvidia/gliner-pii")#, max_length=384)
    device = torch.device('cpu')#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    #if torch.cuda.is_available():
    #    model.compile()
    #    torch.set_float32_matmul_precision('high')
    return model


def _anonymise_with_transformer(model: UniEncoderSpanGLiNER,
                                text: str,
                                threshold: float=0.15,
                                return_entities: bool=False) -> str:
    """Anonymises text using a specified named entity recognition (NER) pipeline.

    Parameters
    ----------
    model : Gliner's UniEncoderSpanGLiNER
        The NER pipeline to use for entity recognition.
    
    text : str
        The input text to be anonymised.

    threhsold: float, optional (default=0.15)
        Confidence needed to flag an entity.

    return_entities: bool, optional (default=False)
        Whether to return a tuple with the entity types.

    Returns
    -------
    str
        The anonymised text with specified entities replaced by "[XXXX]".
    """
    LABELS = [
        "age", "profession", "gender", "name",
        "sex", "language", "ethnicity",
        "country", "city", "state", "suburb",
        "location", "person", "organization"
    ]
    # merged collapses overlapping entity spans into non-overlapping
    # ones so the slice-replacement at the end doesn't
    # double-redact or produce corrupted offsets.
    # e.g. "Dr John Smith" might be person (0–13) and profession (0–2).
    labels_pred: list[str] = []
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
        labels_pred = sorted(e['label'] for e in pred_entities)
    except Exception as e:
        # Fail closed: if recognition errors out we cannot know what is PHI,
        # so the whole text is redacted rather than passed through.
        logger.error("Error while anonymising text, redacting it entirely: %s", e)
        text = "XXXX"
    if return_entities:
        return text, labels_pred
    return text


# Structural elements whose values are DICOM defined terms, not free text.
# They must never be redacted: e.g. ImageType's magnitude component 'M' would
# otherwise match the standalone-M/F gender pattern, corrupting the image, and
# the postcode pattern would hit the "2022" in charset "ISO 2022 IR 100",
# breaking text decoding of the whole file.
_STRUCTURAL_TAGS = frozenset({
    Tag(0x0008, 0x0005),  # Specific Character Set
    Tag(0x0008, 0x0008),  # Image Type
    Tag(0x0008, 0x0060),  # Modality
    Tag(0x0018, 0x5100),  # Patient Position (e.g. HFS)
    Tag(0x0020, 0x0020),  # Patient Orientation (components like 'F' for foot)
    Tag(0x0028, 0x0004),  # Photometric Interpretation
})

# Standard headers that PS3.15 Table E.1-1 gives no action to,
# but they can hold long texts, so the NER models scan them.
_NER_SCANNED_TAGS = frozenset({
    Tag(0x0040, 0xA160),  # Text Value (SR Document Content)
})

# Binary VRs whose bytes routinely hold text when the element is private, so
# private elements carrying them are scanned rather than skipped:
#   UN  what a private attribute of an Implicit VR Little Endian file is read
#       back as whenever its creator is not one pydicom's private dictionary
#       covers -- the file states no VR and there is nothing to look it up in,
#       so the whole block would otherwise go unscanned.
#   OB  vendor blocks such as the Siemens CSA header, which mix framing bytes
#       with plain strings.
# Standard attributes are deliberately not included: OB/UN there is bulk data
# (pixels, overlays, lookup tables) that rewriting would corrupt.
_PRIVATE_BINARY_TEXT_VRS = frozenset({"UN", "OB"})

# Above this size a private binary value is treated as an opaque blob rather
# than as text: a value that cannot be shown to be PHI-free is emptied, which
# is what the Basic Profile does to private attributes anyway.
_MAX_PRIVATE_BINARY_SCAN_BYTES = 1 << 20  # 1 MiB

# Runs of printable ASCII (plus the usual whitespace) are the only part of a
# binary value that can be text. The framing bytes around them must never
# reach the NER pipeline: they are not PHI, and the recognisers' regexes
# backtrack catastrophically over long non-text byte runs -- a 2 KiB blob
# takes ~40s to analyse whole, against milliseconds for the strings in it.
_PRINTABLE_RUN_RE = re.compile(rb"[ -~\t\r\n]+")

# Shortest embedded string run in a binary blob still worth scanning. Below
# this, runs are file magic and framing rather than anything a human wrote.
# It does not apply to a value that is printable end to end -- see
# _redact_private_binary.
_MIN_BINARY_TEXT_RUN = 6


def _contains_ner_scanned_tag(ds: dicom.dataset.Dataset) -> bool:
    """Checks if the dicom has any of the tags in ``_NER_SCANNED_TAGS``.
    """
    for elem in ds:
        if elem.tag in _NER_SCANNED_TAGS:
            return True
        if elem.VR == "SQ":
            for sub_ds in elem.value:
                if isinstance(sub_ds, dicom.dataset.Dataset) and _contains_ner_scanned_tag(sub_ds):
                    return True
    return False


def _redact_text(text: str,
                 analyser: AnalyzerEngine,
                 anonymizer: AnonymizerEngine,
                 score_threshold: float,
                 gliner_pii=None) -> str:
    analyzer_results = analyser.analyze(text=text, language="en", score_threshold=score_threshold)
    redacted = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "XXXX"})},
    ).text
    if gliner_pii and len(redacted) > 30:
        redacted = _anonymise_with_transformer(gliner_pii, redacted, threshold=score_threshold, return_entities=False)
    return redacted


def _redact_private_binary(value: bytes,
                           analyser: AnalyzerEngine,
                           anonymizer: AnonymizerEngine,
                           score_threshold: float,
                           gliner_pii=None) -> bytes:
    """Scans the text embedded in a private UN/OB element and returns it redacted.

    Only the printable runs are scanned, each through the same pipeline as any
    text header. A value that is printable end to end is one string that
    happened to be typed as UN or OB -- how a private element of an Implicit
    VR file arrives when its creator is outside pydicom's private dictionary
    -- so it is scanned whole, however short; in a real binary blob only runs
    of at least ``_MIN_BINARY_TEXT_RUN`` characters are taken as text.

    Each redacted run is written back over the bytes it came from, padded or
    truncated to the run's original length, so the value's length and byte
    layout are preserved: private binary formats are length-prefixed and
    offset-addressed, and shifting their bytes would corrupt them. Bytes
    outside a scanned run are never touched.

    Parameters
    ----------
    value : bytes
        The element's raw value.

    analyser : AnalyzerEngine
        Presidio analyser engine carrying the custom recognisers.

    anonymizer : AnonymizerEngine
        Presidio anonymizer engine, used to replace the recognised spans.

    score_threshold : float
        Entities scoring below this are not redacted.

    gliner_pii : UniEncoderSpanGLiNER, optional
        If set, GLiNER runs on top of Presidio's output for long values.

    Returns
    -------
    bytes
        The redacted value, the same length as ``value``.

    Raises
    ------
    ValueError
        If the value is larger than ``_MAX_PRIVATE_BINARY_SCAN_BYTES``, so the
        caller can fail closed rather than let an unscanned value through.
    """
    if len(value) > _MAX_PRIVATE_BINARY_SCAN_BYTES:
        raise ValueError(
            f"private binary value of {len(value)} bytes is too large to scan "
            f"(limit {_MAX_PRIVATE_BINARY_SCAN_BYTES})"
        )
    runs = [match.span() for match in _PRINTABLE_RUN_RE.finditer(value)]
    if runs != [(0, len(value))]:
        runs = [(start, end) for start, end in runs if end - start >= _MIN_BINARY_TEXT_RUN]
    redacted_value = bytearray(value)
    # A vendor blob repeats the same strings over and over (a CSA header holds
    # thousands of runs but few distinct ones), and the pipeline is
    # deterministic, so each distinct run is only ever analysed once.
    seen: dict[bytes, bytes] = {}
    for start, end in runs:
        run = value[start:end]
        redacted = seen.get(run)
        if redacted is None:
            redacted = _redact_text(
                run.decode("ascii"), analyser, anonymizer, score_threshold, gliner_pii,
            ).encode("ascii", errors="replace")
            seen[run] = redacted
        if redacted == run:
            continue
        # Padding is only ever added, and truncation only ever drops the tail
        # of an already-redacted string, so neither can reinstate PHI.
        redacted_value[start:end] = redacted[:end - start].ljust(end - start, b" ")
    return bytes(redacted_value)


def _anonymise_ds(ds: dicom.dataset.Dataset,
                  analyser: AnalyzerEngine,
                  anonymizer: AnonymizerEngine,
                  score_threshold: float,
                  gliner_pii=None,
                  use_case: str='Standard',
                  anonymised_headers: list | None = None,
                  private_only: bool = False) -> None:
    """Recursively anonymises all elements in a DICOM dataset in-place.

    When ``private_only`` is True, only private attributes and the free-text
    attributes in ``_NER_SCANNED_TAGS`` have their values scanned/redacted; the
    remaining standard attributes are left untouched (the caller has already
    de-identified them, e.g. via the PS3.15 Basic Profile).

    Text-valued elements are scanned by the NER pipeline. Private elements
    carrying text under a binary VR (``_PRIVATE_BINARY_TEXT_VRS``) are scanned
    too, which is what keeps a private block whose VRs pydicom could not
    resolve -- an Implicit VR file with a creator outside its private
    dictionary -- from going through unscanned.

    Private creator elements are never scrubbed:
    the creator string identifies the block's owner; redacting it'd corrupt the creator-to-data mapping of every element in the block.
    """
    if anonymised_headers is None:
        anonymised_headers = []
    for elem in ds:
        if elem.tag in _STRUCTURAL_TAGS:
            continue
        if elem.VR == "SQ":
            for sub_ds in elem.value:
                if not isinstance(sub_ds, dicom.dataset.Dataset):
                    continue
                _anonymise_ds(
                    sub_ds, analyser, anonymizer, score_threshold,
                    gliner_pii, use_case,
                    anonymised_headers, private_only
                )
            continue
        if elem.tag.is_private_creator:
            continue  # Private creators ("SIEMENS CSA HEADER") never touched.
        if private_only and not elem.tag.is_private and elem.tag not in _NER_SCANNED_TAGS:
            continue
        if elem.VR == "PN" or elem.tag == (0x0010, 0x0010):
            ds[elem.tag].value = PersonName("XXXX")
            anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})
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
            anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})
        elif elem.VR == "AS":
            if str(elem.value).strip() in ("", "000Y"):
                continue
            ds[elem.tag].value = "000Y"
            anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})
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
                new_values = [
                    _redact_text(v, analyser, anonymizer, score_threshold, gliner_pii)
                    for v in values
                ]
                if new_values != values:
                    anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})
                if is_multi:
                    ds[elem.tag].value = dicom.multival.MultiValue(str, new_values)
                else:
                    ds[elem.tag].value = new_values[0]
            except Exception as e:
                # A value that couldn't be analysed may contain PHI, so blank it.
                logger.error(
                    "Failed to redact %s (%s), blanking it. %s: %s",
                    elem.tag, elem.name, type(e).__name__, e,
                )
                try:
                    ds[elem.tag].value = ""
                except Exception:
                    del ds[elem.tag]
                anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})
        elif elem.tag.is_private and elem.VR in _PRIVATE_BINARY_TEXT_VRS:
            # Private text hiding under a binary VR -- see
            # _PRIVATE_BINARY_TEXT_VRS. Skipping these lets a whole private
            # block through unscanned whenever its VRs could not be resolved.
            try:
                original = elem.value
                if original is None or len(original) == 0:
                    continue
                original = bytes(original)
                redacted = _redact_private_binary(
                    original, analyser, anonymizer, score_threshold, gliner_pii,
                )
                if redacted != original:
                    ds[elem.tag].value = redacted
                    anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})
            except Exception as e:
                # Same fail-closed rule as for text: a value that could not be
                # scanned may contain PHI, so it does not survive.
                logger.error(
                    "Failed to redact private binary %s (%s), emptying it. %s: %s",
                    elem.tag, elem.name, type(e).__name__, e,
                )
                try:
                    ds[elem.tag].value = b""
                except Exception:
                    del ds[elem.tag]
                anonymised_headers.append({"tag": str(elem.tag), "name": elem.name, "source": SOURCE_NER})


def anonymise_image(ds: dicom.dataset.FileDataset,
                    analyser: AnalyzerEngine=None,
                    anonymizer: AnonymizerEngine=None,
                    image_redactor: DicomImageRedactorEngine = None,
                    score_threshold: float=0.5,
                    gliner_pii: UniEncoderSpanGLiNER=None,
                    use_case: str='Standard',
                    spacy_model_name: str="en_core_web_lg",
                    ) -> dicom.dataset.FileDataset:
    """Anonymises a DICOM image by redacting personal information.

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
    
    gliner_pii: UniEncoderSpanGLiNER, optional (default False)
        If set, the model will be used for anonymisation on top of Presidio's output.

    use_case : str, optional (default 'dicom_retain_patient_scan_private')
        * PS3.15 (alias 'dicom_default'): headers are de-identified with the
        DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile;
        Presidio and GLiNER are not used on the headers.
        PS3.15_Rtn. Pat. (alias 'dicom_retain_patient'): as PS3.15, plus the
        * Retain Patient Characteristics Option, so patient characteristics
        (age, sex, weight, ...) are kept.
        * 'dicom_default_scan_private' / 'dicom_retain_patient_scan_private': as
        the matching PS3.15 variant for the standard headers, but private
        attributes are kept and scanned with the Presidio/GLiNER pipeline
        instead of being removed.
        * Any other value: use Presidio (plus GLiNER when gliner_pii is given).

    spacy_model_name : str, optional (default "en_core_web_lg")
        Only used when ``analyser`` is not supplied and one has to be built here.

    Returns
    -------
    pydicom.dataset.FileDataset
        The anonymised DICOM.
    """
    new_dict_items = {
        # Private tag to store the list of anonymised headers. UT rather than
        # LT because the list can exceed LT's 10240-character limit.
        0x02091000: ('UT', '1', 'Flagged Headers PHI-Finder')
    }
    add_private_dict_entries(private_creator="phi-finder", new_entries_dict=new_dict_items)

    ps3_15_mode = ps3_15.is_ps3_15_use_case(use_case)
    scan_private = ps3_15.scan_private_headers(use_case)
    ner_scanned_tags_present = ps3_15_mode and _contains_ner_scanned_tag(ds)
    if not ps3_15_mode or scan_private or ner_scanned_tags_present:
        if analyser is None:
            analyser = _build_presidio_analyser(score_threshold, spacy_model_name)
        if anonymizer is None:
            anonymizer = AnonymizerEngine()
    if image_redactor is not None:
        ds = image_redactor.redact(ds, fill="contrast", score_threshold=score_threshold, ocr_kwargs={"config": "--psm 11 --oem 1"})  # fill="background") --psm 11 ("sparse text)
    # operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})}

    anonymised_headers = []
    if ps3_15_mode:
        ps3_15.apply_basic_profile(
            ds, anonymised_headers,
            retain_patient_characteristics=ps3_15.retain_patient_characteristics(use_case),
            scan_private=scan_private,
        )
        if scan_private or ner_scanned_tags_present:
            _anonymise_ds(ds, analyser, anonymizer, score_threshold,
                          gliner_pii, use_case, anonymised_headers,
                          private_only=True)
    else:
        _anonymise_ds(ds, analyser, anonymizer, score_threshold,
                      gliner_pii, use_case, anonymised_headers)
    '''
    Adding a private header with the flagged headers list.
    private_block() reserves a slot (e.g., 0x10) and writes the creator name at (0x0209, 0x0010).
    The actual data then lives at (0x0209, 0x10XX).
    Then, ds.add_new([0x0209, 0x0010], ...) overwrites the Private Creator element itself.
    '''
    flagged_headers = json.dumps(anonymised_headers)
    block = ds.private_block(0x0209, "phi-finder", create=True)
    block.add_new(0x00, 'UT', flagged_headers)  # 0x00 offset within block → maps to (0x0209, 0x1000)
    return ds
