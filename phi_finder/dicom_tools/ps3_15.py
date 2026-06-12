"""DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile table.

Generated from Table E.1-1 of the current DICOM standard (column "Basic Prof."):
https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html

Actions:
    D   replace with a dummy value consistent with the VR
    Z   replace with a zero-length (empty) value
    X   remove the attribute
    U   replace the UID with a consistent, deterministically generated one

Combined actions (e.g. X/Z, X/Z/D, X/Z/U*) depend on IOD type-1/2
requirements, which are not tracked here; apply_basic_profile() resolves
them to the value-preserving option so the de-identified file stays valid
for any IOD.

Rows that are not plain (group,element) tags are handled in code rather
than in this table: private attributes (odd group, X), Curve Data
(50xx,xxxx, X), Overlay Data (60xx,3000, X) and Overlay Comments
(60xx,4000, X).
"""
import logging

import pydicom as dicom
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)

BASIC_PROFILE_ACTIONS: dict[int, str] = {
    0x00001000: "X",  # Affected SOP Instance UID
    0x00001001: "U",  # Requested SOP Instance UID
    0x00020003: "U",  # Media Storage SOP Instance UID
    0x00041511: "U",  # Referenced SOP Instance UID in File
    0x00080012: "X/D",  # Instance Creation Date
    0x00080013: "X/Z/D",  # Instance Creation Time
    0x00080014: "U",  # Instance Creator UID
    0x00080015: "X",  # Instance Coercion DateTime
    0x00080017: "U",  # Acquisition UID
    0x00080018: "U",  # SOP Instance UID
    0x00080019: "U",  # Pyramid UID
    0x00080020: "Z",  # Study Date
    0x00080021: "X/D",  # Series Date
    0x00080022: "X/Z",  # Acquisition Date
    0x00080023: "Z/D",  # Content Date
    0x00080024: "X",  # Overlay Date
    0x00080025: "X",  # Curve Date
    0x0008002A: "X/Z/D",  # Acquisition DateTime
    0x00080030: "Z",  # Study Time
    0x00080031: "X/D",  # Series Time
    0x00080032: "X/Z",  # Acquisition Time
    0x00080033: "Z/D",  # Content Time
    0x00080034: "X",  # Overlay Time
    0x00080035: "X",  # Curve Time
    0x00080050: "Z",  # Accession Number
    0x00080054: "X",  # Retrieve AE Title
    0x00080055: "X",  # Station AE Title
    0x00080058: "U",  # Failed SOP Instance UID List
    0x00080080: "X/Z/D",  # Institution Name
    0x00080081: "X",  # Institution Address
    0x00080082: "X/Z/D",  # Institution Code Sequence
    0x00080090: "Z",  # Referring Physician's Name
    0x00080092: "X",  # Referring Physician's Address
    0x00080094: "X",  # Referring Physician's Telephone Numbers
    0x00080096: "X",  # Referring Physician Identification Sequence
    0x0008009C: "Z",  # Consulting Physician's Name
    0x0008009D: "X",  # Consulting Physician Identification Sequence
    0x00080106: "D",  # Context Group Version
    0x00080107: "D",  # Context Group Local Version
    0x00080201: "X",  # Timezone Offset From UTC
    0x00081000: "X",  # Network ID
    0x00081010: "X/Z/D",  # Station Name
    0x00081030: "X",  # Study Description
    0x0008103E: "X",  # Series Description
    0x00081040: "X",  # Institutional Department Name
    0x00081041: "X",  # Institutional Department Type Code Sequence
    0x00081048: "X",  # Physician(s) of Record
    0x00081049: "X",  # Physician(s) of Record Identification Sequence
    0x00081050: "X",  # Performing Physician's Name
    0x00081052: "X",  # Performing Physician Identification Sequence
    0x00081060: "X",  # Name of Physician(s) Reading Study
    0x00081062: "X",  # Physician(s) Reading Study Identification Sequence
    0x00081070: "X/Z/D",  # Operators' Name
    0x00081072: "X/D",  # Operator Identification Sequence
    0x00081080: "X",  # Admitting Diagnoses Description
    0x00081084: "X",  # Admitting Diagnoses Code Sequence
    0x00081088: "X",  # Pyramid Description
    0x00081110: "X/Z",  # Referenced Study Sequence
    0x00081111: "X/Z/D",  # Referenced Performed Procedure Step Sequence
    0x00081120: "X",  # Referenced Patient Sequence
    0x00081140: "X/Z/U*",  # Referenced Image Sequence
    0x00081155: "U",  # Referenced SOP Instance UID
    0x00081195: "U",  # Transaction UID
    0x00081301: "X",  # Principal Diagnosis Code Sequence
    0x00081302: "X",  # Primary Diagnosis Code Sequence
    0x00081303: "X",  # Secondary Diagnoses Code Sequence
    0x00081304: "X",  # Histological Diagnoses Code Sequence
    0x00082111: "X",  # Derivation Description
    0x00082112: "X/Z/U*",  # Source Image Sequence
    0x00083010: "U",  # Irradiation Event UID
    0x00084000: "X",  # Identifying Comments
    0x00100010: "Z",  # Patient's Name
    0x00100011: "X",  # Person Names to Use Sequence
    0x00100012: "X",  # Name to Use
    0x00100013: "X",  # Name to Use Comment
    0x00100014: "X",  # Third Person Pronouns Sequence
    0x00100015: "X",  # Pronoun Code Sequence
    0x00100016: "X",  # Pronoun Comment
    0x00100020: "Z/D",  # Patient ID
    0x00100021: "X",  # Issuer of Patient ID
    0x00100030: "Z",  # Patient's Birth Date
    0x00100032: "X",  # Patient's Birth Time
    0x00100040: "Z",  # Patient's Sex
    0x00100041: "X",  # Gender Identity Sequence
    0x00100042: "X",  # Sex Parameters for Clinical Use Category Comment
    0x00100043: "X",  # Sex Parameters for Clinical Use Category Sequence
    0x00100044: "X",  # Gender Identity Code Sequence
    0x00100045: "X",  # Gender Identity Comment
    0x00100046: "X",  # Sex Parameters for Clinical Use Category Code Sequence
    0x00100047: "X",  # Sex Parameters for Clinical Use Category Reference
    0x00100050: "X",  # Patient's Insurance Plan Code Sequence
    0x00100101: "X",  # Patient's Primary Language Code Sequence
    0x00100102: "X",  # Patient's Primary Language Modifier Code Sequence
    0x00101000: "X",  # Other Patient IDs
    0x00101001: "X",  # Other Patient Names
    0x00101002: "X",  # Other Patient IDs Sequence
    0x00101005: "X",  # Patient's Birth Name
    0x00101010: "X",  # Patient's Age
    0x00101020: "X",  # Patient's Size
    0x00101030: "X",  # Patient's Weight
    0x00101040: "X",  # Patient's Address
    0x00101050: "X",  # Insurance Plan Identification
    0x00101060: "X",  # Patient's Mother's Birth Name
    0x00101080: "X",  # Military Rank
    0x00101081: "X",  # Branch of Service
    0x00101090: "X",  # Medical Record Locator
    0x00101100: "X",  # Referenced Patient Photo Sequence
    0x00102000: "X",  # Medical Alerts
    0x00102110: "X",  # Allergies
    0x00102150: "X",  # Country of Residence
    0x00102152: "X",  # Region of Residence
    0x00102154: "X",  # Patient's Telephone Numbers
    0x00102155: "X",  # Patient's Telecom Information
    0x00102160: "X",  # Ethnic Group
    0x00102161: "X",  # Ethnic Group Code Sequence
    0x00102162: "X",  # Ethnic Groups
    0x00102180: "X",  # Occupation
    0x001021A0: "X",  # Smoking Status
    0x001021B0: "X",  # Additional Patient History
    0x001021C0: "X",  # Pregnancy Status
    0x001021D0: "X",  # Last Menstrual Date
    0x001021F0: "X",  # Patient's Religious Preference
    0x00102203: "X/Z",  # Patient's Sex Neutered
    0x00102297: "X",  # Responsible Person
    0x00102299: "X",  # Responsible Organization
    0x00104000: "X",  # Patient Comments
    0x00120010: "D",  # Clinical Trial Sponsor Name
    0x00120020: "D",  # Clinical Trial Protocol ID
    0x00120021: "Z",  # Clinical Trial Protocol Name
    0x00120022: "X",  # Issuer of Clinical Trial Protocol ID
    0x00120023: "X",  # Other Clinical Trial Protocol IDs Sequence
    0x00120030: "Z",  # Clinical Trial Site ID
    0x00120031: "Z",  # Clinical Trial Site Name
    0x00120032: "X",  # Issuer of Clinical Trial Site ID
    0x00120040: "D",  # Clinical Trial Subject ID
    0x00120041: "X",  # Issuer of Clinical Trial Subject ID
    0x00120042: "D",  # Clinical Trial Subject Reading ID
    0x00120043: "X",  # Issuer of Clinical Trial Subject Reading ID
    0x00120050: "Z",  # Clinical Trial Time Point ID
    0x00120051: "X",  # Clinical Trial Time Point Description
    0x00120055: "X",  # Issuer of Clinical Trial Time Point ID
    0x00120060: "Z",  # Clinical Trial Coordinating Center Name
    0x00120071: "X",  # Clinical Trial Series ID
    0x00120072: "X",  # Clinical Trial Series Description
    0x00120073: "X",  # Issuer of Clinical Trial Series ID
    0x00120081: "D",  # Clinical Trial Protocol Ethics Committee Name
    0x00120082: "X",  # Clinical Trial Protocol Ethics Committee Approval Number
    0x00120086: "X",  # Ethics Committee Approval Effectiveness Start Date
    0x00120087: "X",  # Ethics Committee Approval Effectiveness End Date
    0x0014407C: "X",  # Calibration Time
    0x0014407E: "X",  # Calibration Date
    0x0016002B: "X",  # Maker Note
    0x0016004B: "X",  # Device Setting Description
    0x0016004D: "X",  # Camera Owner Name
    0x0016004E: "X",  # Lens Specification
    0x0016004F: "X",  # Lens Make
    0x00160050: "X",  # Lens Model
    0x00160051: "X",  # Lens Serial Number
    0x00160070: "X",  # GPS Version ID
    0x00160071: "X",  # GPS Latitude​ Ref
    0x00160072: "X",  # GPS Latitude​
    0x00160073: "X",  # GPS Longitude Ref
    0x00160074: "X",  # GPS Longitude
    0x00160075: "X",  # GPS Altitude​ Ref
    0x00160076: "X",  # GPS Altitude​
    0x00160077: "X",  # GPS Time​ Stamp
    0x00160078: "X",  # GPS Satellites
    0x00160079: "X",  # GPS Status
    0x0016007A: "X",  # GPS Measure ​Mode
    0x0016007B: "X",  # GPS DOP
    0x0016007C: "X",  # GPS Speed​ Ref
    0x0016007D: "X",  # GPS Speed​
    0x0016007E: "X",  # GPS Track ​Ref
    0x0016007F: "X",  # GPS Track
    0x00160080: "X",  # GPS Img​ Direction Ref
    0x00160081: "X",  # GPS Img ​Direction
    0x00160082: "X",  # GPS Map​ Datum
    0x00160083: "X",  # GPS Dest​ Latitude Ref
    0x00160084: "X",  # GPS Dest​ Latitude
    0x00160085: "X",  # GPS Dest ​Longitude Ref
    0x00160086: "X",  # GPS Dest ​Longitude
    0x00160087: "X",  # GPS Dest​ Bearing Ref
    0x00160088: "X",  # GPS Dest ​Bearing
    0x00160089: "X",  # GPS Dest ​Distance Ref
    0x0016008A: "X",  # GPS Dest ​Distance
    0x0016008B: "X",  # GPS Processing​ Method
    0x0016008C: "X",  # GPS Area ​Information
    0x0016008D: "X",  # GPS Date​ Stamp
    0x0016008E: "X",  # GPS Differential
    0x00180010: "Z/D",  # Contrast/Bolus Agent
    0x00180027: "X",  # Intervention Drug Stop Time
    0x00180035: "X",  # Intervention Drug Start Time
    0x00181000: "X/Z/D",  # Device Serial Number
    0x00181002: "U",  # Device UID
    0x00181004: "X",  # Plate ID
    0x00181005: "X",  # Generator ID
    0x00181007: "X",  # Cassette ID
    0x00181008: "X",  # Gantry ID
    0x00181009: "X",  # Unique Device Identifier
    0x0018100A: "X",  # UDI Sequence
    0x0018100B: "U",  # Manufacturer's Device Class UID
    0x00181010: "X",  # Secondary Capture Device ID
    0x00181011: "X",  # Hardcopy Creation Device ID
    0x00181012: "X",  # Date of Secondary Capture
    0x00181014: "X",  # Time of Secondary Capture
    0x00181030: "X/D",  # Protocol Name
    0x00181042: "X",  # Contrast/Bolus Start Time
    0x00181043: "X",  # Contrast/Bolus Stop Time
    0x00181072: "X",  # Radiopharmaceutical Start Time
    0x00181073: "X",  # Radiopharmaceutical Stop Time
    0x00181078: "X",  # Radiopharmaceutical Start DateTime
    0x00181079: "X",  # Radiopharmaceutical Stop DateTime
    0x001811BB: "D",  # Acquisition Field Of View Label
    0x00181200: "X",  # Date of Last Calibration
    0x00181201: "X",  # Time of Last Calibration
    0x00181202: "X",  # DateTime of Last Calibration
    0x00181203: "Z",  # Calibration DateTime
    0x00181204: "X",  # Date of Manufacture
    0x00181205: "X",  # Date of Installation
    0x00181400: "X/D",  # Acquisition Device Processing Description
    0x00182042: "U",  # Target UID
    0x00184000: "X",  # Acquisition Comments
    0x00185011: "X",  # Transducer Identification Sequence
    0x0018700A: "X/D",  # Detector ID
    0x0018700C: "X/D",  # Date of Last Detector Calibration
    0x0018700E: "X/D",  # Time of Last Detector Calibration
    0x00189074: "D",  # Frame Acquisition DateTime
    0x00189151: "D",  # Frame Reference DateTime
    0x00189185: "X",  # Respiratory Motion Compensation Technique Description
    0x00189367: "D",  # X-Ray Source ID
    0x00189369: "D",  # Source Start DateTime
    0x0018936A: "D",  # Source End DateTime
    0x00189371: "D",  # X-Ray Detector ID
    0x00189373: "X",  # X-Ray Detector Label
    0x0018937B: "X",  # Multi-energy Acquisition Description
    0x0018937F: "X",  # Decomposition Description
    0x00189424: "X",  # Acquisition Protocol Description
    0x00189516: "X/D",  # Start Acquisition DateTime
    0x00189517: "X/D",  # End Acquisition DateTime
    0x00189623: "D",  # Functional Sync Pulse
    0x00189701: "D",  # Decay Correction DateTime
    0x00189804: "D",  # Exclusion Start DateTime
    0x00189919: "Z/D",  # Instruction Performed DateTime
    0x00189937: "X",  # Requested Series Description
    0x0018A002: "X",  # Contribution DateTime
    0x0018A003: "X",  # Contribution Description
    0x0020000D: "U",  # Study Instance UID
    0x0020000E: "U",  # Series Instance UID
    0x00200010: "Z",  # Study ID
    0x00200027: "X",  # Pyramid Label
    0x00200052: "U",  # Frame of Reference UID
    0x00200200: "U",  # Synchronization Frame of Reference UID
    0x00203401: "X",  # Modifying Device ID
    0x00203403: "X",  # Modified Image Date
    0x00203405: "X",  # Modified Image Time
    0x00203406: "X",  # Modified Image Description
    0x00204000: "X",  # Image Comments
    0x00209158: "X",  # Frame Comments
    0x00209161: "U",  # Concatenation UID
    0x00209164: "U",  # Dimension Organization UID
    0x00281199: "U",  # Palette Color Lookup Table UID
    0x00281214: "U",  # Large Palette Color Lookup Table UID
    0x00284000: "X",  # Image Presentation Comments
    0x00320012: "X",  # Study ID Issuer
    0x00320032: "X",  # Study Verified Date
    0x00320033: "X",  # Study Verified Time
    0x00320034: "X",  # Study Read Date
    0x00320035: "X",  # Study Read Time
    0x00321000: "X",  # Scheduled Study Start Date
    0x00321001: "X",  # Scheduled Study Start Time
    0x00321010: "X",  # Scheduled Study Stop Date
    0x00321011: "X",  # Scheduled Study Stop Time
    0x00321020: "X",  # Scheduled Study Location
    0x00321021: "X",  # Scheduled Study Location AE Title
    0x00321030: "X",  # Reason for Study
    0x00321032: "X",  # Requesting Physician
    0x00321033: "X",  # Requesting Service
    0x00321040: "X",  # Study Arrival Date
    0x00321041: "X",  # Study Arrival Time
    0x00321050: "X",  # Study Completion Date
    0x00321051: "X",  # Study Completion Time
    0x00321060: "X/Z",  # Requested Procedure Description
    0x00321066: "X",  # Reason for Visit
    0x00321067: "X",  # Reason for Visit Code Sequence
    0x00321070: "X",  # Requested Contrast Agent
    0x00324000: "X",  # Study Comments
    0x00340001: "D",  # Flow Identifier Sequence
    0x00340002: "D",  # Flow Identifier
    0x00340005: "D",  # Source Identifier
    0x00340007: "D",  # Frame Origin Timestamp
    0x00380004: "X",  # Referenced Patient Alias Sequence
    0x00380010: "X",  # Admission ID
    0x00380011: "X",  # Issuer of Admission ID
    0x00380014: "X",  # Issuer of Admission ID Sequence
    0x0038001A: "X",  # Scheduled Admission Date
    0x0038001B: "X",  # Scheduled Admission Time
    0x0038001C: "X",  # Scheduled Discharge Date
    0x0038001D: "X",  # Scheduled Discharge Time
    0x0038001E: "X",  # Scheduled Patient Institution Residence
    0x00380020: "X",  # Admitting Date
    0x00380021: "X",  # Admitting Time
    0x00380030: "X",  # Discharge Date
    0x00380032: "X",  # Discharge Time
    0x00380040: "X",  # Discharge Diagnosis Description
    0x00380050: "X",  # Special Needs
    0x00380060: "X",  # Service Episode ID
    0x00380061: "X",  # Issuer of Service Episode ID
    0x00380062: "X",  # Service Episode Description
    0x00380064: "X",  # Issuer of Service Episode ID Sequence
    0x00380300: "X",  # Current Patient Location
    0x00380400: "X",  # Patient's Institution Residence
    0x00380500: "X",  # Patient State
    0x00384000: "X",  # Visit Comments
    0x003A0020: "X",  # Multiplex Group Label
    0x003A0203: "X",  # Channel Label
    0x003A020C: "X",  # Channel Derivation Description
    0x003A0310: "U",  # Multiplex Group UID
    0x003A0314: "D",  # Impedance Measurement DateTime
    0x003A0329: "X",  # ​Waveform Filter Description
    0x003A032B: "X",  # Filter Lookup Table Description
    0x00400001: "X",  # Scheduled Station AE Title
    0x00400002: "X",  # Scheduled Procedure Step Start Date
    0x00400003: "X",  # Scheduled Procedure Step Start Time
    0x00400004: "X",  # Scheduled Procedure Step End Date
    0x00400005: "X",  # Scheduled Procedure Step End Time
    0x00400006: "X",  # Scheduled Performing Physician's Name
    0x00400007: "X",  # Scheduled Procedure Step Description
    0x00400009: "X",  # Scheduled Procedure Step ID
    0x0040000B: "X",  # Scheduled Performing Physician Identification Sequence
    0x00400010: "X",  # Scheduled Station Name
    0x00400011: "X",  # Scheduled Procedure Step Location
    0x00400012: "X",  # Pre-Medication
    0x00400241: "X",  # Performed Station AE Title
    0x00400242: "X",  # Performed Station Name
    0x00400243: "X",  # Performed Location
    0x00400244: "X",  # Performed Procedure Step Start Date
    0x00400245: "X",  # Performed Procedure Step Start Time
    0x00400250: "X",  # Performed Procedure Step End Date
    0x00400251: "X",  # Performed Procedure Step End Time
    0x00400253: "X",  # Performed Procedure Step ID
    0x00400254: "X",  # Performed Procedure Step Description
    0x00400275: "X",  # Request Attributes Sequence
    0x00400280: "X",  # Comments on the Performed Procedure Step
    0x00400310: "X",  # Comments on Radiation Dose
    0x0040050A: "X",  # Specimen Accession Number
    0x00400512: "D",  # Container Identifier
    0x00400513: "Z",  # Issuer of the Container Identifier Sequence
    0x0040051A: "X",  # Container Description
    0x00400551: "D",  # Specimen Identifier
    0x00400554: "U",  # Specimen UID
    0x00400555: "X/Z",  # Acquisition Context Sequence
    0x00400556: "X",  # Acquisition Context Description
    0x00400562: "Z",  # Issuer of the Specimen Identifier Sequence
    0x00400600: "X",  # Specimen Short Description
    0x00400602: "X",  # Specimen Detailed Description
    0x00400610: "Z",  # Specimen Preparation Sequence
    0x004006FA: "X",  # Slide Identifier
    0x00401001: "X",  # Requested Procedure ID
    0x00401002: "X",  # Reason for the Requested Procedure
    0x00401004: "X",  # Patient Transport Arrangements
    0x00401005: "X",  # Requested Procedure Location
    0x0040100A: "X",  # Reason for Requested Procedure Code Sequence
    0x00401010: "X",  # Names of Intended Recipients of Results
    0x00401011: "X",  # Intended Recipients of Results Identification Sequence
    0x00401101: "D",  # Person Identification Code Sequence
    0x00401102: "X",  # Person's Address
    0x00401103: "X",  # Person's Telephone Numbers
    0x00401104: "X",  # Person's Telecom Information
    0x00401400: "X",  # Requested Procedure Comments
    0x00402001: "X",  # Reason for the Imaging Service Request
    0x00402004: "X",  # Issue Date of Imaging Service Request
    0x00402005: "X",  # Issue Time of Imaging Service Request
    0x00402008: "X",  # Order Entered By
    0x00402009: "X",  # Order Enterer's Location
    0x00402010: "X",  # Order Callback Phone Number
    0x00402011: "X",  # Order Callback Telecom Information
    0x00402016: "Z",  # Placer Order Number / Imaging Service Request
    0x00402017: "Z",  # Filler Order Number / Imaging Service Request
    0x00402400: "X",  # Imaging Service Request Comments
    0x00403001: "X",  # Confidentiality Constraint on Patient Data Description
    0x00404005: "X",  # Scheduled Procedure Step Start DateTime
    0x00404008: "X",  # Scheduled Procedure Step Expiration DateTime
    0x00404010: "X",  # Scheduled Procedure Step Modification DateTime
    0x00404011: "X",  # Expected Completion DateTime
    0x00404023: "U",  # Referenced General Purpose Scheduled Procedure Step Transaction UID
    0x00404025: "X",  # Scheduled Station Name Code Sequence
    0x00404027: "X",  # Scheduled Station Geographic Location Code Sequence
    0x00404028: "X",  # Performed Station Name Code Sequence
    0x00404030: "X",  # Performed Station Geographic Location Code Sequence
    0x00404034: "X",  # Scheduled Human Performers Sequence
    0x00404035: "X",  # Actual Human Performers Sequence
    0x00404036: "X",  # Human Performer's Organization
    0x00404037: "X",  # Human Performer's Name
    0x00404050: "X",  # Performed Procedure Step Start DateTime
    0x00404051: "X",  # Performed Procedure Step End DateTime
    0x00404052: "X",  # Procedure Step Cancellation DateTime
    0x0040A023: "X",  # Findings Group Recording Date (Trial)
    0x0040A024: "X",  # Findings Group Recording Time (Trial)
    0x0040A027: "D",  # Verifying Organization
    0x0040A030: "D",  # Verification DateTime
    0x0040A032: "X/D",  # Observation DateTime
    0x0040A033: "X",  # Observation Start DateTime
    0x0040A034: "X",  # Effective Start DateTime
    0x0040A035: "X",  # Effective Stop DateTime
    0x0040A073: "D",  # Verifying Observer Sequence
    0x0040A075: "D",  # Verifying Observer Name
    0x0040A078: "X",  # Author Observer Sequence
    0x0040A07A: "X",  # Participant Sequence
    0x0040A07C: "X",  # Custodial Organization Sequence
    0x0040A082: "Z",  # Participation DateTime
    0x0040A088: "Z",  # Verifying Observer Identification Code Sequence
    0x0040A110: "X",  # Date of Document or Verbal Transaction (Trial)
    0x0040A112: "X",  # Time of Document Creation or Verbal Transaction (Trial)
    0x0040A120: "D",  # DateTime
    0x0040A121: "D",  # Date
    0x0040A122: "D",  # Time
    0x0040A123: "D",  # Person Name
    0x0040A124: "U",  # UID
    0x0040A13A: "D",  # Referenced DateTime
    0x0040A171: "U",  # Observation UID
    0x0040A172: "U",  # Referenced Observation UID (Trial)
    0x0040A192: "X",  # Observation Date (Trial)
    0x0040A193: "X",  # Observation Time (Trial)
    0x0040A307: "X",  # Current Observer (Trial)
    0x0040A352: "X",  # Verbal Source (Trial)
    0x0040A353: "X",  # Address (Trial)
    0x0040A354: "X",  # Telephone Number (Trial)
    0x0040A358: "X",  # Verbal Source Identifier Code Sequence (Trial)
    0x0040A402: "U",  # Observation Subject UID (Trial)
    0x0040A730: "D",  # Content Sequence
    0x0040B020: "X/D",  # Waveform Annotation Sequence
    0x0040B034: "X",  # Annotation DateTime
    0x0040B036: "X",  # Segment Definition DateTime
    0x0040B03B: "X",  # Montage Name
    0x0040B03F: "X",  # Montage Channel Label
    0x0040DB06: "X",  # Template Version
    0x0040DB07: "X",  # Template Local Version
    0x0040DB0C: "U",  # Template Extension Organization UID
    0x0040DB0D: "U",  # Template Extension Creator UID
    0x0040E004: "X",  # HL7 Document Effective Time
    0x00420011: "D",  # Encapsulated Document
    0x00440004: "X",  # Approval Status DateTime
    0x0044000B: "X",  # Product Expiration DateTime
    0x00440010: "X",  # Substance Administration DateTime
    0x00440104: "D",  # Assertion DateTime
    0x00440105: "X",  # Assertion Expiration DateTime
    0x0050001B: "X",  # Container Component ID
    0x00500020: "X",  # Device Description
    0x00500021: "X",  # Long Device Description
    0x00620021: "U",  # Tracking UID
    0x00640003: "U",  # Source Frame of Reference UID
    0x00686226: "D",  # Effective DateTime
    0x00686270: "D",  # Information Issue DateTime
    0x006A0003: "D",  # Annotation Group UID
    0x006A0005: "D",  # Annotation Group Label
    0x006A0006: "X",  # Annotation Group Description
    0x00700001: "D",  # Graphic Annotation Sequence
    0x00700006: "D",  # Unformatted Text Value
    0x00700082: "X",  # Presentation Creation Date
    0x00700083: "X",  # Presentation Creation Time
    0x00700084: "Z/D",  # Content Creator's Name
    0x00700086: "X",  # Content Creator's Identification Code Sequence
    0x0070031A: "U",  # Fiducial UID
    0x00701101: "U",  # Presentation Display Collection UID
    0x00701102: "U",  # Presentation Sequence Collection UID
    0x0072000A: "D",  # Hanging Protocol Creation DateTime
    0x0072005E: "D",  # Selector AE Value
    0x0072005F: "D",  # Selector AS Value
    0x00720061: "D",  # Selector DA Value
    0x00720063: "D",  # Selector DT Value
    0x00720065: "D",  # Selector OB Value
    0x00720066: "D",  # Selector LO Value
    0x00720068: "D",  # Selector LT Value
    0x0072006A: "D",  # Selector PN Value
    0x0072006B: "D",  # Selector TM Value
    0x0072006C: "D",  # Selector SH Value
    0x0072006D: "D",  # Selector UN Value
    0x0072006E: "D",  # Selector ST Value
    0x00720070: "D",  # Selector UT Value
    0x00720071: "D",  # Selector UR Value
    0x00741234: "X",  # Receiving AE
    0x00741236: "X",  # Requesting AE
    0x00880140: "U",  # Storage Media File-set UID
    0x00880200: "X",  # Icon Image Sequence (see Note 11)
    0x00880904: "X",  # Topic Title
    0x00880906: "X",  # Topic Subject
    0x00880910: "X",  # Topic Author
    0x00880912: "X",  # Topic Keywords
    0x01000420: "X",  # SOP Authorization DateTime
    0x04000100: "U",  # Digital Signature UID
    0x04000105: "D",  # Digital Signature DateTime
    0x04000115: "D",  # Certificate of Signer
    0x04000310: "X",  # Certified Timestamp
    0x04000402: "X",  # Referenced Digital Signature Sequence
    0x04000403: "X",  # Referenced SOP Instance MAC Sequence
    0x04000404: "X",  # MAC
    0x04000550: "X",  # Modified Attributes Sequence
    0x04000551: "X",  # Nonconforming Modified Attributes Sequence
    0x04000552: "X",  # Nonconforming Data Element Value
    0x04000561: "X",  # Original Attributes Sequence
    0x04000562: "D",  # Attribute Modification DateTime
    0x04000563: "D",  # Modifying System
    0x04000564: "Z",  # Source of Previous Values
    0x04000565: "D",  # Reason for the Attribute Modification
    0x04000600: "X",  # Instance Origin Status
    0x20300020: "X",  # Text String
    0x21000040: "X",  # Creation Date
    0x21000050: "X",  # Creation Time
    0x21000070: "X",  # Originator
    0x21000140: "D",  # Destination AE
    0x22000002: "X/Z",  # Label Text
    0x22000005: "X/Z",  # Barcode Value
    0x30020121: "X",  # Position Acquisition Template Name
    0x30020123: "X",  # Position Acquisition Template Description
    0x30060002: "D",  # Structure Set Label
    0x30060004: "X",  # Structure Set Name
    0x30060006: "X",  # Structure Set Description
    0x30060008: "Z",  # Structure Set Date
    0x30060009: "Z",  # Structure Set Time
    0x30060024: "U",  # Referenced Frame of Reference UID
    0x30060026: "Z",  # ROI Name
    0x30060028: "X",  # ROI Description
    0x3006002D: "X",  # ROI DateTime
    0x3006002E: "X",  # ROI Observation DateTime
    0x30060038: "X",  # ROI Generation Description
    0x3006004D: "X",  # ROI Creator Sequence
    0x3006004E: "X",  # ROI Interpreter Sequence
    0x30060085: "X",  # ROI Observation Label
    0x30060088: "X",  # ROI Observation Description
    0x300600A6: "Z",  # ROI Interpreter
    0x300600C2: "U",  # Related Frame of Reference UID
    0x30080024: "D",  # Treatment Control Point Date
    0x30080025: "D",  # Treatment Control Point Time
    0x30080054: "X/D",  # First Treatment Date
    0x30080056: "X/D",  # Most Recent Treatment Date
    0x30080105: "X/Z",  # Source Serial Number
    0x30080162: "D",  # Safe Position Exit Date
    0x30080164: "D",  # Safe Position Exit Time
    0x30080166: "D",  # Safe Position Return Date
    0x30080168: "D",  # Safe Position Return Time
    0x30080250: "X/D",  # Treatment Date
    0x30080251: "X/D",  # Treatment Time
    0x300A0002: "D",  # RT Plan Label
    0x300A0003: "X",  # RT Plan Name
    0x300A0004: "X",  # RT Plan Description
    0x300A0006: "X/D",  # RT Plan Date
    0x300A0007: "X/D",  # RT Plan Time
    0x300A000B: "X",  # Treatment Sites
    0x300A000E: "X",  # Prescription Description
    0x300A0013: "U",  # Dose Reference UID
    0x300A0016: "X",  # Dose Reference Description
    0x300A0054: "U",  # Table Top Position Alignment UID
    0x300A0072: "X",  # Fraction Group Description
    0x300A0083: "U",  # Referenced Dose Reference UID
    0x300A00B2: "X/Z",  # Treatment Machine Name
    0x300A00C3: "X",  # Beam Description
    0x300A00DD: "X",  # Bolus Description
    0x300A0196: "X",  # Fixation Device Description
    0x300A01A6: "X",  # Shielding Device Description
    0x300A01B2: "X",  # Setup Technique Description
    0x300A0216: "X",  # Source Manufacturer
    0x300A022C: "D",  # Source Strength Reference Date
    0x300A022E: "D",  # Source Strength Reference Time
    0x300A02EB: "X",  # Compensator Description
    0x300A0608: "D",  # Treatment Position Group Label
    0x300A0609: "U",  # Treatment Position Group UID
    0x300A0611: "Z",  # RT Accessory Holder Slot ID
    0x300A0615: "Z",  # RT Accessory Device Slot ID
    0x300A0619: "D",  # Radiation Dose Identification Label
    0x300A0623: "D",  # Radiation Dose In-Vivo Measurement Label
    0x300A062A: "D",  # RT Tolerance Set Label
    0x300A0650: "U",  # Patient Setup UID
    0x300A0676: "X",  # Equipment Frame of Reference Description
    0x300A067C: "D",  # Radiation Generation Mode Label
    0x300A067D: "Z",  # Radiation Generation Mode Description
    0x300A0700: "U",  # Treatment Session UID
    0x300A0734: "D",  # Treatment Tolerance Violation Description
    0x300A0736: "D",  # Treatment Tolerance Violation DateTime
    0x300A073A: "D",  # Recorded RT Control Point DateTime
    0x300A0741: "D",  # Interlock DateTime
    0x300A0742: "D",  # Interlock Description
    0x300A0760: "D",  # Override DateTime
    0x300A0783: "D",  # Interlock Origin Description
    0x300A0785: "U",  # Referenced Treatment Position Group UID
    0x300A078E: "X",  # Patient Treatment Preparation Procedure Parameter Description
    0x300A0792: "X",  # Patient Treatment Preparation Method Description
    0x300A0794: "X",  # Patient Setup Photo Description
    0x300A079A: "X",  # Displacement Reference Label
    0x300C0113: "X",  # Reason for Omission Description
    0x300C0127: "D",  # Beam Hold Transition DateTime
    0x300E0004: "Z",  # Review Date
    0x300E0005: "Z",  # Review Time
    0x300E0008: "X/Z",  # Reviewer Name
    0x30100006: "U",  # Conceptual Volume UID
    0x3010000B: "U",  # Referenced Conceptual Volume UID
    0x3010000F: "Z",  # Conceptual Volume Combination Description
    0x30100013: "U",  # Constituent Conceptual Volume UID
    0x30100015: "U",  # Source Conceptual Volume UID
    0x30100017: "Z",  # Conceptual Volume Description
    0x3010001B: "Z",  # Device Alternate Identifier
    0x3010002D: "D",  # Device Label
    0x30100031: "U",  # Referenced Fiducials UID
    0x30100033: "D",  # User Content Label
    0x30100034: "D",  # User Content Long Label
    0x30100035: "D",  # Entity Label
    0x30100036: "X",  # Entity Name
    0x30100037: "X",  # Entity Description
    0x30100038: "D",  # Entity Long Label
    0x3010003B: "U",  # RT Treatment Phase UID
    0x30100043: "Z",  # Manufacturer's Device Identifier
    0x3010004C: "X/D",  # Intended Phase Start Date
    0x3010004D: "X/D",  # Intended Phase End Date
    0x30100054: "D",  # RT Prescription Label
    0x30100056: "X/D",  # RT Treatment Approach Label
    0x3010005A: "Z",  # RT Physician Intent Narrative
    0x3010005C: "Z",  # Reason for Superseding
    0x30100061: "X",  # Prior Treatment Dose Description
    0x3010006E: "U",  # Dosimetric Objective UID
    0x3010006F: "U",  # Referenced Dosimetric Objective UID
    0x30100077: "X/D",  # Treatment Site
    0x3010007A: "Z",  # Treatment Technique Notes
    0x3010007B: "Z",  # Prescription Notes
    0x3010007F: "Z",  # Fractionation Notes
    0x30100081: "Z",  # Prescription Notes Sequence
    0x30100085: "X",  # Intended Fraction Start Time
    0x40000010: "X",  # Arbitrary
    0x40004000: "X",  # Text Comments
    0x40080040: "X",  # Results ID
    0x40080042: "X",  # Results ID Issuer
    0x40080100: "X",  # Interpretation Recorded Date
    0x40080101: "X",  # Interpretation Recorded Time
    0x40080102: "X",  # Interpretation Recorder
    0x40080108: "X",  # Interpretation Transcription Date
    0x40080109: "X",  # Interpretation Transcription Time
    0x4008010A: "X",  # Interpretation Transcriber
    0x4008010B: "X",  # Interpretation Text
    0x4008010C: "X",  # Interpretation Author
    0x40080111: "X",  # Interpretation Approver Sequence
    0x40080112: "X",  # Interpretation Approval Date
    0x40080113: "X",  # Interpretation Approval Time
    0x40080114: "X",  # Physician Approving Interpretation
    0x40080115: "X",  # Interpretation Diagnosis Description
    0x40080118: "X",  # Results Distribution List Sequence
    0x40080119: "X",  # Distribution Name
    0x4008011A: "X",  # Distribution Address
    0x40080200: "X",  # Interpretation ID
    0x40080202: "X",  # Interpretation ID Issuer
    0x40080300: "X",  # Impressions
    0x40084000: "X",  # Results Comments
    0xFFFAFFFA: "X",  # Digital Signatures Sequence
    0xFFFCFFFC: "X",  # Data Set Trailing Padding
}

# VRs that hold raw bytes, for which the only safe dummy is an empty payload.
_BINARY_VRS = frozenset({"OB", "OD", "OF", "OL", "OV", "OW", "UN"})

# Dummy replacements for the D action, chosen to be valid for their VR.
_DUMMY_VALUES = {
    "AS": "000Y",
    "DA": "19000101",
    "DT": "19000101000000.000000",
    "TM": "000000",
    "PN": "XXXX",
    "DS": "0",
    "IS": "0",
    "FL": 0.0,
    "FD": 0.0,
    "SL": 0,
    "SS": 0,
    "UL": 0,
    "US": 0,
    "SV": 0,
    "UV": 0,
}


def is_ps3_15_use_case(use_case: str | None) -> bool:
    """True if the use_case string selects the PS3.15 basic profile.

    Matching is case-insensitive and tolerant of separator spelling, so
    "PS3.15", "ps3.15", "PS3_15" and "PS3-15" all select it.
    """
    if use_case is None:
        return False
    return use_case.strip().lower().replace("_", ".").replace("-", ".") == "ps3.15"


def _resolve_action(action: str) -> str:
    """Resolves a combined Table E.1-1 action (e.g. "X/Z/D") to a single one.

    The combined actions depend on whether the attribute is type 1/2 in the
    IOD of the object being de-identified, which is not tracked here. The
    value-preserving option is preferred (U > D > Z > X) so the resulting
    file stays valid regardless of the IOD.
    """
    if "U" in action:
        return "U"
    if "D" in action:
        return "D"
    if "Z" in action:
        return "Z"
    return "X"


def _replace_uid(original: str) -> str:
    """Maps a UID to a replacement UID, deterministically.

    Hashing the original UID (entropy_srcs) means the same input always maps
    to the same output, so Study/Series/Frame-of-Reference UIDs stay
    consistent across the files of a series without keeping any state, and
    referential integrity between objects is preserved.
    """
    return generate_uid(entropy_srcs=[original])


# String-based VRs, for which a zero-length value is the empty string. Using
# "" rather than None matters for PN: PersonName(None) prints as 'None'.
_STRING_VRS = frozenset({
    "AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT", "PN",
    "SH", "ST", "TM", "UC", "UI", "UR", "UT",
})


def _empty(elem: dicom.dataelem.DataElement) -> None:
    if elem.VR == "SQ":
        elem.value = []
    elif elem.VR in _STRING_VRS:
        elem.value = ""
    else:
        elem.value = None


def _dummify(elem: dicom.dataelem.DataElement) -> None:
    if elem.VR == "SQ":
        elem.value = []
    elif elem.VR == "UI":
        _replace_uids(elem)
    elif elem.VR in _BINARY_VRS:
        elem.value = b""
    else:
        elem.value = _DUMMY_VALUES.get(elem.VR, "XXXX")


def _replace_uids(elem: dicom.dataelem.DataElement) -> None:
    value = elem.value
    if value in (None, ""):
        return
    if isinstance(value, (list, dicom.multival.MultiValue)):
        elem.value = [_replace_uid(str(v)) for v in value]
    else:
        elem.value = _replace_uid(str(value))


def _action_for(tag: dicom.tag.Tag) -> str | None:
    action = BASIC_PROFILE_ACTIONS.get(int(tag))
    if action is not None:
        return action
    if tag.is_private:
        return "X"
    if (tag.group & 0xFF00) == 0x5000:  # Curve Data (50xx,xxxx)
        return "X"
    if (tag.group & 0xFF00) == 0x6000 and tag.element in (0x3000, 0x4000):
        return "X"  # Overlay Data / Overlay Comments
    return None


def _apply(ds: dicom.dataset.Dataset, anonymised_headers: list) -> None:
    for elem in list(ds):
        action = _action_for(elem.tag)
        if action is None:
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, dicom.dataset.Dataset):
                        _apply(item, anonymised_headers)
            continue
        record = {"tag": str(elem.tag), "name": elem.name}
        resolved = _resolve_action(action)
        try:
            if resolved == "X":
                del ds[elem.tag]
            elif resolved == "Z":
                _empty(elem)
            elif resolved == "U":
                if elem.VR == "SQ":
                    # U* rows (e.g. Source Image Sequence): the sequence is
                    # kept and the UIDs within its items are replaced, so
                    # recurse into the items instead of rewriting the SQ.
                    for item in elem.value:
                        if isinstance(item, dicom.dataset.Dataset):
                            _apply(item, anonymised_headers)
                else:
                    _replace_uids(elem)
            else:  # D
                _dummify(elem)
        except Exception as e:
            # Fail closed: an attribute that could not be processed may still
            # contain PHI, so remove it rather than leave the original.
            logger.error(
                "Failed to apply PS3.15 action %s to %s (%s), removing it. %s: %s",
                resolved, elem.tag, elem.name, type(e).__name__, e,
            )
            del ds[elem.tag]
        anonymised_headers.append(record)


def apply_basic_profile(ds: dicom.dataset.Dataset,
                        anonymised_headers: list | None = None) -> dicom.dataset.Dataset:
    """De-identifies DICOM headers in-place per PS3.15 Annex E Basic Profile.

    Applies the Basic Application Level Confidentiality Profile actions from
    Table E.1-1 to every element (recursing into sequences), removes private
    attributes, curve and overlay data, and marks the dataset as
    de-identified via Patient Identity Removed (0012,0062) and
    De-identification Method (0012,0063).

    Pixel data is not touched; burned-in PHI must be handled separately
    (e.g. with destroy_pixels or an image redactor).

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The DICOM dataset to de-identify in-place.

    anonymised_headers : list, optional
        If given, a record {"tag", "name"} is appended for every element an
        action was applied to.

    Returns
    -------
    pydicom.dataset.Dataset
        The same dataset, de-identified.
    """
    if anonymised_headers is None:
        anonymised_headers = []
    _apply(ds, anonymised_headers)
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "PS3.15 E.1 Basic Application Level Confidentiality Profile"
    # Group 0002 lives in file_meta, outside the main dataset walk; keep the
    # Media Storage SOP Instance UID consistent with the replaced SOP Instance UID.
    if getattr(ds, "file_meta", None) is not None and "SOPInstanceUID" in ds:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds
