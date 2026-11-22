"""Anonymity audit (report Table 15). Needs only simple stand-in objects, not pydicom."""

from types import SimpleNamespace

import pytest

from rad.dicom_io import AnonymityError, audit_identity, normalize_name

PATTERN = r"^(P\d{3}|LUNG1-\d{3})$"
TAGS = ("PatientBirthDate", "PatientAddress")


def header(**kw):
    base = dict(PatientID="P001", PatientName="", PatientBirthDate="", PatientAddress="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_pseudonymous_header_passes():
    assert audit_identity(header(), PATTERN, TAGS) == "P001"
    assert audit_identity(header(PatientID="LUNG1-017", PatientName="LUNG1-017"), PATTERN, TAGS) == "LUNG1-017"


@pytest.mark.parametrize(
    "bad",
    [
        dict(PatientID="12345678901"),
        dict(PatientName="DOE^JOHN"),
        dict(PatientBirthDate="19600101"),
        dict(PatientAddress="somewhere"),
    ],
)
def test_identity_fields_stop_processing(bad):
    with pytest.raises(AnonymityError):
        audit_identity(header(**bad), PATTERN, TAGS)


def test_roi_name_matching_is_exact_after_normalisation():
    assert normalize_name("GTV-1") == "gtv1"
    assert normalize_name("Lung-GTV") != normalize_name("GTV")
