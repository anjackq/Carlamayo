from module.proposal_audit import coc_audit_fields


def test_coc_audit_fields_preserve_multiline_unicode_exactly():
    text = "Pedestrian approaching.\n减速并停车 🚶"

    fields = coc_audit_fields(text)

    assert fields["coc_text_full"] == text
    assert fields["coc_character_count"] == len(text)
    assert fields["coc_utf8_byte_count"] == len(text.encode("utf-8"))
    assert fields["coc_sha256"] == (
        "87197f156e45a1a0ccf831ca755d4e9167e7940a85f8bdb8d20a5e7d0f38dc00"
    )
    assert fields["coc_truncated"] is False


def test_coc_audit_fields_normalize_none_to_an_auditable_empty_string():
    fields = coc_audit_fields(None)

    assert fields["coc_text_full"] == ""
    assert fields["coc_character_count"] == 0
    assert fields["coc_utf8_byte_count"] == 0
    assert fields["coc_sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
