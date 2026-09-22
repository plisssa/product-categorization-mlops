from ecom_category_project.features.text import extract_domain, extract_url_tokens


def test_extract_domain_strips_www():
    assert extract_domain("https://www.ormatek.com/catalog/x") == "ormatek.com"
    assert extract_domain("") == ""


def test_url_tokens_skip_numbers_and_short():
    tokens = extract_url_tokens("https://shop.ru/catalog/krovati/komo-veda-4/140-190", limit=12)
    assert "krovati" in tokens
    assert "140" not in tokens.split()  # чистые числа отбрасываются
