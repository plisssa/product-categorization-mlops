import pandas as pd

from ecom_category_project.features.text import TextBuildConfig, build_text_from_row


def test_build_text_includes_domain_and_markers():
    row = pd.Series(
        {
            "vendor": "BrandX",
            "name": "Test Product",
            "model": "A-1",
            "type_prefix": "headphones",
            "description": "Short description",
            "url": "https://www.example.com/p/1",
        }
    )
    cfg = TextBuildConfig(
        text_fields=["vendor", "name", "model", "type_prefix", "description"],
        include_domain=True,
    )
    text = build_text_from_row(row, cfg)
    assert "vendor: BrandX" in text
    assert "name: Test Product" in text
    assert "domain: example.com" in text
