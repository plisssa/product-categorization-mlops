import pandas as pd


def test_submission_is_one_based(tmp_path):
    submission = pd.DataFrame({"ID": range(1, 4), "category_ind": [4, 5, 6]})
    path = tmp_path / "submission.csv"
    submission.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert loaded["ID"].tolist() == [1, 2, 3]
