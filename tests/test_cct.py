import pandas as pd
import pytest

from src.data.cct import split_by_location


def _images_df():
    return pd.DataFrame(
        {"location": [1, 1, 2, 3, 3, 3]},
        index=["img_a", "img_b", "img_c", "img_d", "img_e", "img_f"],
    )


def test_split_by_location_assigns_correct_images():
    train_df, val_df, test_df = split_by_location(
        _images_df(), train_locations={1, 2}, val_locations={3}, test_locations=set()
    )
    assert set(train_df.index) == {"img_a", "img_b", "img_c"}
    assert set(val_df.index) == {"img_d", "img_e", "img_f"}
    assert len(test_df) == 0


def test_split_by_location_raises_on_overlapping_locations():
    with pytest.raises(ValueError):
        split_by_location(_images_df(), train_locations={1, 2}, val_locations={2, 3}, test_locations=set())


def test_split_by_location_raises_on_unassigned_location():
    # location 2 isn't in any of the three sets
    with pytest.raises(ValueError):
        split_by_location(_images_df(), train_locations={1}, val_locations={3}, test_locations=set())
