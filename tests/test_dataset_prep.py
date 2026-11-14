from pathlib import Path

from dataset_prep import build_class_id_map, names_to_list, remap_label_file


def test_names_to_list_accepts_list_and_mapping_forms():
    assert names_to_list(["drone", "bird"]) == ["drone", "bird"]
    assert names_to_list({0: "drone", 1: "bird"}) == ["drone", "bird"]
    assert names_to_list({1: "bird", 0: "drone"}) == ["drone", "bird"]


def test_build_class_id_map_finds_case_insensitive_match():
    mapping = build_class_id_map(["Drone", "bird", "plane"], wanted_name="drone", combined_id=1)
    assert mapping == {0: 1}


def test_build_class_id_map_empty_when_name_absent():
    mapping = build_class_id_map(["bird", "plane"], wanted_name="drone", combined_id=1)
    assert mapping == {}


def test_remap_label_file_drops_unwanted_classes_and_renumbers_kept_ones(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n1 0.9 0.9 0.1 0.1\n")
    dst = tmp_path / "dst.txt"

    # keep original class 1 -> combined class 7; drop classes 0 and 2
    malformed = remap_label_file(src, dst, class_id_map={1: 7})

    lines = dst.read_text().strip().splitlines()
    assert lines == ["7 0.900000 0.900000 0.100000 0.100000"]
    assert malformed == 0


def test_remap_label_file_converts_polygon_rows_to_their_bounding_box(tmp_path: Path):
    src = tmp_path / "src.txt"
    # polygon with x in [0.2, 0.6] and y in [0.1, 0.5]
    src.write_text("0 0.2 0.1 0.6 0.1 0.6 0.5 0.2 0.5\n")
    dst = tmp_path / "dst.txt"

    malformed = remap_label_file(src, dst, class_id_map={0: 1})

    assert dst.read_text().strip() == "1 0.400000 0.300000 0.400000 0.400000"
    assert malformed == 0


def test_remap_label_file_drops_and_counts_malformed_or_degenerate_rows(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text(
        "0 0.5 0.5\n"            # too few values
        "0 0.5 0.5 0.0 0.2\n"    # zero width
        "0 0.5 0.5 0.2 0.2\n"    # valid
    )
    dst = tmp_path / "dst.txt"

    malformed = remap_label_file(src, dst, class_id_map={0: 0})

    assert dst.read_text().strip() == "0 0.500000 0.500000 0.200000 0.200000"
    assert malformed == 2


def test_normalize_box_clips_boxes_that_spill_past_the_image_edge():
    from dataset_prep import normalize_box
    cx, cy, w, h = normalize_box([0.95, 0.5, 0.2, 0.2])  # spans x 0.85..1.05
    assert (round(cx, 6), round(w, 6)) == (0.925, 0.15)


def test_remap_label_file_writes_empty_file_when_nothing_kept(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("0 0.5 0.5 0.2 0.2\n")
    dst = tmp_path / "dst.txt"

    remap_label_file(src, dst, class_id_map={})

    assert dst.read_text() == ""
