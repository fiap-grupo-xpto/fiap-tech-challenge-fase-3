from backend.rehab_video.dataset import (
    RepetitionRecord,
    load_segmentation,
    split_by_person,
    video_frame_range,
)


def test_load_segmentation_reads_semicolon_delimited_annotations(tmp_path):
    source = tmp_path / "Segmentation.csv"
    source.write_text(
        "video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;correctness\n"
        "PM_000;1;1;1;12;42;1\n"
        "PM_001;2;1;2;43;72;0\n",
        encoding="utf-8",
    )

    records = load_segmentation(source)

    assert len(records) == 2
    assert records[0].video_id == "PM_000"
    assert records[0].first_frame == 12
    assert records[0].last_frame == 42
    assert records[0].correctness == 1
    assert records[1].correctness == 0


def test_load_segmentation_rejects_invalid_frame_ranges(tmp_path):
    source = tmp_path / "Segmentation.csv"
    source.write_text(
        "video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;correctness\n"
        "PM_000;1;1;1;42;12;1\n",
        encoding="utf-8",
    )

    try:
        load_segmentation(source)
    except ValueError as exc:
        assert "frame" in str(exc).lower()
    else:
        raise AssertionError("Invalid frame range should be rejected.")


def test_video_frame_range_converts_one_based_inclusive_annotation():
    record = RepetitionRecord(
        video_id="PM_000",
        repetition_number=1,
        exercise_id=1,
        person_id=1,
        first_frame=1,
        last_frame=30,
        correctness=1,
    )

    assert video_frame_range(record) == (0, 29)


def test_video_frame_range_clamps_single_trailing_annotation_frame():
    record = RepetitionRecord(
        video_id="PM_117a",
        repetition_number=9,
        exercise_id=5,
        person_id=9,
        first_frame=1110,
        last_frame=1185,
        correctness=0,
    )

    assert video_frame_range(record, frame_count=1184) == (1109, 1183)


def test_split_by_person_keeps_participants_disjoint():
    records = [
        RepetitionRecord(
            video_id=f"PM_{person:03}",
            repetition_number=repetition,
            exercise_id=1,
            person_id=person,
            first_frame=0,
            last_frame=30,
            correctness=(repetition % 2),
        )
        for person in range(1, 11)
        for repetition in range(1, 5)
    ]

    split = split_by_person(records, random_state=7)

    train_people = {records[index].person_id for index in split.train}
    validation_people = {records[index].person_id for index in split.validation}
    test_people = {records[index].person_id for index in split.test}

    assert train_people.isdisjoint(validation_people)
    assert train_people.isdisjoint(test_people)
    assert validation_people.isdisjoint(test_people)
    assert len(train_people) == 6
    assert len(validation_people) == 2
    assert len(test_people) == 2
    assert sorted(split.train + split.validation + split.test) == list(range(len(records)))
