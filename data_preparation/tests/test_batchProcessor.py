from data_preparation.xml_to_json_batch import BatchProcessor

def test_batch_processor_multiple_files(tmp_path):
    input_dir = tmp_path / "xml"
    output_dir = tmp_path / "json"
    input_dir.mkdir()

    for i in range(3):
        (input_dir / f"a{i}.xml").write_text("<article><id>A</id></article>")

    processor = BatchProcessor(str(input_dir), str(output_dir))
    processor.process_all()

    assert len(list(output_dir.glob("*.json"))) == 3


def test_output_directory_created(tmp_path):
    input_dir = tmp_path / "xml"
    input_dir.mkdir()
    (input_dir / "a.xml").write_text("<article />")

    output_dir = tmp_path / "json"

    BatchProcessor(str(input_dir), str(output_dir))
    assert output_dir.exists()
