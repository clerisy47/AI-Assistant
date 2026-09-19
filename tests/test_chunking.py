from app.rag.chunking import chunk_text, split_into_sentences


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_a_single_chunk():
    text = "This is one short sentence. Here is another."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("This is one short sentence.")


def test_long_text_is_split_and_respects_chunk_size():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 40  # ~1880 chars, well past a small chunk_size
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200


def test_chunks_carry_overlap_from_previous_tail():
    sentence = "Sentence number {}. "
    text = "".join(sentence.format(i) for i in range(30))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    # The start of chunk N should share some text with the tail of chunk N-1.
    tail_of_first = chunks[0].text[-15:]
    assert any(tail_of_first[:8] in chunks[i].text for i in range(1, len(chunks)))


def test_oversized_single_sentence_is_hard_split():
    huge_sentence = "word " * 500 + "."
    chunks = chunk_text(huge_sentence, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300


def test_metadata_is_attached_to_every_chunk():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_text(text, chunk_size=15, chunk_overlap=5, metadata={"source": "doc.md"})
    assert len(chunks) > 1
    assert all(c.metadata == {"source": "doc.md"} for c in chunks)


def test_split_into_sentences_basic():
    sentences = split_into_sentences("Hello world. How are you? Fine!")
    assert sentences == ["Hello world.", "How are you?", "Fine!"]


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=100)
