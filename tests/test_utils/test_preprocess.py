from text_dedup.utils.preprocess import news_copy_preprocessing
from text_dedup.utils.preprocess import normalize


class TestNormalize:
    def test_basic_normalization(self) -> None:
        assert normalize("Hello, world!") == "hello world"

    def test_digits_replaced_with_zero(self) -> None:
        assert normalize("Hello, 123!") == "hello 000"

    def test_whitespace_stripped(self) -> None:
        assert normalize("  Hello, world!  ") == "hello world"
        assert normalize("\tHello, world!\n") == "hello world"

    def test_punctuation_removed(self) -> None:
        assert normalize("Hello, world!!!") == "hello world"
        assert normalize("test@example.com") == "testexamplecom"
        assert normalize("foo-bar_baz") == "foobarbaz"

    def test_multiple_digits(self) -> None:
        assert normalize("Test 123 and 456") == "test 000 and 000"
        assert normalize("2023-01-15") == "00000000"

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_whitespace_only(self) -> None:
        assert normalize("   ") == ""
        assert normalize("\n\t  ") == ""

    def test_lowercase_conversion(self) -> None:
        assert normalize("HELLO WORLD") == "hello world"
        assert normalize("HeLLo WoRLd") == "hello world"

    def test_special_characters(self) -> None:
        assert normalize("Hello™ World®") == "hello world"
        assert normalize("Test © 2023") == "test  0000"

    def test_unicode_punctuation(self) -> None:
        assert normalize("Hello\u2019s world") == "hellos world"
        assert normalize("Test\u2014dash") == "testdash"

    def test_mixed_content(self) -> None:
        result = normalize("Hello, 123!\n\t\b")
        assert result == "hello 000"

    def test_only_punctuation(self) -> None:
        assert normalize("!!!???...") == ""
        assert normalize("@#$%^&*()") == ""

    def test_only_digits(self) -> None:
        assert normalize("123456") == "000000"

    def test_control_characters(self) -> None:
        assert normalize("Hello\x00World\x01") == "helloworld"
        assert normalize("Test\b\n\r\t") == "test"

    def test_preserves_spaces_between_words(self) -> None:
        assert normalize("one two three") == "one two three"
        assert normalize("a  b    c") == "a  b    c"

    def test_combined_operations(self) -> None:
        assert normalize("  HELLO, 123 WORLD!  ") == "hello 000 world"
        assert normalize("Test@123.com") == "test000com"

    def test_non_printing_chars(self) -> None:
        assert normalize("Hello\u200bWorld") == "helloworld"


class TestNewsCopyPreprocessing:
    def test_basic_text(self) -> None:
        result = news_copy_preprocessing("Hello world")
        assert result == "Hello world"

    def test_remove_specified_chars(self) -> None:
        result = news_copy_preprocessing("Test\"#$%&()*+/:;<=>@[\\]^_`{|}~.?,!' text")
        assert result == "Test text"

    def test_hyphenated_line_breaks(self) -> None:
        result = news_copy_preprocessing("Test-\nword")
        assert result == "Testword"

    def test_newlines_to_spaces(self) -> None:
        result = news_copy_preprocessing("Line1\nLine2\nLine3")
        assert result == "Line1 Line2 Line3"

    def test_combined_newline_handling(self) -> None:
        result = news_copy_preprocessing("Test-\nword and\nnormal newline")
        assert result == "Testword and normal newline"

    def test_ascii_encoding(self) -> None:
        result = news_copy_preprocessing("Test café résumé")
        assert result == "Test caf rsum"

    def test_non_ascii_removed(self) -> None:
        result = news_copy_preprocessing("Hello 世界")
        assert result == "Hello "

    def test_empty_string(self) -> None:
        result = news_copy_preprocessing("")
        assert result == ""

    def test_only_special_chars(self) -> None:
        result = news_copy_preprocessing('"#$%&()*+')
        assert result == ""

    def test_quotes_removed(self) -> None:
        result = news_copy_preprocessing("\"Hello\" and 'world'")
        assert result == "Hello and world"

    def test_brackets_removed(self) -> None:
        result = news_copy_preprocessing("Test [foo] (bar) {baz}")
        assert result == "Test foo bar baz"

    def test_punctuation_removed(self) -> None:
        result = news_copy_preprocessing("Hello, world! How are you?")
        assert result == "Hello world How are you"

    def test_math_symbols_removed(self) -> None:
        result = news_copy_preprocessing("x = y + z")
        assert result == "x  y  z"

    def test_preserves_hyphens_within_lines(self) -> None:
        result = news_copy_preprocessing("test-case-example")
        assert result == "test-case-example"

    def test_multiple_newlines(self) -> None:
        result = news_copy_preprocessing("Line1\n\n\nLine2")
        assert result == "Line1   Line2"

    def test_mixed_content(self) -> None:
        text = "Hello, world!\nThis is a test-\ncase with 'quotes' and (parentheses)."
        result = news_copy_preprocessing(text)
        assert result == "Hello world This is a testcase with quotes and parentheses"

    def test_preserves_numbers(self) -> None:
        result = news_copy_preprocessing("Test 123 numbers")
        assert result == "Test 123 numbers"

    def test_preserves_basic_alphanumeric(self) -> None:
        result = news_copy_preprocessing("abc123XYZ")
        assert result == "abc123XYZ"

    def test_url_like_text(self) -> None:
        result = news_copy_preprocessing("https://example.com/path?query=value")
        assert result == "httpsexamplecompathqueryvalue"

    def test_email_like_text(self) -> None:
        result = news_copy_preprocessing("test@example.com")
        assert result == "testexamplecom"
