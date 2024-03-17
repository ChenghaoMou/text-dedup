def news_copy_preprocessing(text: str) -> str:
    chars_to_remove = r'"#$%&\()*+/:;<=>@[\\]^_`{|}~.?,!\''
    text = text.replace("-\n", "").replace("\n", " ")
    text = text.translate(str.maketrans("", "", chars_to_remove))
    text = text.encode("ascii", "ignore").decode()
    return text
