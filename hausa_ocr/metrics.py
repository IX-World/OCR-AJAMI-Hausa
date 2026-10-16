import unicodedata


UWER_DEFINITION = "Unicode casefold, NFD, suppression des marques Mn, puis NFC et separation des mots par les espaces ; ponctuation et lettres haoussa non decomposees conservees."


def normalize_uwer(text):
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if unicodedata.category(c) != "Mn"))
