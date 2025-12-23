# src/ibsen/cli/deconstruct.py
from __future__ import annotations
import argparse
import random
import re
import sys
import time

YES_RE = re.compile(r"^(yes)[\W_]*$", re.IGNORECASE)


def tokenize_words(text: str) -> list[str]:
    # Keep punctuation attached to words (simple, terminal-friendly)
    return re.findall(r"\S+", text)


def stream_words(
    words: list[str],
    *,
    wpm: int = 220,
    jitter: float = 0.15,
    seed: int | None = None,
    yes_newline: bool = True,
):
    if seed is not None:
        random.seed(seed)

    # seconds per word
    base = 60.0 / max(1, wpm)

    for w in words:
        if YES_RE.match(w):
            out = "YES"
            if yes_newline:
                sys.stdout.write("\n" + out + " \n")
            else:
                sys.stdout.write(out + " ")
        else:
            sys.stdout.write(w + " ")

        sys.stdout.flush()

        # jitter in delay feels more human
        dt = base * (1.0 + random.uniform(-jitter, jitter))
        time.sleep(max(0.0, dt))

    sys.stdout.write("\n")
    sys.stdout.flush()


def load_text() -> str:
    return (
        """O that awful deepdown torrent O and the sea the sea crimson
        sometimes like fire and the glorious sunsets and the figtrees in the
        Alameda gardens yes and all the queer little streets and pink and blue 
        and yellow houses and the rosegardens and the jessamine and geraniums 
        and cactuses and Gibraltar as a girl where 
        I was a Flower of the mountain yes when I put the rose in my hair like
        the Andalusian girls used or shall I wear a red yes and how he kissed 
        me under the Moorish wall and I thought well as well him as another 
        and then I asked him with my eyes to ask again yes and then he asked 
        me would I yes to say yes my mountain flower and first I put my arms 
        around him yes and drew him down to me so he could feel my breasts all 
        perfume yes and his heart was going like mad and yes I said yes I will 
        Yes.
        """
    )


def main(argv: list[str] | None = None) -> int:
    # ap = argparse.ArgumentParser(prog="ibsen-deconstruct")
    text = load_text()
    words = tokenize_words(text)
    stream_words(words)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

