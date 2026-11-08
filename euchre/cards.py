"""Card representation and trump-dependent lookup tables for Euchre.

A card is an int 0..23:  card = suit * 6 + rank
  suits: 0=S, 1=C (black), 2=H, 3=D (red)  -> same-colour suit is  s ^ 1
  ranks: 0=9, 1=T, 2=J, 3=Q, 4=K, 5=A
"""

SUITS = "SCHD"
SUIT_NAMES = ["Spades", "Clubs", "Hearts", "Diamonds"]
RANKS = "9TJQKA"
NUM_CARDS = 24
JACK, ACE = 2, 5

ALL_CARDS = tuple(range(NUM_CARDS))


def suit_of(c):
    return c // 6


def rank_of(c):
    return c % 6


def make_card(rank, suit):
    return suit * 6 + rank


def card_str(c):
    return RANKS[c % 6] + SUITS[c // 6]


def cards_str(cards):
    return " ".join(card_str(c) for c in cards)


def parse_card(s):
    s = s.strip().upper().replace("10", "T")
    if len(s) != 2 or s[0] not in RANKS or s[1] not in SUITS:
        raise ValueError(f"bad card {s!r}; use e.g. JH, TS, 9D, AC")
    return make_card(RANKS.index(s[0]), SUITS.index(s[1]))


def parse_cards(s):
    return [parse_card(x) for x in s.replace(",", " ").split()]


def parse_suit(s):
    s = s.strip().upper()[0]
    if s not in SUITS:
        raise ValueError(f"bad suit {s!r}")
    return SUITS.index(s)


# --- trump tables -------------------------------------------------------
# EFF[t][c]          effective suit of card c when t is trump (left bower -> t)
# TRUMP_POWER[t][c]  13 right bower, 12 left bower, 6+rank other trumps, -1 non-trump
# CANON[t][c]        card with suits relabelled so trump is suit 0, same colour suit 1
#                    (an involution: CANON[t][CANON[t][c]] == c)
EFF = [[0] * NUM_CARDS for _ in range(4)]
TRUMP_POWER = [[-1] * NUM_CARDS for _ in range(4)]
CANON = [[0] * NUM_CARDS for _ in range(4)]

for _t in range(4):
    for _c in range(NUM_CARDS):
        _s, _r = suit_of(_c), rank_of(_c)
        left = _r == JACK and _s == (_t ^ 1)
        EFF[_t][_c] = _t if left else _s
        if _r == JACK and _s == _t:
            TRUMP_POWER[_t][_c] = 13
        elif left:
            TRUMP_POWER[_t][_c] = 12
        elif _s == _t:
            TRUMP_POWER[_t][_c] = 6 + _r
        CANON[_t][_c] = (_s ^ _t) * 6 + _r


def is_trump(c, trump):
    return TRUMP_POWER[trump][c] >= 0


def power(c, trump, led):
    """Strength of card c in a trick whose led (effective) suit is `led`. -1 = cannot win."""
    tp = TRUMP_POWER[trump][c]
    if tp >= 0:
        return tp
    return c % 6 if EFF[trump][c] == led else -1
