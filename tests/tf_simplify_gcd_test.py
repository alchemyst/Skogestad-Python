import utils


def test_gcd():
    G1 = utils.tf([1], [1, 1])

    G2 = utils.tf([1], [2, 1])

    G3 = G1 * G2

    G4 = G3 * G1

    G5 = G4 / G1

    assert (G5.numerator == G3.numerator) and (G5.denominator == G3.denominator)


if __name__ == "__main__":
    test_gcd()
