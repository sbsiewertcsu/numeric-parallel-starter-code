print ("Eratoshenes Sieve")

def primes():
    yield 2; yield 3; yield 5; yield 7;       # define the base primes
    bps = (p for p in primes())               # base primes
    p = next(bps) and next(bps)               # discard 2, get 3
    q = p * p                                 # start sieve at offsets of prime from square of prime
    sieve = {}                                # empty dictionary to start
    n = 9                                     # start with first candidate
    while True:
        if n not in sieve:                    # n is not a multiple of any base prime, so not found in dictionary
            if n < q:                         # below next base prime's square, so
                yield n                       # n is prime
            else:
                p2 = p + p                    # next composite is offset of prime
                sieve[q + p2] = p2            # which is added to dictonary with 2 * p as the increment step
                p = next(bps); q = p * p      # pull next base prime, and get its square
        else:
            s = sieve.pop(n); nnext = n + s   # n's a multiple of base prime, find next multiple
            while nnext in sieve: nnext += s  # and ensure each entry is unique
            sieve[nnext] = s                  # nnext is next non-marked multiple of this prime
        n += 2                                # work on odds only to make more efficient

import itertools
def primes_up_to(limit):
    return list(itertools.takewhile(lambda p: p <= limit, primes()))

print(list(primes_up_to(120)))