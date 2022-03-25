import os
import random
import statistics
import string
import sys
import timeit

import numpy as np


def letter():
    """Genera un caracter ascii aleatorio"""
    return random.choice(string.ascii_letters)


def word():
    """Genera una palabra ascii aleatoria de entre 8 y 23 caracteres"""
    length = random.randint(8, 24)
    return ''.join((letter() for _ in range(length)))


def number():
    """Genera un entero de python aleatorio entre 1 y 1024"""
    return random.randint(1, 1024)


def numpy():
    """Genera un entero de numpy aleatorio entre 1 y 1024"""
    return np.intc(random.randint(1, 1024))


def bytes():
    """Genera una cadena de 24 bits aleatoria"""
    return os.urandom(1)


def compare(comparison: str, function: str):
    letter_eq = timeit.repeat(stmt=f'a {comparison} b',
                              setup=f'a={function}();b={function}()',
                              number=100_000,
                              repeat=10_000,
                              globals=globals())
    return statistics.mean(letter_eq)


def size(function: str):
    print(eval(function + '()'))
    return statistics.mean(
        (sys.getsizeof(eval(function + '()')) for _ in range(1000)))


if __name__ == '__main__':

    print(
        "Tiempo medio para realizar 100_000 operaciones. Experimento repetido "
        + "10_000 veces.")
    for func in ['letter', 'word', 'number', 'numpy', 'bytes']:
        print(f'{func} equality: media de ', compare('==', func),
              'segundos para realizar 100_000 comparaciones')

        print(f'{func} unequality: media de', compare('!=', func),
              'segundos para realizar 100_000 comparaciones')

        print(f'{func} size: {size(func):.0f} bytes')
