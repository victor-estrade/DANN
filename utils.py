#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function


def reverse_lines(fname):
    """
    Reverse the lines order in a file

    Params
    ------
        fname: the file name to be reversed
    """
    with open(fname, 'r') as file:
        lines = file.readlines()
    with open(fname, 'w') as file:
        file.write(''.join(reversed(lines)))



def pop_last_line(fname):
    """
    Get and remove the last line from the given file

    Params
    ------
        fname: the file name from which to extract the last line
    Return
    ------
        line: le last line of the given file
    """
    pos = []
    with open(fname, 'rw+') as file:
        pos.append(file.tell())
        line = file.readline()
        while line:
            pos.append(file.tell())
            line = file.readline()
        if len(pos) > 1:
            file.seek(pos[-2])
            line = file.readline()
            file.truncate(pos[-2])
    return line

if __name__ == '__main__':
    print('I am at your service Master')
    reverse_lines('TODO.txt')
    print(pop_last_line('TODO.txt'))