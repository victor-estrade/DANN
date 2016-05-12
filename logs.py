#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import logging
import os
import time
import re

__author__ = 'Estrade Victor'

def empty_logger():
    """
    Empty the logger
    """
    logger = logging.getLogger('empty')
    logger.handlers = []
    return logger

def new_logger(fname=None, name=None):
    """
    Build and return a new logger that will write its logs into the console
    and into the optional given file.
    
    Params
    ------
        fname(default=None): the optionnal file log. 
            (See log_fname for automatic file naming)
        name(default=None): the optionnal name of this logger. 
            Usefull to separate code area using serveral loggers
    Return
    ------
        logger: the logger
    
    Example
    -------
    >>>logger = new_logger('myLog.log')
    >>>logger.debug('debug message')
    >>>logger.info('info message')
    >>>logger.warn('warn message')
    >>>logger.error('error message')
    >>>logger.critical('critical message')
    """
    # create logger
    logger = logging.getLogger('log')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    # create formatter
    if name is None:
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s: %(message)s',
            '[%H:%M:%S]')
    else:
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s: %(message)s',
            '[%H:%M:%S]')
    # add formatter to console_log
    console_log.setFormatter(formatter)
    # add console_log to logger
    logger.addHandler(console_log)

    if fname:
        # create file handler and set level to debug
        f_log = logging.FileHandler(fname, mode='a')
        f_log.setLevel(logging.DEBUG)
        # create formatter
        if name is None:
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s',
                '[%m-%d-%Y||%H:%M:%S]')
        else:
            formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s: %(message)s',
                '[%m-%d-%Y||%H:%M:%S]')
        # add formatter to f_log
        f_log.setFormatter(formatter)
        # add f_log to logger
        logger.addHandler(f_log)
    logger.propagate = False

    return logger


def log_fname(name=''):
    log = 'log'
    if not os.path.isdir(log):
        os.mkdir(log)
    clock = time.strftime('%Hh%Mm%S')
    day = time.strftime('%m-%d-%Y')

    path = os.path.join(log, day)
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(log, day, name+clock+'.log')
    return path


def extract(log, name, sep=r'\s*'):
    re_float = r"""
                 [-+]? # optional sign
                 (?:
                     (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
                     |
                     (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
                 )
                 # followed by optional exponent part if desired
                 (?: [Ee] [+-]? \d+ ) ?
                 """
    regex = r'{}{}({})\s?%?'.format(name, sep, re_float)
    rx = re.compile(regex, re.VERBOSE)
    values = []
    with open(log, 'r') as file:
        lines = file.readlines()
    for line in lines:
        m = rx.search(line)
        if m:
            values.append(float(m.group(1)))
    return values


def multi_extract(log, names, seps=None):
    re_float = r"""
                 [-+]? # optional sign
                 (?:
                     (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
                     |
                     (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
                 )
                 # followed by optional exponent part if desired
                 (?: [Ee] [+-]? \d+ ) ?
                 """
    if seps is None:
        seps = [r'\s*' for name in names]
    rx = {}
    for name, sep in zip(names, seps):
        regex = r'{}{}({})\s?%?'.format(name, sep, re_float)
        rx[name] = re.compile(regex, re.VERBOSE)
    values = {name: [] for name in names}
    with open(log, 'r') as file:
        lines = file.readlines()
    for line in lines:
        for name in names:
            m = rx[name].search(line)
            if m:
                values[name].append(float(m.group(1)))
    return values


if __name__ == '__main__':
    # print('I am at your service Master')
    f_log = log_fname('papa')
    print(f_log)
    logger  = new_logger(f_log)
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')
