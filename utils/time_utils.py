#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time

DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
DATE_ADJOIN_FORMAT = "%Y%m%d"
DATE_TIME_ADJOIN_FORMAT = "%Y%m%d_%Hh%Mm%Ss"


def time_stamp_to_date_time(time_stamp):
    time_array = time.localtime(time_stamp)
    otherStyleTime = time.strftime(DATE_TIME_FORMAT, time_array)
    return otherStyleTime


def time_stamp_to_date(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_FORMAT, time_array)


def time_stamp_to_date_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_ADJOIN_FORMAT, time_array)


def time_stamp_to_date_time_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_TIME_ADJOIN_FORMAT, time_array)