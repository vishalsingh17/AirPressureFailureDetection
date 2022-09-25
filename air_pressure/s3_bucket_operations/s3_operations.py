import imp
import json
from logging import exception
import os
import pickle
from io import StringIO
from pprint import pp

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class S3_Operations:
    """
    Class Name  :   S3_Operations
    Description :   This method is used for all the S3 bucket operations
    Written by  :   Vishal Singh

    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.file_format = self.config["save_format"]

        self.s3_client = boto3.client("s3")

        self.s3_resource = boto3.resource("s3")

    def read_object(self, object, log_file, decode=True, make_readable=False):
        """
        Method Name :   read_object
        Description :   This method reads the object with kwargs

        Output      :   A object is read with kwargs
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.0
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.read_object.__name__, __file__, log_file
        )

        self.log_writer.start_log("Start", **log_dic)

        try:
            func = (
                lambda: object.get()["Body"].read().decode()
                if decode is True
                else object.get()["Body"].read()
            )

            self.log_writer.log(
                f"Read the s3 object with decode as {decode}", **log_dic
            )

            conv_func = lambda: StringIO(func()) if make_readable is True else func()

            self.log_writer.log(
                f"read the s3 object with make_readable as {make_readable}", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return conv_func()

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def read_text(self, fname, bucket, log_file):
        """
        Method Name :   read_text
        Description :   This method reads the text data from s3 bucket

        Output      :   Text data is read from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.0
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.read_text.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            txt_obj = self.get_file_object(fname, bucket, log_file)

            content = self.read_object(txt_obj, log_file)

            self.log_writer.log(
                f"Read {fname} file as text from {bucket} bucket", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return content

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
            
            
