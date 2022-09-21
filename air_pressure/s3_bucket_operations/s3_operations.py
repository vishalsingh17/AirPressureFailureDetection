import imp
import json
import os
import pickle
from io import StringIO

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
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud
    """