from cProfile import label
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from air_pressure.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params

class Preprocessor:
    """
    Description : This class shall  be used to clean and transform the data before training.
    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    
    def __init__(self, log_file):
        self.log_writer = App_Logger()
        
        self.config = read_params()
        
        self.log_file = log_file
        
        self.knn_neighbours = self.config["knn_imputer"]["n_neighbors"]
        
        self.knn_weights = self.config["knn_imputer"]["weights"]

        self.null_values_file = self.config["null_values_csv_file"]

        self.n_components = self.config["pca_model"]["n_components"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3 = S3_Operation()
        
    def remove_columns(self, data, columns):
        """
        Method Name :   remove_columns
        Description :   This method removes the given columns from a pandas dataframe.
        
        Output      :   A pandas DataFrame after removing the specified columns.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """ 
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.remove_columns.__name__,
            __file__,
            self.log_file,
        )
        
        self.log_writer.start_log("start", **log_dic)
        
        self.data = data
        
        self.columns = columns
        
        try:
            self.useful_data = self.data.drop(labels = self.columns, axis=1)
            
            self.log_writer.log(f"Dropped {columns} from {data}", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return self.useful_data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
            
    def separate_label_feature(self, data, label_column_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the features and a Label Coulmns.
        
        Output      :   Returns two separate dataframes, one containing features and the other containing Labels .
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.separate_label_feature.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writer.log(f"Separated {label_column_name} from {data}", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return self.X, self.Y

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
        
    def replace_invalid_values(self, data):
        """
        Method Name :   replace_invalid_values
        Description :   This method replaces the invalid values with np.nan
        
        Output      :   A dataframe without invalid values is returned
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.replace_invalid_values.__name__,
            __file__,
            self.log_file,
        )

        try:
            self.log_writer.start_log("start", **log_dic)

            data.replace(to_replace="'na'", value=np.nan, inplace=True)

            self.log_writer.log("Replaced " "na" " with np.nan", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)