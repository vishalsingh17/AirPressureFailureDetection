from air_pressure.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Data_Getter_Train:
    """
    Description :   This class shall be used for obtaining the df from the input files s3 bucket where the training file is present

    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, log_file):
        self.config = read_params()

        self.log_file = log_file

        self.train_csv_file = self.config["export_csv_file"]["train"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the input files s3 bucket where the training file is stored
        Output      :   A pandas dataframe

        On Failure  :   Write an exception log and then raise exception

        Version     :   1.0
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_data.__name__, __file__, self.log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            df = self.s3.read_csv(
                self.train_csv_file, self.input_files_bucket, self.log_file
            )

            self.log_writer.start_log("exit", **log_dic)

            return df

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
