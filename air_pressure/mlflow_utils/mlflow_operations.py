import os

import mlflow
from mlflow.tracking import MlflowClient

from air_pressure.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class MLFlow_Operation:
    """
    Description :    This class shall be used for handling all the mlflow operations

    Version     :   1.0
    Revisions   :   Moved to setup to cloud
    """

    def __init__(self, log_file):
        self.config = read_params()

        self.log_writer = App_Logger()

        self.s3 = S3_Operation()

        self.log_file = log_file

        self.mlflow_save_format = self.config["mlflow_config"]["serialization_format"]

        self.trained_models_dir = self.config["model_dir"]["trained"]

        self.staged_models_dir = self.config["model_dir"]["stag"]

        self.prod_models_dir = self.config["model_dir"]["prod"]

        self.model_save_format = self.config["save_format"]

    def get_experiment_from_mlflow(self, exp_name):
        """
        Method Name :   get_experiment_from_mlflow
        Description :   This method gets the experiment from mlflow server using the experiment name

        Output      :   An experiment which was stored in mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2

        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_experiment_from_mlflow.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            exp = mlflow.get_experiment_by_name(name=exp_name)

            self.log_writer.log(f"Got {exp_name} experiment from mlflow", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return exp

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_runs_from_mlflow(self, exp_id):
        """
        Method Name :   get_runs_from_mlflow
        Description :   This method gets the runs from the mlflow server for a particular experiment id

        Output      :   A pandas series object consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2

        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_runs_from_mlflow.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            runs = mlflow.search_runs(experiment_ids=exp_id)

            self.log_writer.log(
                f"Completed searching for runs in mlflow with experiment ids as {exp_id}",
                **log_dic,
            )

            self.log_writer.start_log("exit", **log_dic)

            return runs

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def set_mlflow_experiment(self, experiment_name):
        """
        Method Name :   set_mlflow_experiment
        Description :   This method sets the mlflow experiment with the particular experiment name

        Output      :   An experiment with experiment name will be created in mlflow server
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2

        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.set_mlflow_experiment.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            mlflow.set_experiment(experiment_name=experiment_name)

            self.log_writer.log(
                f"Set mlflow experiment with name as {experiment_name}", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_mlflow_client(self, server_uri):
        """
        Method Name :   get_mlflow_client
        Description :   This method gets mlflow client for the particular server uri

        Output      :   A mlflow client is created with particular server uri
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2

        Revisions   :   moved setup to cloud
        """

    # log_dic = get_log_dic(
    #     self.__class__.__name__,
    #     self.get_mlflow_client.__name__,
    #     __file__,
    #     self.log_file,
    # )

    # self.log_writer.start_log("start", **log_dic)

    # try:
    #     client = MlflowClient(tracking_uri=server_uri)

    #     self.log_writer.log("Got mlflow client with tracking uri", **log_dic)

    #     self.log_writer.start_log("exit", **log_dic)

    #     return client

    # except Exception as e:
    #     self.log_writer.exception_log(e, **log_dic)
