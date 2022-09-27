import json
import os

import pandas as pd
from pymongo import MongoClient

from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class MongoDB_Operation:
    """
    Description :   This method is used for all mongodb operations
    Written by  :   Vishal Singh

    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.DB_URL = os.environ["MONGODB_URL"]

        self.client = MongoClient(self.DB_URL)

        self.log_writer = App_Logger()

    def get_database(self, db_name, log_file):
        """
        Method Name :   get_database
        Description :   This method gets database from MongoDB from the db_name

        Output      :   A database is created in MongoDB with name as db_name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.0
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_database.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            db = self.client[db_name]

            self.log_writer.log(f"Created {db_name} database in MongoDB", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return db

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_collection(self, database, collection_name, log_file):
        """
        Method Name :   get_collection
        Description :   This method gets collection from the particular database and collection name

        Output      :   A collection is returned from database with name as collection name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.0
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_collection.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            collection = database[collection_name]

            self.log_writer.log(
                f"Created {collection_name} collection in mongodb", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return collection

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_collection_as_dataframe(self, db_name, collection_name, log_file):
        """
        Method Name :   get_collection_as_dataframe
        Description :   This method is used for converting the selected collection to dataframe

        Output      :   A collection is returned from the selected db_name and collection_name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.0
        Written by  :   iNeuron Intelligence
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_collection_as_dataframe.__name__,
            __file__,
            log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            database = self.get_database(db_name, log_file)

            collection = database.get_collection(name=collection_name)

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            self.log_writer.log("Converted collection to dataframe", **log_dic)

            self.log_writer.start_log("exit", *log_dic)

            return df

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def insert_dataframe_as_record(
        self, data_frame, db_name, collection_name, log_file
    ):
        """
        Method Name :   insert_dataframe_as_record
        Description :   This method inserts the dataframe as record in database collection

        Output      :   The dataframe is inserted in database collection
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.0
        Revisions   :   None
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.insert_dataframe_as_record.__name__,
            __file__,
            log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            records = json.loads(data_frame.T.to_json()).values()

            self.log_writer.log(f"Converted dataframe to json records", **log_dic)

            database = self.get_database(db_name, log_file)

            collection = database.get_collection(collection_name)

            self.log_writer.log("Inserting records to MongoDB", **log_dic)

            collection.insert_many(records)

            self.log_writer.log("Inserted records to MongoDB", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
