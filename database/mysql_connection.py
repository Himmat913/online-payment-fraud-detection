import os
import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error


def get_connection():
    """
    Creates a MySQL connection using Streamlit secrets first,
    then falls back to local environment/default values.
    """

    try:
        host = st.secrets["mysql"]["host"]
        user = st.secrets["mysql"]["user"]
        password = st.secrets["mysql"]["password"]
        database = st.secrets["mysql"]["database"]
        port = int(st.secrets["mysql"].get("port", 3306))
    except Exception:
        host = os.getenv("MYSQL_HOST", "localhost")
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        database = os.getenv("MYSQL_DATABASE", "fraud_risk_platform")
        port = int(os.getenv("MYSQL_PORT", 3306))

    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )

        if connection.is_connected():
            return connection

    except Error as e:
        st.error(f"MySQL connection failed: {e}")
        return None


def run_query(query):
    """
    Runs a SELECT query and returns the result as a pandas DataFrame.
    """

    connection = get_connection()

    if connection is None:
        return pd.DataFrame()

    try:
        df = pd.read_sql(query, connection)
        return df

    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

    finally:
        if connection and connection.is_connected():
            connection.close()