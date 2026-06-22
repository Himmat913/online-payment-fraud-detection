import os
import tempfile
import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error


def get_secret_value(section, key, env_key, default=None):
    """
    Reads config in this order:
    1. Streamlit secrets.toml
    2. Environment variables / deployment secrets
    3. Default value
    """

    try:
        return st.secrets[section][key]
    except Exception:
        return os.getenv(env_key, default)


def create_ssl_ca_file_from_secret():
    """
    On cloud deployment, the CA certificate can be stored as a secret string.
    This writes it to a temporary .pem file and returns the path.
    """

    ca_content = get_secret_value("mysql", "ssl_ca_content", "MYSQL_SSL_CA_CONTENT", None)

    if not ca_content:
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem", mode="w")
    temp_file.write(ca_content)
    temp_file.close()

    return temp_file.name


def get_connection():
    """
    Creates a MySQL connection.

    Local:
    - Uses .streamlit/secrets.toml.

    Deployment:
    - Uses Hugging Face / Streamlit secrets exposed as environment variables.
    """

    host = get_secret_value("mysql", "host", "MYSQL_HOST", "localhost")
    user = get_secret_value("mysql", "user", "MYSQL_USER", "root")
    password = get_secret_value("mysql", "password", "MYSQL_PASSWORD", "")
    database = get_secret_value("mysql", "database", "MYSQL_DATABASE", "fraud_risk_platform")
    port = int(get_secret_value("mysql", "port", "MYSQL_PORT", 3306))

    ssl_ca = get_secret_value("mysql", "ssl_ca", "MYSQL_SSL_CA", None)
    ssl_ca_content_path = create_ssl_ca_file_from_secret()

    if ssl_ca_content_path:
        ssl_ca = ssl_ca_content_path

    ssl_disabled_value = get_secret_value("mysql", "ssl_disabled", "MYSQL_SSL_DISABLED", "false")
    ssl_disabled = str(ssl_disabled_value).lower() in ["true", "1", "yes"]

    connection_config = {
        "host": host,
        "user": user,
        "password": password,
        "database": database,
        "port": port,
    }

    if ssl_ca and not ssl_disabled:
        connection_config["ssl_ca"] = ssl_ca
        connection_config["ssl_verify_cert"] = True

    try:
        connection = mysql.connector.connect(**connection_config)

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