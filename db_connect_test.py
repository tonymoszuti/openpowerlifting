from psycopg2 import connect, DatabaseError

def connect_to_postgres():
    try:
        connection = connect(
            dbname="openpowerlifting",
            user="xxxx",
            password="xxxx",
            host="postgres-db",
            port="xxxx"
        )

        cursor = connection.cursor()

        print("Hello, world!")

        cursor.close()
        connection.commit()  # Commit any pending transaction
        connection.close()

    except (Exception, DatabaseError) as error:
        print(error)

if __name__ == "__main__":
    connect_to_postgres()
