from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2




try:
    connection = psycopg2.connect(
        host="ip", #put ip here
        database="database",
        user="database",
        password="password", #password
        port="port"
    )
 
    cursor = connection.cursor()

    # i to 2 is the range
    for i in range(2):
    #get items from table here
        print("loop number: ", i)
        all_rows = []
        cursor.execute(f"SELECT columns FROM table ORDER BY id LIMIT 1 OFFSET {i}")
        rows = cursor.fetchall()
        for row in rows:
            # all_rows.append(rows[0])
            # print("row: ", row)
            all_rows.append(row)
            
        cursor.execute("UPDATE table SET columns = %s WHERE id = %s;", (str(all_rows[0][0]), str(all_rows[0][1]), str(all_rows[0][2]), str(all_rows[0][3]), str(i+1)))

        #commented out so as not to make a new table every time
        #test table
        #cursor.execute("CREATE TABLE table (id bigserial PRIMARY KEY, column1, column2 sparsevec(30522));")
        
        connection.commit()
 
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
 
finally:
    # Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()