
# Assignment 3: Develop and Deploy a Web API using NGINX, gunicorn, and Flask

The server's IP address is: 35.188.194.6

### GET Requests
The application (test_app.py) implements 2 GET request routes that do different things:

  - Using the route http://35.188.194.6/read/all?table=xxxxx all data from the selected table (xxxxxx) is shown.
  - Using the route http://35.188.194.6/readresearcher?XX we can retrieve particular data from the researchers database.
  
Retrieve all values from table:
![GET1](/Assignment3/GET1_EX.jpg?raw=true "Example 1")

Retrieve specific data from researchers table (all the researchers in department1):
![GET2](/Assignment3/GET_request_Researchers_in_department1.jpg?raw=true "Example 2")

### POST Requests
The application implements also 1 POST request routes that do different things:

  - By requesting a POST request with the route http://35.188.194.6/addresearcher?id=xx&first_name=xx&last_name=xx&phone=xx&department_id=xx and the appropriate api-key authentication header, we are able to insert a new researcher (id, first_name, last_name, phone, department_id) into the the researchers table.
  
If the data insertion was successful, the application returns a message "Record successfully added" and shows the table with the added new data.
![POST1](/Assignment3/POST_request.jpg?raw=true "Example 3")

And now we can check the modification into the table with the route http://35.188.194.6/read/all?table=researchers
![GET3](/Assignment3/Read_table_modifications.jpg?raw=true "Example 4")

If the api-key is not provided, the application returns an UNAUTHORIZED message.
![POST2](/Assignment3/Unauthorized_access.jpg?raw=true "Example 3")

Also, if there is any problem with the insertion of data (duplicate id, wrong table key...) the application returns an "Adding data" error message.
![POST3](/Assignment3/Error_adding_data.jpg?raw=true "Example 3")


