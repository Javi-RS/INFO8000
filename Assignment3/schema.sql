DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS researchers;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS tasks
DROP TABLE IF EXISTS equipment

CREATE TABLE projects(
  id INT NOT NULL,
  name TEXT,
  status TEXT,
  start_month TEXT,
  start_year INT,
  PRIMARY KEY (id),
  UNIQUE(id))
);

CREATE TABLE researchers(
  id INT NOT NULL,
  first_name TEXT,
  last_name TEXT,
  email TEXT,
  phone TEXT,
  department_id INT,
  PRIMARY KEY (id),
  FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE CASCADE ON UPDATE CASCADE)
);

CREATE TABLE departments(
  id INT NOT NULL,
  name TEXT NOT NULL,
  campus TEXT,
  PRIMARY KEY (id))
);

CREATE TABLE tasks(
  id INT NOT NULL,
  name TEXT,
  status TEXT,
  project_id INT,
  researcher_id INT,
  PRIMARY KEY (id),
  FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (researcher_id) REFERENCES researchers(id) ON DELETE CASCADE ON UPDATE CASCADE)
);

CREATE TABLE equipment(
  id INT NOT NULL,
  type TEXT NOT NULL,
  brand TEXT,
  model TEXT,
  PRIMARY KEY (id))
);
