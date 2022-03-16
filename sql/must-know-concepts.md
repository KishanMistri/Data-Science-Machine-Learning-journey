# SQL for AI-ML

# Database Introduction

Is faster, reliable, secure, 

- Relational database:
    - Based on the relational model, an intuitive, straightforward way of representing data in tables. In a relational database, each row in the table is a record with a unique ID called the key.
    - SQL, MySQL, Oracle
- Non-Relational database: ***
    - MongoDB, PostgreSQL, NoSQL

## **ACID properties and RDBMS**

Four crucial properties define relational database transactions: atomicity, consistency, isolation, and durability—typically referred to as ACID.

- **Atomicity** defines all the elements that make up a complete database transaction.
- **Consistency** defines the rules for maintaining data points in a correct state after a transaction.
- **Isolation** keeps the effect of a transaction invisible to others until it is committed, to avoid confusion.
- **Durability** ensures that data changes become permanent once the transaction is committed.

### Normalization

*Database normalization is the process of restructuring a relational database in accordance with a series of so-called normal forms in order to reduce data redundancy and improve data integrity. It was first proposed by Edgar F. Codd as an integral part of his relational model.*

## SQL

- (Database) Domain specific Language
- Declarative programming language (Not Procedural)
    
    Simple execution steps
    
    - → SQL Query input
    - → Parser (Understands input)
    - → Compiler (creates/converts code to procedural)
    - →  Query Optimizer (Check the optimized/best way internally)
    - → Query Executer (Runs the procedure and retrieves the Results)

- The `PRIMARY KEY` constraint uniquely identifies each record in a table.
    - Primary keys must contain UNIQUE values, and cannot contain NULL values.
- The `FOREIGN KEY` constraint is used to prevent actions that would destroy links between tables.
    - A `FOREIGN KEY` is a field (or collection of fields) in one table, that refers to the `[PRIMARY KEY](https://www.w3schools.com/sql/sql_primarykey.asp)` in another table.
    - The table with the foreign key is called the child table, and the table with the primary key is called the referenced or parent table

### SQL KEYWORDS:

- SELECT
- ORDER BY
- GROUP BY
- LIMIT
- OFFSET
- DISTINCT
- WHERE - Works well with individual ROWs
- HAVING (AFTER CONDITIONALLY with GROUP BY) Works well with GROUP
- OPERATORS: AND, OR, NOT, ALL,  ANY, BETWEEN, EXISTS, IN, LIKE, SOME
- AGGREGATE FUNCTIONs: COUNT, MIN, MAX, SUM, AVG  (Creating alias after agregated field:  SELECT COUNT(field) Field_count FROM <TABLE> ORDER BY field_count)
  

###  Order of keywords

"SELECT ***  FROM *table* WHERE x **GROUP BY x HAVING *x* ORDER BY x LIMIT x OFFSET"

Refer: [https://dev.mysql.com/doc/refman/8.0/en/select.html](https://dev.mysql.com/doc/refman/8.0/en/select.html)

### JOIN

Syntax

### 1. INNER JOIN (Intersection)

SELECT m.f1,...,m.fn, n.f1,...,n.fn 

FROM TABLE1 m 

JOIN TABLE2 n 

ON m.f1 = n.f1

### 2. LEFT OUTER JOIN (All A ones)

join → left outer join

### 3. RIGHT OUTER JOIN (All B ones)

join → right outer join

### 4. FULL OUTER JOIN (UNION of A & B)

join → full outer join

### 5. CROSS PRODUCT

without any join word 1→ many 

i.e.,  SELECT m.f1,...,m.fn, n.f1,...,n.fn 

  FROM TABLE1 m , TABLE2 n 

### 5. Natural JOIN:

Where there is a same column NAME in 2 table and joining based on that column

SELECT m.f1,...,m.fn, n.f1,...,n.fn 

FROM TABLE1 m 

JOIN TABLE2 n 

USING <COMMON_COLUMN>

Solve Subquery examples

Solve Corelated examples

[https://platform.stratascratch.com/coding?questionType=2&company=&topic=&curated_filter=&is_correct_solution=&is_bookmarked=&is_freemium=true&in_depth_solution=&difficulty=1&code_type=1&python=1&filters=&page=1&page_size=100](https://platform.stratascratch.com/coding?questionType=2&company=&topic=&curated_filter=&is_correct_solution=&is_bookmarked=&is_freemium=true&in_depth_solution=&difficulty=1&code_type=1&python=1&filters=&page=1&page_size=100)

## Operations:

### DATA QUERY LANG (DQL)

- SELECT

### DATA MANIPULATION LANG (DML)

- CREATE:
    - The following constraints are commonly used in SQL:
    - `[NOT NULL](https://www.w3schools.com/sql/sql_notnull.asp)` - Ensures that a column cannot have a NULL value
    - `[UNIQUE](https://www.w3schools.com/sql/sql_unique.asp)` - Ensures that all values in a column are different
    - `[PRIMARY KEY](https://www.w3schools.com/sql/sql_primarykey.asp)` - A combination of a `NOT NULL` and `UNIQUE`. Uniquely identifies each row in a table
    - `[FOREIGN KEY](https://www.w3schools.com/sql/sql_foreignkey.asp)` - Prevents actions that would destroy links between tables
    - `[CHECK](https://www.w3schools.com/sql/sql_check.asp)` - Ensures that the values in a column satisfies a specific condition
    - `[DEFAULT](https://www.w3schools.com/sql/sql_default.asp)` - Sets a default value for a column if no value is specified
    - `[CREATE INDEX](https://www.w3schools.com/sql/sql_create_index.asp)` - Used to create and retrieve data from the database very quickly
- ALTER TABLE
    - Add
    - Modify
    - Drop
- TRUNCATE: All the records/data will be deleted and the table remains empty. (The table/relation doesn't get deleted)
- INSERT
- UPDATE
- DELETE

### DATA CONTROL LANG (DCL)

- GRANT
- REVOKE

Problems:

Easy Problems:

1. [https://leetcode.com/problems/second-highest-salary/](https://leetcode.com/problems/second-highest-salary/)
2. [https://leetcode.com/problems/combine-two-tables/](https://leetcode.com/problems/combine-two-tables/)
3. [https://leetcode.com/problems/employees-earning-more-than-their-managers/](https://leetcode.com/problems/employees-earning-more-than-their-managers/)
4. [https://leetcode.com/problems/delete-duplicate-emails/](https://leetcode.com/problems/delete-duplicate-emails/)
5. [https://leetcode.com/problems/reformat-department-table/](https://leetcode.com/problems/reformat-department-table/)
6. [https://leetcode.com/problems/swap-salary/](https://leetcode.com/problems/swap-salary/)
7. [https://leetcode.com/problems/customers-who-never-order/](https://leetcode.com/problems/customers-who-never-order/)
8. [https://leetcode.com/problems/rising-temperature/](https://leetcode.com/problems/rising-temperature/)
9. [https://leetcode.com/problems/not-boring-movies/](https://leetcode.com/problems/not-boring-movies/)

**Medium And Hard Problems:**

1. **[https://leetcode.com/problems/exchange-seats/](https://leetcode.com/problems/exchange-seats/)**
2. [https://leetcode.com/problems/rank-scores/](https://leetcode.com/problems/rank-scores/)
3. [https://leetcode.com/problems/consecutive-numbers/](https://leetcode.com/problems/consecutive-numbers/)
4. [https://leetcode.com/problems/department-highest-salary/](https://leetcode.com/problems/department-highest-salary/)
5. [https://leetcode.com/problems/nth-highest-salary/](https://leetcode.com/problems/nth-highest-salary/)
6. [https://leetcode.com/problems/human-traffic-of-stadium/](https://leetcode.com/problems/human-traffic-of-stadium/)
7. [https://leetcode.com/problems/department-top-three-salaries/](https://leetcode.com/problems/department-top-three-salaries/)
8. [https://leetcode.com/problems/trips-and-users/](https://leetcode.com/problems/trips-and-users/)

# Advance SQL and Internal workings

Non-Relational Database (NoSQL DBs): 

(Type of noSQL DB - Example)

- Document Store (MongoDB)
- Key-Value Store (redis)
- Graph DB
- Columnar RDBMS (Vertica)
- HIVE (Hadoop Based)
- Elastic search (inverted indexes)
- Multiple Model (DynamoDB, Azure COSMOS DB, Google FireStore)

Normalization:  

- To avoid reduntant data storage
- Atomic attributes

B+ Trees:

- Blocked based
- balanced Tree
- m-ary trees/Multilevel index

B+ tree V/S Hash Table

- Hash table is good with equality Search query (ID=11)
- B+ tree is good with range queries which is often used in DBs. (10<ID<50)

Read about in-memory store:

Databases in details:

## [MongoDB](https://www.notion.so/MongoDB-a291bc847fcb4f7c8326ef08ced21339)
  
[Notion Page](https://www.notion.so/kmistri/SQL-for-AI-ML-9966968faf214beaaff42b919fc6809f)
