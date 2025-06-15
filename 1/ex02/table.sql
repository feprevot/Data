DROP TABLE IF EXISTS data_2022_oct;

CREATE TABLE IF NOT EXISTS data_2022_oct (
    event_time      TIMESTAMP NOT NULL,
    event_type      VARCHAR(20),
    product_id      INTEGER,
    price           NUMERIC(10,2),
    user_id         BIGINT,
    user_session    UUID
);

COPY data_2022_oct
FROM '/customer/data_2022_oct.csv'
DELIMITER ','
CSV HEADER;

