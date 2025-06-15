DROP TABLE IF EXISTS items;

CREATE TABLE IF NOT EXISTS items (
    product_id      INTEGER,
    category_id     BIGINT,
    category_code   VARCHAR(100),
    brand           VARCHAR(100)
);

COPY items
FROM '/items/item.csv'
DELIMITER ','
CSV HEADER;
