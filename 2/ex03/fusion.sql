ALTER TABLE customers ADD COLUMN IF NOT EXISTS category_id BIGINT;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS category_code VARCHAR(100);
ALTER TABLE customers ADD COLUMN IF NOT EXISTS brand VARCHAR(100);

MERGE INTO customers AS c
USING (
    SELECT DISTINCT ON (product_id)
        product_id,
        category_id,
        category_code,
        brand
    FROM items
    ORDER BY product_id
) AS i
ON c.product_id = i.product_id
WHEN MATCHED THEN
  UPDATE SET
    category_id = i.category_id,
    category_code = NULLIF(i.category_code, ''),
    brand = NULLIF(i.brand, '');