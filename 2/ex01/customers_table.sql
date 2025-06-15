DO $$
DECLARE
    tbl RECORD;
BEGIN
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        event_time TIMESTAMP,
        event_type TEXT,
        product_id INTEGER,
        price NUMERIC(10,2),
        user_id BIGINT,
        user_session UUID
    );

    FOR tbl IN
        SELECT tablename FROM pg_tables
        WHERE tablename LIKE 'data_202%' AND schemaname = 'public'
    LOOP
        EXECUTE format('INSERT INTO customers SELECT * FROM %I;', tbl.tablename);
        RAISE NOTICE '✅ data inserted from %', tbl.tablename;
    END LOOP;
END $$;
