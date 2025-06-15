DO $$
DECLARE
    file_name TEXT;
    table_name TEXT;
    full_path TEXT;
    create_sql TEXT;
    copy_sql TEXT;
BEGIN
	FOR file_name IN
        SELECT TRIM(name) AS file_name
        FROM pg_ls_dir('/customer') AS name
        WHERE TRIM(name) LIKE '%.csv'
    LOOP
        table_name := replace(file_name, '.csv', '');
        full_path := '/customer/' || file_name;

        EXECUTE format('DROP TABLE IF EXISTS %I;', table_name);

        create_sql := format($f$
            CREATE TABLE %I (
                event_time TIMESTAMP,
                event_type TEXT,
                product_id INTEGER,
                price NUMERIC(10,2),
                user_id BIGINT,
                user_session UUID
            );
        $f$, table_name);

        EXECUTE create_sql;

        copy_sql := format($f$
            COPY %I FROM %L DELIMITER ','
            CSV HEADER;$f$, table_name, full_path);
        EXECUTE copy_sql;

        RAISE NOTICE 'Table % created and filled from %', table_name, full_path;
    END LOOP;
END $$;
