DO $$
DECLARE
    deleted_rows INTEGER;
BEGIN
    SELECT COUNT(*) INTO deleted_rows
    FROM (
        SELECT *,
               LAG(event_time) OVER (
                   PARTITION BY event_type, product_id, user_id, price, user_session
                   ORDER BY event_time
               ) AS previous_time
        FROM customers
    ) sub
    WHERE previous_time IS NOT NULL
      AND event_time - previous_time <= INTERVAL '1 second';

    DELETE FROM customers c
    USING (
        SELECT 
            ctid,
            LAG(event_time) OVER (
                PARTITION BY event_type, product_id, user_id, price, user_session
                ORDER BY event_time
            ) AS previous_time,
            event_time,
            event_type,
            product_id,
            user_id,
            price,
            user_session
        FROM customers
    ) dup
    WHERE c.ctid = dup.ctid
      AND dup.previous_time IS NOT NULL
      AND dup.event_time - dup.previous_time <= INTERVAL '1 second';

    RAISE NOTICE '🧹 % duplicate row(s) deleted from customers.', deleted_rows;
END $$;
