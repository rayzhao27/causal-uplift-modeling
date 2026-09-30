SELECT 'age' AS feature, MIN(age) AS min_value, approx_quantile(age, 0.01) AS p01, approx_quantile(age, 0.50) AS p50, approx_quantile(age, 0.99) AS p99, MAX(age) AS max_value FROM experiment
UNION ALL
SELECT 'tenure_months' AS feature, MIN(tenure_months) AS min_value, approx_quantile(tenure_months, 0.01) AS p01, approx_quantile(tenure_months, 0.50) AS p50, approx_quantile(tenure_months, 0.99) AS p99, MAX(tenure_months) AS max_value FROM experiment
UNION ALL
SELECT 'prior_purchases' AS feature, MIN(prior_purchases) AS min_value, approx_quantile(prior_purchases, 0.01) AS p01, approx_quantile(prior_purchases, 0.50) AS p50, approx_quantile(prior_purchases, 0.99) AS p99, MAX(prior_purchases) AS max_value FROM experiment
UNION ALL
SELECT 'prior_spend' AS feature, MIN(prior_spend) AS min_value, approx_quantile(prior_spend, 0.01) AS p01, approx_quantile(prior_spend, 0.50) AS p50, approx_quantile(prior_spend, 0.99) AS p99, MAX(prior_spend) AS max_value FROM experiment
UNION ALL
SELECT 'recency_days' AS feature, MIN(recency_days) AS min_value, approx_quantile(recency_days, 0.01) AS p01, approx_quantile(recency_days, 0.50) AS p50, approx_quantile(recency_days, 0.99) AS p99, MAX(recency_days) AS max_value FROM experiment
UNION ALL
SELECT 'sessions_30d' AS feature, MIN(sessions_30d) AS min_value, approx_quantile(sessions_30d, 0.01) AS p01, approx_quantile(sessions_30d, 0.50) AS p50, approx_quantile(sessions_30d, 0.99) AS p99, MAX(sessions_30d) AS max_value FROM experiment
UNION ALL
SELECT 'email_opens_30d' AS feature, MIN(email_opens_30d) AS min_value, approx_quantile(email_opens_30d, 0.01) AS p01, approx_quantile(email_opens_30d, 0.50) AS p50, approx_quantile(email_opens_30d, 0.99) AS p99, MAX(email_opens_30d) AS max_value FROM experiment
UNION ALL
SELECT 'pages_viewed_30d' AS feature, MIN(pages_viewed_30d) AS min_value, approx_quantile(pages_viewed_30d, 0.01) AS p01, approx_quantile(pages_viewed_30d, 0.50) AS p50, approx_quantile(pages_viewed_30d, 0.99) AS p99, MAX(pages_viewed_30d) AS max_value FROM experiment
UNION ALL
SELECT 'avg_order_value' AS feature, MIN(avg_order_value) AS min_value, approx_quantile(avg_order_value, 0.01) AS p01, approx_quantile(avg_order_value, 0.50) AS p50, approx_quantile(avg_order_value, 0.99) AS p99, MAX(avg_order_value) AS max_value FROM experiment
UNION ALL
SELECT 'discount_sensitivity' AS feature, MIN(discount_sensitivity) AS min_value, approx_quantile(discount_sensitivity, 0.01) AS p01, approx_quantile(discount_sensitivity, 0.50) AS p50, approx_quantile(discount_sensitivity, 0.99) AS p99, MAX(discount_sensitivity) AS max_value FROM experiment
UNION ALL
SELECT 'is_loyalty_member' AS feature, MIN(is_loyalty_member) AS min_value, approx_quantile(is_loyalty_member, 0.01) AS p01, approx_quantile(is_loyalty_member, 0.50) AS p50, approx_quantile(is_loyalty_member, 0.99) AS p99, MAX(is_loyalty_member) AS max_value FROM experiment
UNION ALL
SELECT 'is_weekend_shopper' AS feature, MIN(is_weekend_shopper) AS min_value, approx_quantile(is_weekend_shopper, 0.01) AS p01, approx_quantile(is_weekend_shopper, 0.50) AS p50, approx_quantile(is_weekend_shopper, 0.99) AS p99, MAX(is_weekend_shopper) AS max_value FROM experiment
UNION ALL
SELECT 'device_mobile' AS feature, MIN(device_mobile) AS min_value, approx_quantile(device_mobile, 0.01) AS p01, approx_quantile(device_mobile, 0.50) AS p50, approx_quantile(device_mobile, 0.99) AS p99, MAX(device_mobile) AS max_value FROM experiment
UNION ALL
SELECT 'device_desktop' AS feature, MIN(device_desktop) AS min_value, approx_quantile(device_desktop, 0.01) AS p01, approx_quantile(device_desktop, 0.50) AS p50, approx_quantile(device_desktop, 0.99) AS p99, MAX(device_desktop) AS max_value FROM experiment;
