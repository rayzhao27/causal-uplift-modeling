SELECT feature, bin, treatment, COUNT(*) AS n, AVG(conversion) AS conversion_rate, AVG(visit) AS visit_rate
FROM (
SELECT treatment, conversion, visit, 'age' AS feature, ntile(5) OVER (ORDER BY age) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'tenure_months' AS feature, ntile(5) OVER (ORDER BY tenure_months) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'prior_purchases' AS feature, ntile(5) OVER (ORDER BY prior_purchases) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'prior_spend' AS feature, ntile(5) OVER (ORDER BY prior_spend) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'recency_days' AS feature, ntile(5) OVER (ORDER BY recency_days) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'sessions_30d' AS feature, ntile(5) OVER (ORDER BY sessions_30d) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'email_opens_30d' AS feature, ntile(5) OVER (ORDER BY email_opens_30d) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'pages_viewed_30d' AS feature, ntile(5) OVER (ORDER BY pages_viewed_30d) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'avg_order_value' AS feature, ntile(5) OVER (ORDER BY avg_order_value) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'discount_sensitivity' AS feature, ntile(5) OVER (ORDER BY discount_sensitivity) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'is_loyalty_member' AS feature, ntile(5) OVER (ORDER BY is_loyalty_member) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'is_weekend_shopper' AS feature, ntile(5) OVER (ORDER BY is_weekend_shopper) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'device_mobile' AS feature, ntile(5) OVER (ORDER BY device_mobile) AS bin FROM experiment
UNION ALL
SELECT treatment, conversion, visit, 'device_desktop' AS feature, ntile(5) OVER (ORDER BY device_desktop) AS bin FROM experiment
) AS binned
GROUP BY feature, bin, treatment
ORDER BY feature, bin, treatment;
