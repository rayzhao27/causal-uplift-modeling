-- Full-experiment allocation and outcome counts.
SELECT
    COUNT(*) AS n_rows,
    SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END) AS n_treatment,
    SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END) AS n_control,
    AVG(treatment) AS treatment_share,
    SUM(CASE WHEN treatment = 1 THEN conversion ELSE 0 END) AS conversion_treatment,
    SUM(CASE WHEN treatment = 0 THEN conversion ELSE 0 END) AS conversion_control,
    SUM(CASE WHEN treatment = 1 THEN visit ELSE 0 END) AS visit_treatment,
    SUM(CASE WHEN treatment = 0 THEN visit ELSE 0 END) AS visit_control,
    AVG(CASE WHEN treatment = 1 THEN conversion END) AS conversion_rate_treatment,
    AVG(CASE WHEN treatment = 0 THEN conversion END) AS conversion_rate_control,
    AVG(CASE WHEN treatment = 1 THEN visit END) AS visit_rate_treatment,
    AVG(CASE WHEN treatment = 0 THEN visit END) AS visit_rate_control,
    AVG(CASE WHEN treatment = 1 THEN exposure END) AS exposure_rate_treatment,
    AVG(CASE WHEN treatment = 0 THEN exposure END) AS exposure_rate_control
FROM experiment;
