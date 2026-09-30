-- Outcome rates by randomly assigned arm.
SELECT
    treatment,
    COUNT(*) AS n,
    SUM(conversion) AS conversions,
    AVG(conversion) AS conversion_rate,
    SUM(visit) AS visits,
    AVG(visit) AS visit_rate,
    AVG(exposure) AS exposure_rate
FROM experiment
GROUP BY treatment
ORDER BY treatment;
