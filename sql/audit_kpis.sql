-- Monthly coverage and error monitoring for the synthetic holdout.
SELECT month,
       COUNT(*) AS applications,
       ROUND(AVG(CASE WHEN decision='human_review' THEN 1.0 ELSE 0.0 END),4) AS review_rate,
       ROUND(AVG(ABS(default-risk_probability)),4) AS mean_absolute_probability_error
FROM fact_scores
GROUP BY month
ORDER BY month;
