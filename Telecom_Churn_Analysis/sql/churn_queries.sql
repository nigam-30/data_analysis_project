-- ============================================================================
-- SQL Queries for Telecom Customer Churn Analysis
-- ============================================================================
-- Project  : Customer Churn Analysis for Telecom Industry
-- Intern   : Data Analyst Internship — Elevate Labs
-- Database : SQLite (in-memory via Python, table: telecom_churn)
--
-- Business Context:
--   These queries help the telecom business understand WHY customers churn.
--   Each query targets a specific business question to inform retention strategy.
-- ============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 1: Average Call Duration by Churn Status
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Churned customers tend to have LOWER call durations,
-- indicating reduced engagement before they leave. This metric can serve
-- as an early warning signal for the retention team.

SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END AS ChurnStatus,
    ROUND(AVG(CallDuration), 2)                          AS AvgCallDuration,
    COUNT(*)                                              AS CustomerCount
FROM telecom_churn
WHERE CallDuration IS NOT NULL
GROUP BY Churn
ORDER BY Churn;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 2: Complaint Count Comparison by Churn
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Higher complaint counts strongly correlate with churn.
-- If a customer files 3+ complaints, the churn risk increases significantly.
-- This tells the business to prioritize rapid complaint resolution.

SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END AS ChurnStatus,
    ROUND(AVG(Complaints), 2)                             AS AvgComplaints,
    SUM(Complaints)                                       AS TotalComplaints,
    COUNT(*)                                              AS CustomerCount
FROM telecom_churn
GROUP BY Churn
ORDER BY Churn;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 3: Recharge Frequency by Churn
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Active customers recharge more frequently.
-- Low recharge frequency is a behavioral signal of disengagement —
-- the customer is mentally "checking out" before formally churning.

SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END AS ChurnStatus,
    ROUND(AVG(RechargeFrequency), 2)                      AS AvgRechargeFreq,
    MIN(RechargeFrequency)                                 AS MinRechargeFreq,
    MAX(RechargeFrequency)                                 AS MaxRechargeFreq,
    COUNT(*)                                               AS CustomerCount
FROM telecom_churn
GROUP BY Churn
ORDER BY Churn;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 4: Contract Type Churn Rate
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Month-to-Month contracts have dramatically higher churn.
-- This justifies investing in contract conversion campaigns — offering
-- discounts to migrate customers to annual plans locks them in.

SELECT
    ContractType,
    COUNT(*)                                               AS TotalCustomers,
    SUM(Churn)                                             AS ChurnedCustomers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)               AS ChurnRate_Pct
FROM telecom_churn
GROUP BY ContractType
ORDER BY ChurnRate_Pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 5: Monthly Revenue Loss Due to Churn
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Quantifies the financial impact of churn.
-- This single number ("we lose ₹X/month to churn") is the most
-- persuasive metric for C-suite buy-in on retention programs.

SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END  AS ChurnStatus,
    COUNT(*)                                                AS Customers,
    ROUND(SUM(MonthlyCharges), 2)                           AS TotalMonthlyRevenue,
    ROUND(AVG(MonthlyCharges), 2)                           AS AvgMonthlyCharge
FROM telecom_churn
GROUP BY Churn
ORDER BY Churn;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 6: Churn Rate by Internet Service Type
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Fiber Optic customers may churn more due to
-- higher expectations and pricing. DSL and No-service customers
-- have different retention profiles.

SELECT
    InternetService,
    COUNT(*)                                               AS TotalCustomers,
    SUM(Churn)                                             AS ChurnedCustomers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)               AS ChurnRate_Pct
FROM telecom_churn
GROUP BY InternetService
ORDER BY ChurnRate_Pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 7: Churn Rate by Payment Method
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Electronic Check users churn more —
-- this is a well-known telecom industry pattern. Promoting
-- auto-pay or credit card payments reduces friction and churn.

SELECT
    PaymentMethod,
    COUNT(*)                                               AS TotalCustomers,
    SUM(Churn)                                             AS ChurnedCustomers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)               AS ChurnRate_Pct
FROM telecom_churn
GROUP BY PaymentMethod
ORDER BY ChurnRate_Pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 8: Tenure Buckets and Churn Rate
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: New customers (0-12 months) churn the most.
-- This tells the business that the first-year experience is make-or-break.
-- Onboarding programs and early engagement campaigns are critical.

SELECT
    CASE
        WHEN Tenure BETWEEN 1 AND 12  THEN '01. 0-12 Months'
        WHEN Tenure BETWEEN 13 AND 24 THEN '02. 13-24 Months'
        WHEN Tenure BETWEEN 25 AND 48 THEN '03. 25-48 Months'
        WHEN Tenure > 48              THEN '04. 49+ Months'
    END                                                    AS TenureBucket,
    COUNT(*)                                               AS TotalCustomers,
    SUM(Churn)                                             AS ChurnedCustomers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)               AS ChurnRate_Pct
FROM telecom_churn
GROUP BY TenureBucket
ORDER BY TenureBucket;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 9: Senior Citizen Churn Analysis
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Senior citizens may have different churn patterns.
-- If they churn more, consider senior-friendly plans with
-- simplified billing and dedicated support channels.

SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS AgeGroup,
    COUNT(*)                                               AS TotalCustomers,
    SUM(Churn)                                             AS ChurnedCustomers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)               AS ChurnRate_Pct,
    ROUND(AVG(MonthlyCharges), 2)                          AS AvgMonthlyCharge
FROM telecom_churn
GROUP BY SeniorCitizen
ORDER BY ChurnRate_Pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 10: High-Value Customer Churn (Monthly Charges > 75)
-- ─────────────────────────────────────────────────────────────────────────────
-- Business Insight: Losing high-value customers hurts revenue the most.
-- This query identifies the revenue impact of churn in the premium segment.

SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END  AS ChurnStatus,
    COUNT(*)                                                AS HighValueCustomers,
    ROUND(SUM(MonthlyCharges), 2)                           AS MonthlyRevenue,
    ROUND(AVG(Tenure), 1)                                   AS AvgTenure,
    ROUND(AVG(Complaints), 2)                               AS AvgComplaints
FROM telecom_churn
WHERE MonthlyCharges > 75
GROUP BY Churn
ORDER BY Churn;
