# credit-default-prediction

Can we predict which credit card borrowers will default next month, and where should a lender set its approval cutoff?

After building a logistic regression baseline, gradient boosting improved AUC from 0.707 to 0.775 on 30,000 credit card borrowers, and the previous month's repayment status was the largest risk driver. Assuming a default costs five times what a good loan earns, a cutoff chosen through cross-validation earned a profit of 1,256 units on held-out data, while approving everybody resulted in a loss of 1,962.

**Full analysis:** open `credit_default_project_final.ipynb`

Data
UCI "Default of Credit Card Clients" dataset: 30,000 borrowers, 22.1% defaulted, no missing values. I merged undocumented education codes (0, 5, 6) into "other." I removed sex and marital status because US lenders can't legally use them, and accuracy was unchanged without them.

Methods
- Logistic regression baseline (AUC 0.707) vs. gradient boosting (AUC 0.775)
- AUC rather than accuracy, since 78% of borrowers don't default
- Cutoff chosen with an assumed gain of 1 unit per repaid loan and a loss of 5 units per default, selected by 5-fold cross-validation on training data (cutoff 0.16)

Key findings
- Default rates rose from about 13-17% for on-time borrowers to 34% at one month late and 69% at two months late
- Gradient boosting gained most among moderate-risk borrowers, not the highest-risk ones
- The cutoff gets stricter as defaults get costlier: from about 0.28 at a 2:1 loss ratio to 0.09 at 10:1

Limitations
- The gain and loss values are my assumptions, not from the data
- Everyone in the data was already approved for a card, so outcomes for rejected applicants are unobserved (selection bias)
- Single snapshot, so no testing over time, and the models weren't tuned beyond defaults
- Gradient boosting is more accurate but harder to explain than logistic regression

## Tools
Python (pandas, scikit-learn, matplotlib), Google Colab
