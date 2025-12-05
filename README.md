## Overview ##

This repository contains a multi-part quantitative research and portfolio analytics project. It includes:
    Construction and evaluation of momentum-based signals
    Analysis of hedge-fund crowding risk
    Development of custom sector classifications using clustering and feature engineering
    Cross-industry fundamental analysis of earnings drivers
    A portfolio analytics section addressing seven practical questions from a portfolio manager
    All analysis is supported by Python code and a slide deck summarizing results.

## 1. Approach ##
                                    Momentum Signal Construction
Implemented classic momentum using the 12m–1m signal, volatility-adjusted.
Pulled historical data using yfinance and ranked securities by momentum.
Evaluated long-only and composite signals (60% stock + 40% sector) and measured CAGR, Volatility, Sharpe, Max Drawdown, and Hit Rate.
Results and comparisons are summarized in the slide deck.
Crowding Analysis Framework
Outlined risks of holding crowded securities (e.g., liquidity crunch, forced unwinds, squeezes).
Proposed empirical tests using 13F filings, hedge fund ownership datasets, and Bloomberg/Barclays indices.
Constructed a basket of historically crowded securities and examined drawdowns, factor sensitivities, and performance stability.

                                    Custom Sector Construction
Engineered features across three categories:
Market/price features (returns, volatility, liquidity, momentum)
Fundamental features (revenue, cash flow, leverage, R&D intensity)
NLP-based features (company descriptions, filings, news embeddings)
Applied PCA for dimensionality reduction and clustering algorithms such as K-means or spectral clustering.
Evaluated cluster quality based on economic interpretability and improvements in risk-adjusted performance.
Cross-Industry Earnings Analysis
Compared Qualcomm (fabless semiconductor + licensing model) and Axon (hardware-enabled SaaS public-safety ecosystem).
Identified each firm’s primary earnings drivers and KPIs predictive of future performance:
Qualcomm: handset vs. auto/IoT mix, licensing stability, AI-enabled chipset adoption
Axon: ARR growth, subscription mix, backlog, device deployments, attach rates
Used SEC filings, earnings reports, and market data as primary sources.
                                    Portfolio Analytics Responses
Each PM question was answered using established risk-model, attribution, and statistical techniques, including:

Diagnosing factor drift in momentum exposure
Reconciling model Beta vs. empirical Beta
Explaining why idiosyncratic volatility may appear understated
Quantifying trade-timing skill using execution benchmarks
Measuring position-size concentration with HHI and effective number of positions
Decoupling Sharpe vs. IR via factor return attribution
Running event studies on FOMC/CPI dates to estimate macro sensitivity and expected PnL impact

## 2. Challenges & Solutions ##
                                    Data Quality

Addressed missing or inconsistent price data from yfinance using validation and cleaning.
Standardized return calculations to ensure comparability across securities.
High-Dimensional Feature Space
Used PCA and scaling to manage noise and feature imbalance.
Balanced the need for interpretability against clustering performance.
Cluster Stability
Smoothed features over longer windows to avoid regime-specific instability.
Evaluated cluster robustness over multiple time periods.
Risk Model Limitations
Reconciled model outputs with empirical diagnostics (e.g., regressions for Beta, intraday volatility not captured by EOD models).
Identified where factor mis-specification could underestimate idiosyncratic volatility.

## 3. Assumptions ##
                                    Momentum

Assumed availability of continuous, clean pricing data.
Used 12m–1m momentum as a standard baseline with periodic rebalancing.
Feature Engineering & Clustering
Assumed fundamental and NLP features were consistently available across securities.
Assumed PCA captures the most relevant variance while reducing noise.
Portfolio Analytics
Assumed a linear multi-factor risk model with stable factor definitions.
Used given idiosyncratic variance and turnover instead of recomputing exposures from scratch.
Answers follow the 3-sentence constraint required by the prompt.
Fundamental Analysis
Relied on company-reported KPIs, SEC filings, and market consensus as accurate data sources.

## 4. Repository Contents ##
/Question_I.py         # Momentum signal construction & evaluation
/Slide-Deck.pptx       # Slides covering methodology, results, and PM responses
README.md              # Project documentation

Slide deck citation: The results, discussion, and summaries referenced above are documented in Slide-Deck.pptx. 
Slide-Deck

## 5. Getting Started ##
Requirements
Python 3.9+
Packages: yfinance, pandas, numpy, scikit-learn, matplotlib, seaborn
Running Momentum Analysis
python Question_I.py

## Viewing Slides ##
Open Slide-Deck.pptx in PowerPoint, Google Slides, or Keynote.

## 6. Contact ##

If you have questions or would like to propose improvements, feel free to open an issue or submit a pull request.