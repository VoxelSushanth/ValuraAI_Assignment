DISCLAIMER = "This analysis is for informational purposes only and does not constitute investment advice or a solicitation to buy or sell any security. Past performance is not indicative of future results. Please consult a qualified financial advisor before making investment decisions."

OBSERVATIONS_SYSTEM_PROMPT = """You are a portfolio health analyst for Valura wealth management, writing for novice investors. Your task is to analyze portfolio metrics and generate clear, actionable observations.

RULES:
1. Write 2-4 observations surfacing the 1-2 most critical issues FIRST
2. Explain jargon in parentheses on first use (e.g., "concentration risk (having too much in one stock)")
3. Use plain language - no complex financial terms without explanation
4. Reference actual tickers and numbers from the data
5. Severity rules:
   - critical: if top position >60% OR total loss >20%
   - warning: if top position 30-60% OR underperforming benchmark by >10%
   - info: for positive observations or minor suggestions

FOR EMPTY PORTFOLIOS (no holdings):
Pivot to BUILD mode:
- Tell user an empty portfolio is an exciting starting point
- Recommend having an emergency fund first (3-6 months expenses)
- Suggest starting with diversified index funds
- Encourage defining goals before investing

OUTPUT FORMAT:
Return ONLY raw JSON with no markdown formatting:
{
    "observations": [
        {"severity": "info|warning|critical", "text": "observation text"},
        {"severity": "warning", "text": "another observation"}
    ],
    "raw_summary": "3-5 plain sentences summarizing the portfolio health for a novice investor"
}

The raw_summary should be conversational and encouraging while being honest about risks.
"""

OBSERVATIONS_USER_TEMPLATE = """Analyze this portfolio and generate observations:

METRICS:
{metrics_json}

USER RISK PROFILE:
{risk_profile}

MARKET CONTEXT:
{market_context}

Generate 2-4 observations and a raw_summary."""

EMPTY_PORTFOLIO_RESULT = {
    "concentration_risk": {
        "top_position_pct": 0.0,
        "top_3_positions_pct": 0.0,
        "flag": "low",
        "top_holding": None
    },
    "performance": {
        "total_cost": 0.0,
        "total_value": 0.0,
        "total_return_pct": 0.0,
        "annualized_return_pct": None,
        "total_gain_loss": 0.0
    },
    "benchmark_comparison": None,
    "observations": [
        {"severity": "info", "text": "An empty portfolio is an exciting starting point! You have the opportunity to build a solid foundation for your financial future."},
        {"severity": "info", "text": "Before investing, make sure you have an emergency fund covering 3-6 months of expenses in a savings account."},
        {"severity": "info", "text": "Consider starting with diversified index funds (like S&P 500 ETFs) which spread your money across many companies automatically."},
        {"severity": "info", "text": "Take time to define your investment goals (retirement, house down payment, etc.) and understand your risk tolerance before making investments."}
    ],
    "raw_summary": "Your portfolio is empty, which means you're at the beginning of your investment journey! This is actually a great position to be in because you can start fresh with good habits. First, make sure you have an emergency fund saved up. Then consider starting with low-cost index funds that give you instant diversification. Take time to think about your goals and how much risk you're comfortable with. Remember, investing is a marathon, not a sprint.",
    "disclaimer": DISCLAIMER,
    "is_empty_portfolio": True
}
