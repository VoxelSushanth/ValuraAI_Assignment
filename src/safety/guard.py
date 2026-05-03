import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SafetyResult:
    blocked: bool
    category: Optional[str] = None
    response: Optional[str] = None


# Educational bypass pattern - compiled at class level
EDUCATIONAL_PATTERN = re.compile(
    r'\b(how does|what is|what are|explain|educational|research|study|understand|example of|definition of|history of|learn about|tell me about|describe|overview of|introduction to|basics of)\b',
    re.IGNORECASE
)

# Five blocked categories with compiled regex patterns
_INSIDER_TRADING_PATTERN = re.compile(
    r'\b(insider\s+(info|tip|knowledge)|non[- ]?public\s+(info|data|knowledge|information)|tip\s+from\s+(an?\s+)?(executive|employee|insider|ceo|cfo|director)|before\s+(the\s+)?(announcement|earnings|merger|acquisition|ipo)|trade\s+on\s+(inside|non.?public)|front.?running)\b',
    re.IGNORECASE
)
_INSIDER_TRADING_RESPONSE = "I'm unable to assist with strategies based on material non-public information. Trading on insider information is illegal under securities law and carries severe criminal penalties including fines and imprisonment."

_MARKET_MANIPULATION_PATTERN = re.compile(
    r'\b(pump.and.dump|short\s+squeeze\s+(scheme|coordin|manipul)|coordinat\w*\s+.{0,25}(buying|selling|trades)|manipulat\w*\s+.{0,20}(price|market|stock)|fake\s+(volume|orders|trades)|wash\s+trading|spoofing\s+(orders|trades|the\s+market)|artificially\s+(inflate|deflate|boost)\s+.{0,15}price)\b',
    re.IGNORECASE
)
_MARKET_MANIPULATION_RESPONSE = "I can't help with coordinating market activity or artificially influencing prices. Market manipulation is a serious securities violation that can result in regulatory action and criminal prosecution."

_MONEY_LAUNDERING_PATTERN = re.compile(
    r'\b(launder(ing)?\s+money|money\s+launder(ing)?|clean\s+(dirty\s+|illegal\s+)?money|hide\s+.{0,20}(funds|money|assets)|conceal\s+.{0,20}(funds|money|assets|income|origin)|shell\s+compan\w+\s+.{0,20}(hide|conceal|launder)|dirty\s+money|structuring\s+.{0,20}(deposits|transactions)|smurfing)\b',
    re.IGNORECASE
)
_MONEY_LAUNDERING_RESPONSE = "I'm unable to assist with concealing the origin of funds. Money laundering is a serious financial crime with severe legal consequences under domestic and international law."

_GUARANTEED_RETURNS_PATTERN = re.compile(
    r'\b(guarante\w*\s+.{0,25}(return|profit|gain|income|yield)|guaranteed\s+(profit|return|income|investment)|risk[- ]?free\s+(return|profit|investment|gain)|100\s*%\s*.{0,15}(profit|return|gain|guaranteed)|no[\s-]risk\s+.{0,15}(return|profit|investment)|never\s+lose\s+(money|capital|principal))\b',
    re.IGNORECASE
)
_GUARANTEED_RETURNS_RESPONSE = "No investment can guarantee returns. Making or endorsing such claims is prohibited under financial regulations. All investments carry risk including possible loss of principal."

_RECKLESS_ADVICE_PATTERN = re.compile(
    r'\b(bet\s+(it\s+)?all|all[- ]in\s+on|put\s+everything\s+(in|into)\s+(one|a\s+single)|invest\s+all\s+(my\s+)?(savings|money|cash|capital)\s+in\s+(one|a\s+single)|take\s+out\s+.{0,20}loan.{0,20}(invest|stock|crypto)|mortgage\s+.{0,30}(invest|stock|crypto|buy\s+shares)|borrow\s+.{0,20}(invest|put\s+in\s+market)|max\s+out\s+.{0,20}credit.{0,20}invest)\b',
    re.IGNORECASE
)
_RECKLESS_ADVICE_RESPONSE = "This strategy carries extreme financial risk and I'm unable to recommend it. Investing borrowed money or concentrating all assets in one position can lead to devastating losses. Please consult a qualified financial advisor."

# Checks list at class level: (category_name, compiled_regex, response_string)
_CHECKS = [
    ("insider_trading", _INSIDER_TRADING_PATTERN, _INSIDER_TRADING_RESPONSE),
    ("market_manipulation", _MARKET_MANIPULATION_PATTERN, _MARKET_MANIPULATION_RESPONSE),
    ("money_laundering", _MONEY_LAUNDERING_PATTERN, _MONEY_LAUNDERING_RESPONSE),
    ("guaranteed_returns", _GUARANTEED_RETURNS_PATTERN, _GUARANTEED_RETURNS_RESPONSE),
    ("reckless_advice", _RECKLESS_ADVICE_PATTERN, _RECKLESS_ADVICE_RESPONSE),
]


class SafetyGuard:
    """
    Pure Python safety guard with zero network calls, zero LLM usage.
    Runs in <10ms. All regex compiled at class level.
    """
    
    def check(self, query: str) -> SafetyResult:
        # Educational bypass: if query matches educational pattern, return immediately
        if EDUCATIONAL_PATTERN.search(query):
            return SafetyResult(blocked=False)
        
        # Check each blocked category in order, return on first match
        for category, pattern, response in _CHECKS:
            if pattern.search(query):
                return SafetyResult(blocked=True, category=category, response=response)
        
        # No matches found - query is safe
        return SafetyResult(blocked=False)
