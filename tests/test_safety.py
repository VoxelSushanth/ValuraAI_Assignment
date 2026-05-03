import pytest
import time
from src.safety.guard import SafetyGuard


class TestSafetyGuardBlocking:
    """Test that blocked queries are correctly identified"""
    
    def test_insider_trading_blocked(self):
        guard = SafetyGuard()
        result = guard.check("I have insider info from a CEO, should I trade before the announcement?")
        assert result.blocked is True
        assert result.category == "insider_trading"
    
    def test_market_manipulation_blocked(self):
        guard = SafetyGuard()
        result = guard.check("Let's coordinate buying to pump and dump this penny stock")
        assert result.blocked is True
        assert result.category == "market_manipulation"
    
    def test_money_laundering_blocked(self):
        guard = SafetyGuard()
        result = guard.check("I want to launder money through my investment account")
        assert result.blocked is True
        assert result.category == "money_laundering"
    
    def test_guaranteed_returns_blocked(self):
        guard = SafetyGuard()
        result = guard.check("This fund guarantees a 30% return every year")
        assert result.blocked is True
        assert result.category == "guaranteed_returns"
    
    def test_reckless_advice_blocked(self):
        guard = SafetyGuard()
        result = guard.check("I want to take out a mortgage to invest all-in on crypto")
        assert result.blocked is True
        assert result.category == "reckless_advice"


class TestSafetyGuardEducationalBypass:
    """Test that educational queries bypass blocking"""
    
    def test_explain_insider_trading_passes(self):
        guard = SafetyGuard()
        result = guard.check("how does insider trading work?")
        assert result.blocked is False
    
    def test_explain_pump_dump_passes(self):
        guard = SafetyGuard()
        result = guard.check("explain pump and dump schemes")
        assert result.blocked is False
    
    def test_what_is_money_laundering_passes(self):
        guard = SafetyGuard()
        result = guard.check("what is money laundering?")
        assert result.blocked is False
    
    def test_educational_guaranteed_returns_passes(self):
        guard = SafetyGuard()
        result = guard.check("what is a guaranteed return and why is it a red flag?")
        assert result.blocked is False
    
    def test_research_reckless_passes(self):
        guard = SafetyGuard()
        result = guard.check("research on why leveraged investing is risky")
        assert result.blocked is False


class TestSafetyGuardNormalQueries:
    """Test that normal queries pass through"""
    
    def test_portfolio_query_passes(self):
        guard = SafetyGuard()
        result = guard.check("how is my portfolio doing?")
        assert result.blocked is False
    
    def test_stock_recommendation_passes(self):
        guard = SafetyGuard()
        result = guard.check("what stocks should I buy?")
        assert result.blocked is False
    
    def test_empty_query_passes(self):
        guard = SafetyGuard()
        result = guard.check("")
        assert result.blocked is False


class TestResponseUniqueness:
    """Test that all blocked categories have distinct responses"""
    
    def test_all_five_responses_distinct(self):
        guard = SafetyGuard()
        
        test_queries = [
            ("I have insider info from a CEO", "insider_trading"),
            ("pump and dump scheme", "market_manipulation"),
            ("launder money", "money_laundering"),
            ("guarantees a 30% return", "guaranteed_returns"),
            ("take out a mortgage to invest all-in", "reckless_advice")
        ]
        
        responses = []
        for query, expected_category in test_queries:
            result = guard.check(query)
            assert result.blocked is True
            assert result.category == expected_category
            responses.append(result.response)
        
        # All 5 responses should be distinct
        assert len(set(responses)) == 5


class TestPerformance:
    """Test safety guard performance requirements"""
    
    def test_under_10ms(self):
        guard = SafetyGuard()
        start = time.perf_counter()
        guard.check("how is my portfolio?")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10
    
    def test_blocked_under_10ms(self):
        guard = SafetyGuard()
        start = time.perf_counter()
        guard.check("I have insider info from a CEO")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10
