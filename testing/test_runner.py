"""
Comprehensive test runner for the trading system
"""
import sys
import os
import unittest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from testing.unit_tests import TradingBotTests, run_unit_tests
from testing.stress_tester import StressTester


def run_all_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Trading Bot Test Suite")
    print("=" * 50)

    # Unit tests
    print("\n📋 Running Unit Tests...")
    run_unit_tests()

    # Performance tests would require actual trading engine instance
    print("\n⚡ Performance tests require live trading engine instance")
    print("   Run with: python -c 'from testing.test_runner import run_performance_tests; run_performance_tests(engine)'")

    print("\n✅ Test suite completed!")


def run_performance_tests(trading_engine):
    """Run performance and stress tests"""
    print("🔧 Running Performance Tests...")

    # Stress testing
    stress_tester = StressTester(trading_engine)
    stress_results = stress_tester.run_stress_tests()

    print("\n📊 Stress Test Results:")
    for test_name, result in stress_results.items():
        print(f"  {test_name}: {result}")

    # Start monitoring
    from monitoring import PerformanceMonitor
    monitor = PerformanceMonitor(trading_engine.db_manager, trading_engine.config)
    monitor.start_monitoring()

    print("\n📈 Performance monitoring started!")
    return monitor


if __name__ == "__main__":
    run_all_tests()