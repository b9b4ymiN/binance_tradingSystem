import time
import threading
import psutil
import gc

class StressTester:
    """Stress testing for trading system"""

    def __init__(self, trading_engine):
        self.trading_engine = trading_engine

    def run_stress_tests(self):
        """Run comprehensive stress tests"""
        results = {}

        print("Running stress tests...")

        # Test 1: High-frequency signal processing
        results['high_frequency'] = self._test_high_frequency_signals()

        # Test 2: Memory usage under load
        results['memory_stress'] = self._test_memory_usage()

        # Test 3: API failure simulation
        results['api_failure'] = self._test_api_failures()

        # Test 4: Concurrent trade processing
        results['concurrent_trades'] = self._test_concurrent_trades()

        return results

    def _test_high_frequency_signals(self):
        """Test system under high-frequency signal load"""
        start_time = time.time()
        signal_count = 0
        errors = 0

        try:
            for i in range(100):  # Send 100 signals rapidly
                signal = {
                    'action': 'buy' if i % 2 == 0 else 'sell',
                    'symbol': 'BTCUSDT',
                    'price': 50000 + (i * 10),
                    'strategy': 'stress_test'
                }

                try:
                    result = self.trading_engine.process_signal(signal)
                    signal_count += 1
                except Exception as e:
                    errors += 1

                time.sleep(0.01)  # 10ms between signals

            duration = time.time() - start_time

            return {
                'signals_processed': signal_count,
                'errors': errors,
                'duration': duration,
                'signals_per_second': signal_count / duration,
                'error_rate': errors / (signal_count + errors) if (signal_count + errors) > 0 else 0
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_memory_usage(self):
        """Test memory usage under sustained load"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Generate sustained load
        for i in range(1000):
            signal = {
                'action': 'buy',
                'symbol': f'TEST{i}USDT',
                'price': 1000 + i,
                'strategy': 'memory_test'
            }

            try:
                self.trading_engine.process_signal(signal)
            except:
                pass  # Ignore errors for memory test

        gc.collect()  # Force garbage collection
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory,
            'memory_increase_percent': ((final_memory - initial_memory) / initial_memory) * 100
        }

    def _test_api_failures(self):
        """Test system resilience to API failures"""
        # This would require mocking the API to simulate failures
        return {
            'test': 'api_failure_simulation',
            'status': 'requires_mock_implementation'
        }

    def _test_concurrent_trades(self):
        """Test concurrent trade processing"""
        results = {'successful': 0, 'failed': 0}

        def process_trade(trade_id):
            signal = {
                'action': 'buy',
                'symbol': 'BTCUSDT',
                'price': 50000 + trade_id,
                'strategy': f'concurrent_test_{trade_id}'
            }

            try:
                result = self.trading_engine.process_signal(signal)
                if 'error' not in result:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
            except:
                results['failed'] += 1

        # Create 10 concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_trade, args=(i,))
            threads.append(thread)

        start_time = time.time()

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        duration = time.time() - start_time

        return {
            'successful_trades': results['successful'],
            'failed_trades': results['failed'],
            'duration': duration,
            'trades_per_second': (results['successful'] + results['failed']) / duration
        }