"""Entropy audit module for IDIA.

This module provides tools to analyze and test the quality of entropy sources
used in cryptographic operations.
"""

import os
import time
import math
import struct
import hashlib
import statistics
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
from io import BytesIO
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from pathlib import Path

from idia.entropy import EntropySource, QuantumEntropySource, SystemEntropy, FileEntropySource
from idia.entropy_whitening import apply_full_whitening, xor_cascade, toeplitz_whitening_entry
from idia.qrng_adapter import QRNGAdapter, QRNG_AVAILABLE


class EntropyAuditor:
    """Analyzes and tests the quality of entropy sources."""
    
    # Standard NIST entropy tests
    ENTROPY_TESTS = {
        'frequency': 'Test for balanced 0s and 1s',
        'block_frequency': 'Test for balanced 0s and 1s within blocks',
        'runs': 'Test for oscillation patterns',
        'longest_run': 'Test for longest runs of 1s',
        'binary_matrix_rank': 'Test for linear dependence',
        'dft': 'Discrete Fourier Transform test (frequency domain)',
        'non_overlapping_template': 'Test for non-overlapping patterns',
        'overlapping_template': 'Test for overlapping patterns',
        'universal': 'Maurer\'s Universal Statistical test',
        'linear_complexity': 'Linear complexity profile',
        'serial': 'Serial test (overlapping patterns)',
        'approximate_entropy': 'Approximate entropy test',
        'cumulative_sums': 'Cumulative sums test',
        'random_excursions': 'Random excursions test',
        'random_excursions_variant': 'Random excursions variant test'
    }
    
    def __init__(self, source_spec: str = "eris", sample_size: int = 1024, num_samples: int = 10, chunk_size_samples: int = 100):
        """Initialize the entropy auditor with chunked processing."""
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.chunk_size_samples = max(1, chunk_size_samples) # Ensure at least 1
        self.audit_results = {}
        self.processed_data: Optional[bytes] = None # To store the final data for tests/viz
        self.matrix_cache: Dict[Any, np.ndarray] = {} # Initialize cache

        # Parse source_spec to separate base source and whitening
        parts = source_spec.split(":")
        self.base_source_name = parts[0].lower()
        self.whitening_level = "none"
        self.cascade_rounds = 4 # Default rounds
        if len(parts) > 1:
            self.whitening_level = parts[1].lower()
            if len(parts) > 2:
                try: self.cascade_rounds = int(parts[2])
                except ValueError: pass # Use default rounds

        # --- ADD Calculation for raw bytes needed per sample --- 
        self.whitening_factor = 1.0 # Default: no extra data needed
        if self.whitening_level in ["toeplitz", "full"]:
             self.whitening_factor = 1.5 # Estimate 50% extra data needed
             # Add a minimum floor? E.g. ensure at least 64 bytes raw if sample_size is tiny?
             # self.raw_bytes_to_request = max(int(self.sample_size * self.whitening_factor), 64)
             self.raw_bytes_to_request = int(self.sample_size * self.whitening_factor)
        else:
             self.raw_bytes_to_request = self.sample_size
        # Ensure it's at least the output size
        self.raw_bytes_to_request = max(self.raw_bytes_to_request, self.sample_size)
        # -----------------------------------------------------

        # Create ONLY the base source instance
        # We reuse the logic from get_entropy_source, but only for base creation
        print(f"Attempting to create base source: {self.base_source_name}") 
        # Map "eris" (and optionally "quantum") to QuantumEntropySource
        if self.base_source_name == "eris" or self.base_source_name == "quantum":
            print("Base source selected: QuantumEntropySource (via 'eris' or 'quantum')")
            try:
                self.base_source: EntropySource = QuantumEntropySource() # Use imported class
            except ImportError as e: 
                raise RuntimeError(f"Quantum source '{self.base_source_name}' required but failed: {e}")
            # No NameError check needed here, Python would fail earlier if import failed
        elif self.base_source_name == "system":
            print("Base source selected: SystemEntropy")
            self.base_source = SystemEntropy() # Use imported class
        elif Path(self.base_source_name).is_file():
             print(f"Base source selected: FileEntropySource({self.base_source_name})")
             self.base_source = FileEntropySource(self.base_source_name) # Use imported class
        else:
             # Default or unknown: Use 'eris' (Quantum)
             print(f"Unknown base source '{self.base_source_name}'. Defaulting to 'eris' (QuantumEntropySource).")
             try:
                 self.base_source = QuantumEntropySource() # Use imported class
             except ImportError as e:
                 raise RuntimeError(f"Default quantum source 'eris' required but failed: {e}")

        # Keep track of QRNG availability etc.
        self.qrng_available = QRNG_AVAILABLE
        self.has_scipy_stats = self._check_scipy_stats()
        
    def _check_scipy_stats(self) -> bool:
        """Check if scipy.stats is available for advanced tests."""
        try:
            import scipy.stats
            return True
        except ImportError:
            return False
    
    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data.
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Shannon entropy value (bits per byte)
        """
        # Count byte frequencies
        counts = {}
        for byte in data:
            counts[byte] = counts.get(byte, 0) + 1
        
        # Calculate Shannon entropy
        length = len(data)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def run_frequency_test(self, data: bytes) -> Dict[str, Any]:
        """Run the frequency (monobit) test on entropy data.
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Dictionary with test results
        """
        # Convert bytes to bits
        bits = ''.join(format(byte, '08b') for byte in data)
        
        # Count 1s and 0s
        ones = bits.count('1')
        zeros = bits.count('0')
        total = len(bits)
        
        # Calculate proportions
        prop_ones = ones / total
        prop_zeros = zeros / total
        
        # Calculate test statistic (standardized normal)
        s_obs = abs(ones - zeros) / math.sqrt(total)
        
        # Calculate p-value using complementary error function
        p_value = math.erfc(s_obs / math.sqrt(2))
        
        # Determine if test passes (p-value > 0.01)
        passed = p_value > 0.01
        
        return {
            'name': 'Frequency (Monobit) Test',
            'ones': ones,
            'zeros': zeros,
            'proportion_ones': prop_ones,
            'proportion_zeros': prop_zeros,
            'balance': 1.0 - abs(prop_ones - 0.5) / 0.5,  # 1.0 is perfect balance
            'statistic': s_obs,
            'p_value': p_value,
            'passed': passed
        }
    
    def run_runs_test(self, data: bytes) -> Dict[str, Any]:
        """Run the runs test on entropy data (aligned with NIST SP 800-22).
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Dictionary with test results
        """
        bits = ''.join(format(byte, '08b') for byte in data)
        n = len(bits)
        if n == 0:
            return {'name': 'Runs Test', 'error': 'No data', 'passed': False}

        # Calculate proportion (pi in NIST terms)
        ones = bits.count('1')
        pi = float(ones) / n

        # Prerequisite Test (NIST SP 800-22 Section 2.3.4)
        tau = 2.0 / math.sqrt(n)
        if abs(pi - 0.5) >= tau:
            return {
                'name': 'Runs Test',
                'proportion_ones': pi,
                'threshold_tau': tau,
                'statistic': None, # Test not applicable
                'p_value': 0.0, # Fail p-value as per NIST recommendation
                'passed': False,
                'notes': 'Proportion of ones is too far from 0.5'
            }

        # Count runs (V_n in NIST terms)
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Calculate expected number of runs (Using simpler common form)
        expected_runs = 2.0 * n * pi * (1.0 - pi) 

        # Calculate variance (Simpler, common approximation)
        variance = 2.0 * n * pi * (1.0 - pi) # Variance of total runs under H0
        
        # Check for edge cases where variance might be zero (only if pi is exactly 0 or 1)
        if variance <= 0:
             # This should only happen if pi is 0 or 1, which is caught by prerequisite
             # But handle defensively
             std_dev = 0.0 
        else:
            std_dev = math.sqrt(variance)
        
        # Calculate test statistic
        if std_dev == 0:
            statistic = float('inf') if runs != expected_runs else 0.0
        else:
            # Standard Z-score calculation for number of runs
            statistic = (runs - expected_runs) / std_dev 
            
        # Calculate p-value using complementary error function (erfc)
        # For a two-sided test using Z-score:
        p_value = math.erfc(abs(statistic) / math.sqrt(2.0)) 
        
        # Determine if test passes (p-value >= 0.01)
        passed = p_value >= 0.01
        
        return {
            'name': 'Runs Test',
            'proportion_ones': pi,
            'runs': runs,
            'expected_runs': expected_runs,
            'statistic': statistic,
            'p_value': p_value,
            'passed': passed
        }
    
    def run_serial_correlation_test(self, data: bytes) -> Dict[str, Any]:
        """Run serial correlation test to check for dependencies.
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Dictionary with test results
        """
        # Convert to values for correlation analysis
        values = list(data)
        n = len(values)
        
        if n < 2:
            return {
                'name': 'Serial Correlation Test',
                'correlation': 0,
                'p_value': 1.0,
                'passed': True,
                'error': 'Insufficient data'
            }
        
        # Calculate autocorrelation with lag 1
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        
        if variance == 0:
            return {
                'name': 'Serial Correlation Test',
                'correlation': 0,
                'p_value': 1.0,
                'passed': True,
                'error': 'Zero variance'
            }
        
        autocorr = 0
        for i in range(n - 1):
            autocorr += (values[i] - mean) * (values[i + 1] - mean)
        
        autocorr /= (n - 1) * variance
        
        # Calculate test statistic
        statistic = math.sqrt(n) * autocorr
        
        # Corrected p-value calculation (normal approximation, two-sided)
        p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
        
        # Determine if test passes
        passed = p_value > 0.01 # Keep original threshold
        
        return {
            'name': 'Serial Correlation Test',
            'correlation': autocorr,
            'statistic': statistic,
            'p_value': p_value,
            'passed': passed
        }
    
    def run_chi_square_test(self, data: bytes) -> Dict[str, Any]:
        """Run chi-square test for uniformity.
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Dictionary with test results
        """
        # Count byte frequencies
        counts = {}
        for byte in data:
            counts[byte] = counts.get(byte, 0) + 1
        
        n = len(data)
        expected = n / 256  # Expected count for each byte (0-255)
        
        # Calculate chi-square statistic
        chi_square = 0
        observed_categories = 0
        
        for byte in range(256):
            observed = counts.get(byte, 0)
            chi_square += (observed - expected) ** 2 / expected
            if observed > 0:
                observed_categories += 1
        
        # Degrees of freedom (observed categories - 1)
        df = max(1, observed_categories - 1)
        
        # Calculate p-value using complementary incomplete gamma function
        if self.has_scipy_stats:
            import scipy.stats
            p_value = 1 - scipy.stats.chi2.cdf(chi_square, df)
        else:
            # Approximate p-value using normal approximation
            z = math.sqrt(2 * chi_square) - math.sqrt(2 * df - 1)
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
        
        # Determine if test passes
        passed = 0.01 < p_value < 0.99  # Not too uniform, not too biased
        
        return {
            'name': 'Chi-Square Uniformity Test',
            'chi_square': chi_square,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'passed': passed,
            'unique_bytes': observed_categories
        }
    
    def analyze_bit_distribution(self, data: bytes) -> Dict[str, Any]:
        """Analyze the distribution of individual bits.
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Convert bytes to bits
        bits = ''.join(format(byte, '08b') for byte in data)
        
        # Count 1s in each position
        positions = [0] * 8
        for i, byte in enumerate(data):
            for pos in range(8):
                if (byte >> pos) & 1:
                    positions[pos] += 1
        
        # Calculate position biases
        n_bytes = len(data)
        position_biases = [(count / n_bytes) - 0.5 for count in positions]
        max_bias = max(abs(bias) for bias in position_biases)
        
        # Overall bit balance
        total_bits = len(bits)
        ones = bits.count('1')
        bit_balance = ones / total_bits
        
        return {
            'name': 'Bit Distribution Analysis',
            'bit_balance': bit_balance,
            'ideal': 0.5,
            'deviation': abs(bit_balance - 0.5),
            'position_biases': position_biases,
            'max_position_bias': max_bias,
            'passed': max_bias < 0.05  # Less than 5% bias in any position
        }
    
    def adaptive_entropy_test(self, data: bytes) -> Dict[str, Any]:
        """Run an adaptive entropy test that adjusts to quantum properties.
        
        Args:
            data: Bytes to analyze
            
        Returns:
            Dictionary with test results
        """
        # Standard entropy calculation
        entropy = 0.0 # Initialize before try
        try:
            calculated_entropy = self.calculate_entropy(data)
            if not isinstance(calculated_entropy, float):
                 logging.warning(f"calculate_entropy returned non-float: {type(calculated_entropy)}, value: {calculated_entropy}")
                 entropy = 0.0 # Fallback value
            else:
                 entropy = calculated_entropy # Assign if valid
        except Exception as e:
            logging.error(f"Error during self.calculate_entropy: {e}", exc_info=True)
            entropy = 0.0 # Ensure fallback value on error
            
        max_entropy = 8.0
        entropy_threshold = 7.8 if self.base_source_name.lower() in ("quantum", "eris") else 7.5
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

        # Analysis of byte patterns
        pattern_score = 0.0 # Initialize to numeric
        try: # Outer try for pattern scoring
            if self.qrng_available: # Keep check, but import locally if available
                score_func = None # Initialize score_func
                try: # Inner try for specific import and usage
                    # --- Use explicit import instead of generic 'import qrng' ---
                    from utils.lattice_framework.qrng import get_quantum_pattern_score
                    logging.debug("Successfully imported get_quantum_pattern_score from utils.lattice_framework.qrng")
                    score_func = get_quantum_pattern_score # Assign the imported function directly
                    # ------------------------------------------------------------
                    
                except ImportError as import_err:
                    logging.warning(f"Could not import get_quantum_pattern_score from project utils: {import_err}", exc_info=True)
                    # Optionally, could try importing qrng generically as a fallback here if needed
                    # import qrng 
                    # if hasattr(qrng, 'get_quantum_pattern_score'): score_func = qrng.get_quantum_pattern_score
                    # else: # Handle case where generic qrng also fails or lacks function
                    #     score_func = None 
                    # For now, just let score_func remain None if specific import fails.
                    
                except Exception as qe: # Catch other potential errors during import/assignment
                    logging.warning(f"Error setting up qrng score function: {qe}", exc_info=True)
                    score_func = None

                if score_func:
                    logging.debug(f"Calling specific {score_func.__name__}...")
                    # Passing entropy_score (float) instead of data (bytes)
                    logging.debug(f"Passing entropy_score to {score_func.__name__}: {entropy_score} (Type: {type(entropy_score)})")
                    calculated_pattern_score = score_func(entropy_score) 
                    logging.debug(f"Received calculated_pattern_score from {score_func.__name__}: {calculated_pattern_score} (Type: {type(calculated_pattern_score)})")
                    
                    if not isinstance(calculated_pattern_score, (int, float)):
                        logging.warning(f"Function {score_func.__name__} returned non-numeric score. Using fallback.")
                        pattern_score = -1 # Indicate failure for fallback logic
                    else:
                        pattern_score = calculated_pattern_score # Assign if valid
                else: 
                     # Handle case where score_func couldn't be assigned (import failed)
                     logging.warning("QRNG score function unavailable. Proceeding to fallback.")
                     pattern_score = -1 # Trigger fallback logic
            
            # Fallback or if qrng wasn't available/failed
            if not self.qrng_available or pattern_score == -1:
                logging.debug("Using basic pattern scoring fallback.")
                byte_pairs = [(data[i], data[i+1]) for i in range(0, len(data)-1, 2)]
                if not byte_pairs:
                    pattern_score = 0.0
                else:
                        # Ensure division is float division
                        calculated_pattern_score = float(len(set(byte_pairs))) / min(len(byte_pairs), 65536)
                        pattern_score = calculated_pattern_score # Assign here
                        logging.debug(f"Basic pattern score calculated: {pattern_score}")

        except Exception as pse: # Catch errors during outer pattern scoring block
            logging.error(f"Error during pattern score calculation: {pse}", exc_info=True)
            pattern_score = 0.0 # Ensure fallback on error

        # Quantum quality score
        quantum_quality = (entropy_score * 0.7) + (pattern_score * 0.3)

        # Determine pass/fail
        # Ensure comparisons are safe even if entropy/pattern_score are weird somehow (should be floats now)
        try:
            passed = (isinstance(entropy, float) and entropy > entropy_threshold) and \
                     (isinstance(pattern_score, float) and pattern_score > 0.8)
        except TypeError:
             passed = False # Comparison failed

        # --- Log values before returning --- 
        logging.debug(f"AdaptiveEntropyTest Results: entropy={entropy}({type(entropy)}), score={entropy_score}({type(entropy_score)}), pattern={pattern_score}({type(pattern_score)}), quality={quantum_quality}({type(quantum_quality)}), passed={passed}")
        # ------------------------------------
        
        return {
            'name': 'Adaptive Entropy Test',
            'entropy': entropy,
            'max_entropy': max_entropy,
            'entropy_score': entropy_score,
            'pattern_score': pattern_score,
            'quantum_quality': quantum_quality,
            'passed': passed
        }
    
    def apply_whitening_function(self, raw_sample_data: np.ndarray, common_seed: Optional[bytes] = None) -> np.ndarray:
        """
        Applies the configured whitening function to a single raw data sample.
        Passes down the common_seed if provided (for batch Toeplitz).
        """
        processed_sample_array: np.ndarray

        try:
            if self.whitening_level == "none":
                processed_sample_array = raw_sample_data[:self.sample_size] 
            elif self.whitening_level == "cascade":
                processed_bytes = xor_cascade(raw_sample_data.tobytes(), rounds=self.cascade_rounds)
                processed_sample_array = np.frombuffer(processed_bytes[:self.sample_size], dtype=np.uint8)
            elif self.whitening_level == "toeplitz":
                processed_sample_array = toeplitz_whitening_entry(
                    raw_sample_data, 
                    output_size=self.sample_size, # Pass explicit output size
                    matrix_cache=self.matrix_cache,
                    seed_override=common_seed # <--- Pass common_seed
                )
            elif self.whitening_level == "full":
                processed_sample_array = apply_full_whitening(
                    raw_sample_data,
                    output_size=self.sample_size,
                    cascade_rounds=self.cascade_rounds,
                    matrix_cache=self.matrix_cache,
                    seed_override=common_seed # <--- Pass common_seed
                )
            else:
                logging.warning(f"Unknown whitening level '{self.whitening_level}'. Applying no whitening.")
                processed_sample_array = raw_sample_data[:self.sample_size]

            # Ensure the output array has the expected dtype (uint8)
            if processed_sample_array.dtype != np.uint8:
                 processed_sample_array = processed_sample_array.astype(np.uint8)

            return processed_sample_array

        except Exception as e:
            # Log the specific error during whitening this sample
            logging.error(f"Exception in apply_whitening_function (level: {self.whitening_level}): {e}", exc_info=True)
            # Re-raise the exception to be caught by the run_all_tests loop
            raise e
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs all configured entropy tests on data obtained from the source.
        Applies whitening per sample after fetching raw data in chunks.
        
        Returns:
            A dictionary containing the results of all tests.
                {'test_name': {result_details...}}
        """
        if not self.base_source:
            self.audit_results = {"error": "Entropy source not initialized."}
            logging.error("Entropy source not initialized before running tests.")
            return self.audit_results

        total_samples_needed = self.num_samples
        all_processed_samples = [] # Collect individual whitened samples here
        bytes_processed = 0
        samples_processed = 0
        common_seed_for_batch = None # Initialize common seed

        logging.info(f"Starting entropy audit: {total_samples_needed} samples of size {self.sample_size} bytes each.")
        logging.info(f"Processing in chunks of {self.chunk_size_samples} samples.")
        logging.info(f"Whitening level: {self.whitening_level}, Raw bytes per sample: {self.raw_bytes_to_request}")

        try:
            # --- Determine Common Seed (if using Toeplitz/Full) --- 
            if self.whitening_level in ["toeplitz", "full"]:
                 logging.debug("Determining common seed for Toeplitz matrix batch...")
                 # Fetch the first chunk's worth of raw data to generate the seed
                 first_chunk_samples = min(self.chunk_size_samples, total_samples_needed)
                 raw_bytes_for_first_chunk = first_chunk_samples * self.raw_bytes_to_request
                 try:
                     logging.debug(f"Fetching {raw_bytes_for_first_chunk} bytes for initial seed generation...")
                     initial_raw_data_bytes = self.base_source.get_entropy(raw_bytes_for_first_chunk)
                     if len(initial_raw_data_bytes) < self.raw_bytes_to_request: # Need at least enough for one sample matrix base
                         raise ValueError(f"Received insufficient initial data ({len(initial_raw_data_bytes)} bytes) to generate common seed.")
                     common_seed_for_batch = hashlib.sha256(initial_raw_data_bytes).digest()
                     logging.info(f"Generated common seed for batch from initial {len(initial_raw_data_bytes)} bytes.")
                     # Optional: Pre-cache the matrix now? Could add latency upfront.
                     # self.apply_whitening_function(np.frombuffer(initial_raw_data_bytes[:self.raw_bytes_to_request], dtype=np.uint8), common_seed=common_seed_for_batch)
                 except Exception as e:
                      logging.error(f"Failed to get initial data or generate common seed: {e}. Audit cannot proceed with batch Toeplitz.", exc_info=True)
                      self.audit_results = {"error": f"Failed to initialize common seed: {e}"}
                      return self.audit_results
            # -----------------------------------------------------

            # --- Chunked Processing Loop --- 
            # Initialize timing variables before the loop if accumulating
            # total_whitening_time_sec = 0.0 # Needs to be initialized before loop

            for i in range(0, self.num_samples, self.chunk_size_samples):
                chunk_start_time = time.time() # <--- START OUTER CHUNK TIMER
                start_sample = i
                samples_in_chunk = min(self.chunk_size_samples, self.num_samples - start_sample)

                # Calculate raw bytes needed for the entire chunk using self.raw_bytes_to_request
                raw_bytes_needed_for_chunk = samples_in_chunk * self.raw_bytes_to_request
                logging.debug(f"Processing chunk {start_sample // self.chunk_size_samples + 1}: Samples {start_sample} to {start_sample + samples_in_chunk - 1}. Requesting {raw_bytes_needed_for_chunk} raw bytes.")

                # 1. Get Raw Data for the entire chunk
                try:
                    raw_chunk_data_bytes = self.base_source.get_entropy(raw_bytes_needed_for_chunk)
                    if raw_chunk_data_bytes is None or len(raw_chunk_data_bytes) < self.raw_bytes_to_request:
                        logging.error(f"Failed to retrieve sufficient raw data ({len(raw_chunk_data_bytes) if raw_chunk_data_bytes else 'None'} bytes) for chunk starting at sample {start_sample}. Expected at least {self.raw_bytes_to_request}. Aborting.")
                        self.audit_results['error'] = f"Insufficient raw data error in chunk starting at sample {start_sample}."
                        self._update_summary()
                        return self.audit_results 

                    raw_chunk_data_np = np.frombuffer(raw_chunk_data_bytes, dtype=np.uint8)
                    # total_raw_bytes_processed += len(raw_chunk_data_bytes)

                except Exception as e:
                     logging.error(f"Failed to get data from source for chunk starting at sample {start_sample}: {e}", exc_info=True)
                     self.audit_results['error'] = f"Data retrieval error in chunk starting at sample {start_sample}."
                     self._update_summary()
                     return self.audit_results 


                # 2. Whiten per sample within the fetched raw chunk
                whitening_total_chunk_time = 0.0 
                processed_samples_in_chunk = [] 
                for sample_idx_in_chunk in range(samples_in_chunk):
                    current_global_sample_num = start_sample + sample_idx_in_chunk

                    sample_start_byte = sample_idx_in_chunk * self.raw_bytes_to_request
                    sample_end_byte = sample_start_byte + self.raw_bytes_to_request

                    if sample_end_byte > len(raw_chunk_data_np):
                        logging.warning(f"Sample {current_global_sample_num}: Raw data slice end ({sample_end_byte}) exceeds available raw data ({len(raw_chunk_data_np)}). Skipping sample.")
                        continue 

                    raw_single_sample_data_np = raw_chunk_data_np[sample_start_byte:sample_end_byte]

                    sample_whitening_start_inner = time.time() 
                    try:
                        processed_single_sample_np = self.apply_whitening_function(
                            raw_single_sample_data_np,
                            common_seed=common_seed_for_batch
                        )

                        if processed_single_sample_np is None or len(processed_single_sample_np) != self.sample_size:
                             logging.warning(f"Sample {current_global_sample_num}: Whitening output size mismatch or None. Expected {self.sample_size}, got {len(processed_single_sample_np) if processed_single_sample_np is not None else 'None'}. Skipping.")
                             continue 

                        processed_samples_in_chunk.append(processed_single_sample_np.tobytes())
                        bytes_processed += self.sample_size
                        samples_processed += 1

                    except MemoryError as me:
                         logging.error(f"MemoryError during whitening of sample {current_global_sample_num}. Raw size: {len(raw_single_sample_data_np)} bytes.", exc_info=True)
                         raise me
                    except Exception as whitening_error:
                        logging.error(f"Error whitening sample {current_global_sample_num}: {whitening_error}", exc_info=True)
                        continue 
                    finally:
                         sample_whitening_end_inner = time.time()
                         whitening_total_chunk_time += (sample_whitening_end_inner - sample_whitening_start_inner)
                
                all_processed_samples.extend(processed_samples_in_chunk)
                # total_whitening_time_sec += whitening_total_chunk_time # Accumulate overall whitening time
                logging.debug(f"  Total time spent in whitening calls for chunk: {whitening_total_chunk_time:.4f} seconds.")

                # <--- CAPTURE END TIME AND LOG DURATION (for the whole outer chunk processing) --->
                chunk_end_time = time.time()
                chunk_duration = chunk_end_time - chunk_start_time
                total_chunks = (self.num_samples + self.chunk_size_samples - 1) // self.chunk_size_samples
                current_chunk_num = (start_sample // self.chunk_size_samples) + 1
                logging.info(f"Chunk {current_chunk_num}/{total_chunks} processed in {chunk_duration:.3f} seconds.")
                # <--------------------------------------------------------------------->
            # --- End Outer Chunked Processing Loop ---

            # Combine processed chunks
            final_data = b''.join(all_processed_samples)
            self.processed_data = final_data

            logging.info(f"Successfully processed {samples_processed} samples ({bytes_processed} bytes) after whitening.")
            logging.info("Running statistical tests on the combined whitened data...")

            # Run tests on the final concatenated data
            self.audit_results = {}
            test_functions = {
                "Frequency (Monobit)": self.run_frequency_test,
                "Runs": self.run_runs_test,
                "Serial Correlation": self.run_serial_correlation_test,
                "Chi-Square Uniformity": self.run_chi_square_test,
                "Adaptive Entropy": self.adaptive_entropy_test,
            }

            for name, test_func in test_functions.items():
                try:
                    logging.debug(f"Running test: {name}...")
                    result = test_func(final_data)
                    self.audit_results[name] = result
                    logging.debug(f"Result for {name}: {result}")
                    
                    # --- More robust p-value logging ---
                    p_value = result.get('p_value')
                    if isinstance(p_value, (int, float)):
                        p_value_str = f"{p_value:.4f}"
                    else:
                        # Handle other numeric keys for logging if no p_value
                        numeric_keys = ['entropy_score', 'quantum_quality', 'correlation', 'chi_square']
                        value_str = "N/A"
                        for key in numeric_keys:
                             val = result.get(key)
                             if isinstance(val, (int, float)):
                                 value_str = f"{key}={val:.4f}"
                                 break
                        p_value_str = value_str if p_value is None else str(p_value)
                        
                    logging.info(f"  {name}: {'Passed' if result.get('passed', False) else 'Failed'} ({p_value_str})")
                    # -----------------------------------
                    
                except Exception as e:
                    logging.error(f"Error running test '{name}': {e}", exc_info=True)
                    self.audit_results[name] = {"error": str(e), "passed": False}

            # Optionally run analysis functions
            try:
                self.audit_results["Bit Distribution"] = self.analyze_bit_distribution(final_data)
            except Exception as e:
                logging.error(f"Error running bit distribution analysis: {e}")
                self.audit_results["Bit Distribution"] = {"error": str(e)}

            logging.info("Entropy audit tests completed.")

        except Exception as e:
            logging.error(f"An error occurred during the entropy audit process: {e}", exc_info=True)
            self.audit_results = {"error": f"Audit failed: {e}"}
            # Ensure processed_data is cleared or handled if audit fails mid-way
            if 'self.processed_data' in locals(): del self.processed_data 

        # Try to generate summary even if some tests failed
        self._update_summary() 
        return self.audit_results

    def _update_summary(self):
        """Calculates and updates the summary part of the audit results."""
        if not hasattr(self, 'audit_results') or not isinstance(self.audit_results, dict):
             logging.error("Cannot update summary: audit_results not available or not a dict.")
             # Ensure summary exists even on error
             self.audit_results = self.audit_results or {}
             self.audit_results['summary'] = { 'error': 'Audit results missing for summary', 'passed': False, 'timestamp': time.time() }
             return

        # Filter out non-test results like 'error' or 'summary' itself before counting
        test_keys = [k for k in self.audit_results if isinstance(self.audit_results[k], dict) and 'passed' in self.audit_results[k]]
        passed_tests = sum(1 for k in test_keys if self.audit_results[k].get('passed', False))
        total_tests = len(test_keys)
        actual_data_len = len(self.processed_data) if hasattr(self, 'processed_data') and self.processed_data else 0 # Added check for None
        total_whitening_time = self.audit_results.get('summary', {}).get('whitening_time_sec', 0.0) 
        
        self.audit_results['summary'] = {
            'name': 'Entropy Audit Summary',
            'source': f"{self.base_source_name}:{self.whitening_level}",
            'sample_size_requested': self.sample_size,
            'num_samples_requested': self.num_samples,
            'chunk_size_samples_used': self.chunk_size_samples, 
            'total_bytes_processed': actual_data_len,
            'whitening_time_sec': total_whitening_time, 
            'tests_total': total_tests, # <-- USE 'tests_total' CONSISTENTLY
            'tests_passed': passed_tests,
            'pass_rate': (passed_tests / total_tests) if total_tests > 0 else 0,
            'overall_passed': (passed_tests / total_tests > 0.8) if total_tests > 0 else False, 
            'timestamp': time.time()
        }
        logging.info(f"Audit Summary: {passed_tests}/{total_tests} tests passed.")
    
    def visualize_entropy(self, output_path: Optional[str] = None) -> Optional[str]:
        """Create visualizations of the processed entropy batch."""
        if not hasattr(self, 'processed_data') or not self.processed_data:
            logging.error("Error: No processed data available for visualization. Run run_all_tests first.")
            return None
        
        combined_data = self.processed_data # Use the attribute
        logging.info(f"Visualizing {len(combined_data)} bytes...") # Changed from print
        try:
            data_array = np.frombuffer(combined_data, dtype=np.uint8)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Entropy Analysis: {self.base_source_name}:{self.whitening_level}', fontsize=16)
            
            # 1. Byte distribution histogram
            ax = axes[0, 0]
            counts, _, _ = ax.hist(data_array, bins=256, range=(-0.5, 255.5), alpha=0.7, color='blue') # Centered bins
            ax.set_title('Byte Distribution')
            ax.set_xlabel('Byte Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Calculate uniformity index
            try:
                # Use counts from histogram for efficiency
                non_zero_counts = counts[counts > 0]
                mean_count = np.mean(non_zero_counts) if len(non_zero_counts) > 0 else 0
                stdev_count = np.std(non_zero_counts) if len(non_zero_counts) > 0 else 0
                uniformity = 1.0 - (stdev_count / mean_count) if mean_count > 0 else 0.0
            except Exception as ue:
                 logging.warning(f"Could not calculate uniformity index: {ue}")
                 uniformity = 0.0
            ax.text(0.05, 0.95, f'Uniformity Idx: {uniformity:.4f}', # Renamed for clarity
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 2. Autocorrelation plot
            ax = axes[0, 1]
            max_lag = min(100, len(data_array) // 10)
            autocorr = np.array([]) # Initialize before if
            independence = 1.0 # Initialize before if
            if max_lag > 1:
                try:
                    autocorr = np.array([np.corrcoef(data_array[:-lag], data_array[lag:])[0, 1] 
                                       for lag in range(1, max_lag)])
                    ax.plot(range(1, max_lag), autocorr, '-o', markersize=3, alpha=0.7)
                    independence = 1.0 - np.mean(np.abs(autocorr))
                except Exception as ac_err:
                    logging.warning(f"Could not calculate/plot autocorrelation: {ac_err}")
                    ax.text(0.5, 0.5, 'Autocorrelation failed', ha='center', va='center')
                    autocorr = np.array([])
                    independence = 0.0 # Indicate failure
            else:
                ax.text(0.5, 0.5, 'Not enough data for autocorrelation', ha='center', va='center')
                 
            ax.set_title('Autocorrelation')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Correlation')
            ax.grid(True, alpha=0.3)
            ax.text(0.05, 0.95, f'Independence Idx: {independence:.4f}', # Renamed for clarity
                    transform=ax.transAxes, fontsize=10, # ... (bbox)
                    )
            
            # 3. Bit pattern visualization
            ax = axes[1, 0]
            # Ensure data_array is used here if it wasn't already
            data_reshaped = data_array[:len(data_array) - (len(data_array) % 32)].reshape(-1, 32)
            im = ax.imshow(data_reshaped, cmap='binary', aspect='auto', interpolation='none')
            ax.set_title('Bit Patterns')
            ax.set_xlabel('Bit Position')
            ax.set_ylabel('Sample')
            
            # 4. Spectral analysis
            ax = axes[1, 1]
            spectral_flatness = 0.0 # Initialize before try block
            try:
                fft_result = np.abs(np.fft.fft(data_array.astype(float) - np.mean(data_array))) # Use float for FFT
                frequencies = np.fft.fftfreq(len(data_array))
                # Plot positive frequencies only
                half_idx = len(frequencies)//2
                ax.plot(frequencies[1:half_idx], fft_result[1:half_idx], alpha=0.7) # Exclude DC component (index 0)
                ax.set_xscale('log')
                ax.set_yscale('log')
                # Calculate spectral flatness (avoid log(0))
                psd = fft_result[1:half_idx]**2
                # Check denominator is non-zero and handle potential NaN/inf
                mean_psd = np.mean(psd + 1e-10)
                if mean_psd > 1e-9: # Check mean_psd is reasonably positive
                    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / mean_psd 
                else:
                    spectral_flatness = 0.0 # Avoid division by zero or near-zero

            except Exception as spe:
                logging.warning(f"Could not calculate spectral analysis: {spe}")
                spectral_flatness = 0.0 # Ensure it's reset on error
                ax.text(0.5, 0.5, 'Spectral analysis failed', ha='center', va='center')

            ax.set_title('Spectral Analysis')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Magnitude')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Calculate and show spectral flatness
            ax.text(0.05, 0.95, f'Spectral Flatness: {spectral_flatness:.4f}', 
                    transform=ax.transAxes, fontsize=10, # ... (bbox)
                    )
            
            # --- Add overall quality score using summary --- 
            quality_score_text = "Audit Summary Unavailable"
            quality_score_color = 'lightgrey'
            if hasattr(self, 'audit_results') and 'summary' in self.audit_results:
                summary = self.audit_results['summary']
                tests_passed = summary.get('tests_passed', 'N/A')
                tests_total = summary.get('tests_total', 'N/A') # USE CORRECT KEY
                overall_passed = summary.get('overall_passed', False) # USE CORRECT KEY
                pass_rate = summary.get('pass_rate', 0.0)
                
                quality_score_text = (
                    f"Overall Result: {'PASSED' if overall_passed else 'FAILED'} | "
                    f"Score: {pass_rate*100:.1f}% " 
                    f"({tests_passed}/{tests_total} tests passed)"
                )
                quality_score_color = 'lightgreen' if overall_passed else 'lightcoral'
            else:
                 logging.warning("Audit summary not found for visualization text.")
            
            fig.text(0.5, 0.01, quality_score_text, ha='center', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor=quality_score_color, alpha=0.8))
            # ---------------------------------------------
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # --- Save Figure --- 
            save_path = output_path
            if not save_path:
                 # Create temp file path if none provided
                 temp_file = f"entropy_viz_{int(time.time())}.png"
                 save_path = temp_file
                 logging.debug(f"No output path provided for viz, saving to temp: {save_path}")
                 
            plt.savefig(save_path, dpi=150, bbox_inches='tight') # Lower DPI for speed/size
            plt.close(fig) # Correct indentation
            logging.debug(f"Visualization saved to {save_path}")
            return save_path # Return actual path used
            # -------------------
                
        except Exception as e:
            logging.error(f"Visualization error ({type(e).__name__}): {e}", exc_info=True)
            return None
    
    # +++ HELPER METHOD (Stricter Whitelist) +++
    def _simplify_results_for_yaml(self, results_dict):
        """Creates a strictly simplified version of results for clean YAML output using a whitelist."""
        if not isinstance(results_dict, dict):
            return results_dict

        # --- Whitelist for Keys to Keep (per test, plus summary) --- 
        # Define which keys are allowed in the final simplified output
        ALLOWED_KEYS = {
            # Top-level summary keys (within 'summary' dict originally)
            'summary': ['overall_passed', 'pass_rate', 'tests_passed', 'tests_total', 'total_bytes_processed'],
            # Per-test keys (within 'results' dict originally)
            'Frequency (Monobit)': ['passed', 'p_value', 'balance'], # Keep p-value?, Keep balance?
            'Runs': ['passed', 'p_value'], # Keep p-value?
            'Serial Correlation': ['passed', 'p_value', 'correlation'], # Keep p-value?, Keep correlation?
            'Chi-Square Uniformity': ['passed', 'p_value'], # Keep p-value?
            'Adaptive Entropy': ['passed', 'entropy_score', 'pattern_score', 'quantum_quality'], # Keep scores?
            'Bit Distribution': ['passed', 'bit_balance', 'max_position_bias'], # Keep balance/bias?
            # Add other test names if they exist
        }
        # ---------------------------------------------------------

        simplified = {}
        for key, value in results_dict.items():
            # --- Handle Top-Level Summary --- 
            if key == 'summary' and isinstance(value, dict):
                simplified_summary = {}
                allowed_summary_keys = ALLOWED_KEYS.get('summary', [])
                for sub_key, sub_value in value.items():
                    if sub_key in allowed_summary_keys:
                        # Apply basic type conversion even to summary
                        if isinstance(sub_value, (np.bool_, bool)):
                            simplified_summary[sub_key] = bool(sub_value)
                        elif isinstance(sub_value, np.integer):
                             simplified_summary[sub_key] = int(sub_value)
                        elif isinstance(sub_value, np.floating):
                             simplified_summary[sub_key] = float(sub_value)
                        elif isinstance(sub_value, (int, float, str, bool)):
                            simplified_summary[sub_key] = sub_value
                        # else: skip complex types in summary
                simplified[key] = simplified_summary
                continue # Move to next top-level key
                
            # --- Handle Test Results (Whitelist Approach) --- 
            # Check if the key is a known test name with a whitelist
            if key in ALLOWED_KEYS and isinstance(value, dict):
                simplified_test = {}
                allowed_test_keys = ALLOWED_KEYS[key]
                for sub_key, sub_value in value.items():
                    if sub_key in allowed_test_keys:
                        # Apply basic type conversion
                        if isinstance(sub_value, (np.bool_, bool)):
                             simplified_test[sub_key] = bool(sub_value)
                        elif isinstance(sub_value, np.integer):
                             simplified_test[sub_key] = int(sub_value)
                        elif isinstance(sub_value, np.floating):
                            if np.isnan(sub_value):
                                simplified_test[sub_key] = 'NaN'
                            elif np.isinf(sub_value):
                                simplified_test[sub_key] = 'Infinity' if sub_value > 0 else '-Infinity'
                            else:
                                simplified_test[sub_key] = float(sub_value)
                        elif isinstance(sub_value, (int, float, str, bool)):
                             simplified_test[sub_key] = sub_value
                        # else: skip complex/unwanted types within tests
                simplified[key] = simplified_test
            # --- Keep other top-level keys if needed? --- 
            # Example: Keep 'error' key if present at top level
            elif key == 'error' and isinstance(value, str):
                 simplified[key] = value
            # --- Discard everything else --- 
            else:
                # This key is not summary, not a test with a whitelist, or not 'error'
                # logging.debug(f"Discarding key '{key}' based on whitelist logic.")
                pass 

        return simplified
    # +++++++++++++++++++++++++++++++++++++++++++

    def generate_report(self, output_path: Optional[str] = None,
                        ) -> Dict[str, Any]:
        """Generate a comprehensive entropy audit report.
        Assumes run_all_tests has already been called and results are in self.audit_results.
        """
        if not hasattr(self, 'audit_results') or not self.audit_results or 'summary' not in self.audit_results:
             logging.error("Cannot generate report: audit_results/summary not found. Tests must be run first.")
             return {
                 'error': 'Audit results missing or incomplete',
                 'timestamp': time.time(),
                 'source': f"{self.base_source_name}:{self.whitening_level}"
             }

        # *** SIMPLIFY RESULTS BEFORE BUILDING REPORT ***
        # Use the stricter simplification method
        simplified_audit_results = self._simplify_results_for_yaml(self.audit_results)
        # *********************************************

        # --- Compile report dictionary base (using SIMPLIFIED results) --- 
        report = {
             'source': f"{self.base_source_name}:{self.whitening_level}",
             'timestamp': time.time(),
            'samples': {
                # Keep sample info for context
                'num_samples': self.num_samples,
                'sample_size': self.sample_size,
                'total_bytes': self.num_samples * self.sample_size
            },
             # *** Use the simplified results dict ***
             # The structure might change slightly based on simplification
             # Keep 'results' as the key containing simplified summary + tests
             'results': simplified_audit_results, 
             'visualization': None, # Placeholder
            'quantum_features': {
                'qrng_available': self.qrng_available,
                'import_path': QRNGAdapter.get_import_path() if self.qrng_available else None
            }
        }
        # ------------------------------------------------------------------------

        # --- Calculate overall entropy quality score --- 
        # (Calculation logic remains the same, but uses simplified_audit_results)
        quality_factors = []
        quality_score = 0.0
        logging.debug("Calculating quality score from strictly simplified results...")
        try:
            results_data = simplified_audit_results 
            freq_res = results_data.get('Frequency (Monobit)')
            adapt_res = results_data.get('Adaptive Entropy')
            bit_res = results_data.get('Bit Distribution')
            
            # Access only whitelisted keys (which should be simple types now)
            if freq_res and isinstance(freq_res.get('balance'), float): quality_factors.append(freq_res['balance'])
            if adapt_res and isinstance(adapt_res.get('entropy_score'), float): quality_factors.append(adapt_res['entropy_score'])
            if adapt_res and isinstance(adapt_res.get('pattern_score'), float): quality_factors.append(adapt_res['pattern_score'])
            if bit_res and isinstance(bit_res.get('bit_balance'), float): # Use bit_balance instead of deviation?
                 # Simpler factor: 1.0 - abs(balance - 0.5)*2
                 balance_factor = 1.0 - abs(bit_res['bit_balance'] - 0.5) * 2
                 quality_factors.append(max(0.0, balance_factor))

            if quality_factors:
                    quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            else:
                    logging.warning("No valid factors found for quality score calculation, score is 0.0")
                    quality_score = 0.0

        except Exception as e:
            logging.error(f"Error calculating quality score from strictly simplified results: {e}", exc_info=True)
            quality_score = 0.0

        # *** ADD Quality Score to the TOP LEVEL of the report ***
        report['quality_score'] = quality_score 
        # Add overall passed status directly to top level too for clarity?
        report['overall_passed'] = simplified_audit_results.get('summary', {}).get('overall_passed', False)
        # *******************************************************
        logging.debug(f"Final calculated report quality_score: {report['quality_score']}")
        # -----------------------------------------------

        # --- Saving logic (uses simplified `report` dict) --- 
        if output_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Prepare dict for saving (use the report we just built)
                yaml_report_to_save = {**report}
                # Note: viz path will be added/overwritten by caller before final save
                if 'visualization' in yaml_report_to_save: del yaml_report_to_save['visualization'] # Remove placeholder
                
                logging.debug(f"Saving intermediary report content (viz path added later) to: {output_path}")
                logging.debug(f"Quality score before intermediary save: {yaml_report_to_save.get('quality_score')}") 
                with open(output_path, 'w') as f:
                     # Use NoAliasDumper if available and needed for complex objects
                     try: 
                         yaml.dump(yaml_report_to_save, f, default_flow_style=False, sort_keys=False, Dumper=yaml.NoAliasDumper)
                     except AttributeError: # Catch if NoAliasDumper doesn't exist
                          yaml.dump(yaml_report_to_save, f, default_flow_style=False, sort_keys=False) # Fallback
                logging.debug(f"Intermediary YAML report saved.")
            except Exception as e:
                logging.error(f"Error saving report in generate_report: {e}", exc_info=True)
        # ------------------------------------------------------------------

        return report # Return the strictly simplified dictionary

def audit_entropy_source(source_spec: str = "eris", 
                        sample_size: int = 1024,
                        num_samples: int = 10,
                        chunk_size_samples: int = 100,
                        output_dir: Optional[str] = None,
                        visualize: bool = True,
                        extra_config: Optional[Dict] = None # Add config dict
                        ) -> Dict[str, Any]:
    """Run a comprehensive audit on an entropy source using chunked processing."""
    print("Initializing Entropy Auditor...")
    # Pass chunk size and potentially other config to constructor?
    # For now, only pass needed args.
    auditor = EntropyAuditor(source_spec, sample_size, num_samples, chunk_size_samples)

    # --- Run Tests ONCE --- 
    auditor.run_all_tests() # Populates auditor.audit_results, including summary
    # Check if tests actually ran successfully
    if 'summary' not in auditor.audit_results:
         logging.error("Audit failed during test execution. Cannot proceed.")
         # Return the partial/error results
         return auditor.audit_results 
    # ---------------------
    logging.debug(f"Audit results after run_all_tests: {auditor.audit_results}")

    # --- Generate Visualization (if requested) --- 
    viz_path_final = None
    if visualize:
        viz_output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Use timestamp from summary for consistency
            timestamp = int(auditor.audit_results.get('summary',{}).get('timestamp', time.time()))
            viz_name = f"entropy_viz_{timestamp}.png"
            viz_output_path = os.path.join(output_dir, viz_name)
        
        logging.debug(f"Calling visualize_entropy with path: {viz_output_path}")
        viz_path_final = auditor.visualize_entropy(viz_output_path) # Pass path or None
        if viz_path_final:
             logging.info(f"Visualization generation successful: {viz_path_final}")
        else:
             logging.error("Visualization failed to generate.")
    # -------------------------------------------

    # --- Generate Report Dictionary (using existing results) --- 
    report_path = None
    if output_dir:
        timestamp = int(auditor.audit_results.get('summary',{}).get('timestamp', time.time()))
        report_path = os.path.join(output_dir, f"entropy_audit_{timestamp}.yaml")

    # Call generate_report, gets results+score (it no longer saves internally if path given)
    report_dict = auditor.generate_report(output_path=None) # Generate dict only
    
    # Manually add the final viz path to the report dict
    report_dict['visualization'] = viz_path_final 
    logging.debug(f"Final report dictionary generated. Quality score: {report_dict.get('quality_score')}")
    # -----------------------------------------------------------

    # --- Save Final Report --- 
    if report_path:
         try:
             # Ensure directory exists (redundant if already created for viz, but safe)
             os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

             yaml_report_to_save = {**report_dict} # Use the final dict
             # Adjust viz path for saving just the basename
             if viz_path_final: 
                  yaml_report_to_save['visualization'] = os.path.basename(viz_path_final)
                  
             # Ensure quality score is present (fallback)
             if 'quality_score' not in yaml_report_to_save:
                 logging.warning("Quality score missing before final save, adding 0.0")
                 yaml_report_to_save['quality_score'] = 0.0
             
             logging.debug(f"Attempting to save final report dict to: {report_path}") # Changed wording
             # Log the type of keys and some values to check for weird data
             logging.debug(f"Report keys type: {[type(k) for k in yaml_report_to_save.keys()]}")
             if 'results' in yaml_report_to_save and isinstance(yaml_report_to_save['results'], dict):
                 logging.debug(f"Result keys type: {[type(k) for k in yaml_report_to_save['results'].keys()]}")
             
             with open(report_path, 'w') as f:
                  logging.debug(f"File {report_path} opened successfully for writing.") # More specific log
                  try: 
                      # Use NoAliasDumper for cleaner output if available
                      yaml.dump(yaml_report_to_save, f, default_flow_style=False, sort_keys=False, Dumper=yaml.NoAliasDumper)
                      logging.debug("yaml.dump (NoAliasDumper) executed.") # Log after call
                  except AttributeError: # Catch if NoAliasDumper doesn't exist
                       yaml.dump(yaml_report_to_save, f, default_flow_style=False, sort_keys=False)
                       logging.debug("yaml.dump (default) executed.") # Log after call
                  logging.debug(f"File {report_path} stream closing.") # Log before with block ends
             # This log should appear ONLY if the 'with open' block completes without error
             logging.info(f"Full report successfully saved to: {report_path}") # Added 'successfully'
         except Exception as e:
              # This log appears if any exception occurs within the outer try block
              logging.error(f"Error saving final report to {report_path}: {e}", exc_info=True)
              # Print directly to console as well, in case logging fails
              print(f"\n*** CRITICAL ERROR SAVING REPORT to {report_path}: {type(e).__name__} - {e} ***\n") 
    # -------------------------

    # --- Print Summary --- 
    summary = report_dict.get('results', {}).get('summary') # Get summary from final dict results
    if summary:
         print(f"\n=== Entropy Audit Summary: {summary.get('source','N/A')} ===")
         print(f"Tests Passed: {summary.get('tests_passed','N/A')}/{summary.get('tests_total','N/A')} ({summary.get('pass_rate',0)*100:.1f}%)")
         quality_score = report_dict.get('quality_score', 0.0) # Use score from final dict 
         print(f"Overall Quality Score: {quality_score:.4f}")

         if quality_score > 0.9: print("Assessment: EXCELLENT")
         elif quality_score > 0.8: print("Assessment: GOOD")
         elif quality_score > 0.7: print("Assessment: ADEQUATE")
         else: print("Assessment: POOR")
         
         print(f"Total Whitening Time: {summary.get('whitening_time_sec', 0.0):.2f} seconds") # Add this back?
         if viz_path_final: print(f"Visualization path: {viz_path_final}")
         if report_path: print(f"Report path: {report_path}")
    else:
         print("ERROR: Audit failed to produce a summary.")
    # ---------------------

    return report_dict # Return the final dictionary


if __name__ == "__main__":
    import argparse

    # Set up parser first to get log level
    parser = argparse.ArgumentParser(description="Run entropy audit on a specified source.")
    parser.add_argument("--source", type=str, default="eris", 
                        help='Entropy source specification (e.g., "eris", "eris:full", "system", path/to/file)')
    parser.add_argument("--sample-size", type=int, default=1024,
                        help="Size of each entropy sample in bytes (default: 1024)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of entropy samples to collect (default: 10)")
    parser.add_argument("--chunk-size-samples", type=int, default=100,
                        help="Number of samples to process per whitening chunk (default: 100)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the report and visualization (optional)")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate and save visualizations (default: True)")
    parser.add_argument("--analyze", action="store_true", 
                        help="Analyze existing data/report (Behavior TBD - Not implemented)")
    # --- ADD LOG LEVEL ARGUMENT --- 
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    # -----------------------------

    args = parser.parse_args()

    # --- Configure Logging BASED ON ARG --- 
    log_level_str = args.log_level.upper()
    log_level_int = getattr(logging, log_level_str, logging.INFO) # Default to INFO if getattr fails
    logging.basicConfig(level=log_level_int, format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')
    # -------------------------------------
    
    print(f"Starting Entropy Audit Script...")
    # Log arguments using logging instead of print after setup
    logging.info(f"Arguments: source={args.source}, sample_size={args.sample_size}, num_samples={args.num_samples}, "
                 f"chunk_size_samples={args.chunk_size_samples}, "
                 f"output_dir={args.output_dir}, visualize={args.visualize}, analyze={args.analyze}, "
                 f"log_level={args.log_level}")


    if args.analyze:
        print("\n--analyze flag detected.")
        print("Current script implementation primarily focuses on generating and testing data.")
        print("The exact behavior for --analyze needs clarification.")
        print("\nExiting without performing analysis based on --analyze flag.")
    else:
        print(f"\nRunning entropy audit for source: {args.source}")
        # Call the main audit function if --analyze is not specified
        final_report = audit_entropy_source(
            source_spec=args.source,
            sample_size=args.sample_size,
            num_samples=args.num_samples,
            chunk_size_samples=args.chunk_size_samples, # Pass arg
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        print("\nAudit process finished.")

        # --- ADD THESE LINES --- 
        print(f"\n>>> FINAL REPORT QUALITY SCORE: {final_report.get('quality_score', 'Not Found')}\n")
        summary = final_report.get('results', {}).get('summary')
        print(f">>> FINAL REPORT SUMMARY PASSED: {summary.get('overall_passed', 'Not Found')}\n")
        # -----------------------

# End of script 