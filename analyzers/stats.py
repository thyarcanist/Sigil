import math
from collections import Counter
from scipy.stats import chisquare, norm

def frequency_monobit_test(data_bytes: bytes) -> dict:
    """Performs the NIST Frequency (Monobit) Test.

    Checks if the number of ones and zeros in the sequence are approximately
    the same, as would be expected for a random sequence.

    Args:
        data_bytes: The sequence of bytes to test.

    Returns:
        A dictionary containing the test statistic (s_obs), p-value,
        number of zeros, number of ones, and a boolean indicating if the
        test passed (p-value >= 0.01).
    """
    n = len(data_bytes) * 8
    if n == 0:
        return {'s_obs': 0, 'p_value': 1.0, 'n_zeros': 0, 'n_ones': 0, 'passed': True, 'notes': 'Empty sequence'}

    # Count ones and zeros
    n_ones = 0
    for byte in data_bytes:
        n_ones += bin(byte).count('1')
    n_zeros = n - n_ones

    # Calculate S_n (sum of +1/-1 bits)
    s_n = n_ones - n_zeros

    # Calculate test statistic s_obs
    s_obs = abs(s_n) / math.sqrt(n)

    # Calculate p-value using complementary error function (erfc)
    p_value = math.erfc(s_obs / math.sqrt(2))

    # Determine pass/fail (typically alpha = 0.01)
    passed = p_value >= 0.01

    return {
        's_obs': s_obs,
        'p_value': p_value,
        'n_zeros': n_zeros,
        'n_ones': n_ones,
        'passed': passed,
        'notes': f'Bits: {n}, Zeros: {n_zeros}, Ones: {n_ones}'
    }

def chi_square_byte_distribution_test(data_bytes: bytes) -> dict:
    """Performs a Chi-Square goodness-of-fit test on the byte distribution.

    Tests if the observed frequencies of byte values (0-255) match the
    expected frequencies of a uniform distribution.

    Args:
        data_bytes: The sequence of bytes to test.

    Returns:
        A dictionary containing the chi-square statistic, p-value,
        degrees of freedom, and a boolean indicating if the test passed
        (p-value >= 0.01).
    """
    n = len(data_bytes)
    if n < 256: # Need sufficient data for a meaningful test across all byte values
        return {
            'chisq_stat': None,
            'p_value': None,
            'dof': 255,
            'passed': None,
            'notes': f'Insufficient data (need >= 256 bytes, got {n})'
        }

    # Count observed frequencies for each byte value
    observed_frequencies = Counter(data_bytes)
    
    # Create a list of observed counts for all 256 possible byte values
    observed = [observed_frequencies.get(i, 0) for i in range(256)]

    # Calculate expected frequencies for a uniform distribution
    expected_frequency = n / 256.0
    expected = [expected_frequency] * 256

    # Perform the Chi-Square test
    try:
        chisq_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        dof = 255 # Degrees of freedom = k - 1, where k = 256 possible byte values
        passed = p_value >= 0.01 # Standard alpha level
        notes = f'Bytes: {n}'
    except ValueError as e:
         # Handle cases where chisquare might fail (e.g., expected freq too low, though unlikely here)
        chisq_stat = None
        p_value = None
        dof = 255
        passed = None
        notes = f'ChiSquare calculation error: {e}'

    return {
        'chisq_stat': chisq_stat,
        'p_value': p_value,
        'dof': dof,
        'passed': passed,
        'notes': notes
    }

def runs_test(data_bytes: bytes) -> dict:
    """Performs the NIST Runs Test.

    Checks for oscillations between 0s and 1s that are too fast or too slow.
    Requires the Frequency (Monobit) test to be passed first (proportion of ones > threshold).

    Args:
        data_bytes: The sequence of bytes to test.

    Returns:
        A dictionary containing the test statistic (V_n_obs), p-value,
        and a boolean indicating if the test passed (p-value >= 0.01).
    """
    n_bytes = len(data_bytes)
    n_bits = n_bytes * 8
    if n_bits < 100: # NIST requires at least 100 bits
        return {
            'v_n_obs': None,
            'p_value': None,
            'passed': None,
            'notes': f'Insufficient data (need >= 100 bits, got {n_bits})'
        }

    # --- Prerequisite: Frequency Test --- 
    # Calculate proportion of ones (pi)
    n_ones = 0
    bits = []
    for byte in data_bytes:
        for i in range(8):
            bit = (byte >> (7-i)) & 1
            bits.append(bit)
            if bit == 1:
                n_ones += 1
    
    pi = n_ones / n_bits
    # Threshold tau (typically sqrt(2/n)) - NIST suggests checking if pi is within bounds first.
    # A common check is |pi - 0.5| >= 2/sqrt(n). If true, Runs test is inconclusive.
    tau = 2.0 / math.sqrt(n_bits)
    if abs(pi - 0.5) >= tau:
        return {
            'v_n_obs': None,
            'p_value': None,
            'passed': None,
            'notes': f'Frequency test prerequisite failed (|pi - 0.5| = {abs(pi - 0.5):.4f} >= tau = {tau:.4f})'
        }
    # ------------------------------------

    # Calculate V_n_obs (Total number of runs)
    v_n_obs = 1 # Start with 1 run
    for i in range(n_bits - 1):
        if bits[i] != bits[i+1]:
            v_n_obs += 1

    # Calculate p-value
    # Formula: p_value = erfc(|V_n_obs - 2*n*pi*(1-pi)| / (2*sqrt(2*n)*pi*(1-pi)))
    numerator = abs(v_n_obs - 2.0 * n_bits * pi * (1.0 - pi))
    denominator = 2.0 * math.sqrt(2.0 * n_bits) * pi * (1.0 - pi)

    if denominator < 1e-10: # Avoid division by zero if pi is very close to 0 or 1 (already checked by tau)
        p_value = 0.0 # Indicates extreme deviation
    else:
        p_value = math.erfc(numerator / denominator)

    # Determine pass/fail
    passed = p_value >= 0.01

    return {
        'v_n_obs': v_n_obs,
        'p_value': p_value,
        'passed': passed,
        'notes': f'Bits: {n_bits}, pi: {pi:.4f}, Runs: {v_n_obs}'
    }

# TODO: Implement other relevant statistical tests (e.g., Serial, Poker, etc.)


# Example usage (if run directly):
if __name__ == '__main__':
    import os

    print("Testing Frequency (Monobit) Test...")
    
    # Test 1: Highly biased data (mostly zeros)
    biased_data_zeros = bytes([0] * 1000)
    result_zeros = frequency_monobit_test(biased_data_zeros)
    print(f"Biased (Zeros): {result_zeros}")

    # Test 2: Highly biased data (mostly ones)
    biased_data_ones = bytes([255] * 1000)
    result_ones = frequency_monobit_test(biased_data_ones)
    print(f"Biased (Ones): {result_ones}")

    # Test 3: Pseudo-random data (should pass)
    prng_data = os.urandom(10000) # Larger sample
    result_prng = frequency_monobit_test(prng_data)
    print(f"PRNG Data: {result_prng}")

    # Test 4: Short data
    short_data = os.urandom(10)
    result_short = frequency_monobit_test(short_data)
    print(f"Short Data: {result_short}")

    # Test 5: Empty data
    empty_data = b''
    result_empty = frequency_monobit_test(empty_data)
    print(f"Empty Data: {result_empty}")

    print("\nTesting Chi-Square Byte Distribution Test...")
    # Test 1: Biased data (zeros)
    result_chisq_zeros = chi_square_byte_distribution_test(biased_data_zeros)
    print(f"Biased (Zeros): {result_chisq_zeros}")

    # Test 2: Biased data (ones)
    result_chisq_ones = chi_square_byte_distribution_test(biased_data_ones)
    print(f"Biased (Ones): {result_chisq_ones}")

    # Test 3: PRNG data (should pass)
    result_chisq_prng = chi_square_byte_distribution_test(prng_data)
    print(f"PRNG Data: {result_chisq_prng}")

    # Test 4: Short data (should return None)
    result_chisq_short = chi_square_byte_distribution_test(short_data)
    print(f"Short Data: {result_chisq_short}")

    # Test 5: Empty data (should return None)
    result_chisq_empty = chi_square_byte_distribution_test(empty_data)
    print(f"Empty Data: {result_chisq_empty}")

    print("\nTesting Runs Test...")
    # Test 1: Highly biased data (zeros) - Should fail prerequisite
    result_runs_zeros = runs_test(biased_data_zeros)
    print(f"Biased (Zeros): {result_runs_zeros}")

    # Test 2: Highly biased data (ones) - Should fail prerequisite
    result_runs_ones = runs_test(biased_data_ones)
    print(f"Biased (Ones): {result_runs_ones}")

    # Test 3: Alternating bits (0101...) - Should fail (too many runs)
    alternating_bytes = bytes([0xAA] * 1000) # 10101010...
    result_runs_alt = runs_test(alternating_bytes)
    print(f"Alternating Bits: {result_runs_alt}")

    # Test 4: Solid runs (00...0011...11) - Should fail (too few runs)
    # Create 500 bytes of 0, 500 bytes of 1
    solid_runs_bytes = bytes([0]*500 + [255]*500)
    result_runs_solid = runs_test(solid_runs_bytes)
    print(f"Solid Runs: {result_runs_solid}")

    # Test 5: PRNG data (should pass)
    result_runs_prng = runs_test(prng_data) # Use the same 10k PRNG data
    print(f"PRNG Data: {result_runs_prng}")

    # Test 6: Short data (should return None)
    result_runs_short = runs_test(short_data)
    print(f"Short Data: {result_runs_short}")

    # Test 7: Empty data (should return None)
    result_runs_empty = runs_test(empty_data)
    print(f"Empty Data: {result_runs_empty}") 