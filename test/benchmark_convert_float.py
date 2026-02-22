import time
import numpy as np
import torch
import tileon
import tileon.language as tl
from tileon.runtime.interpreter import _convert_float
from tileon._C import interpreter as _interpreter
from tileon._C import ir as _ir

TORCH_MIN_ACCURACY = 90.0

def get_torch_dtype(tl_dtype):
    try:
        name = tl_dtype.name
    except AttributeError:
        # Fallback if name is not available directly
        name = str(tl_dtype)
        
    if 'float32' in name or 'fp32' in name: return torch.float32
    if 'float16' in name or 'fp16' in name: return torch.float16
    if 'bfloat16' in name or 'bf16' in name: return torch.bfloat16
    if 'float64' in name or 'fp64' in name: return torch.float64
    
    # Check more specific names first to avoid partial matches
    if 'float8e4b8' in name or 'fp8e4b8' in name: return getattr(torch, 'float8_e4m3fnuz', None)
    if 'float8e5b16' in name or 'fp8e5b16' in name: return getattr(torch, 'float8_e5m2fnuz', None)
    if 'float8e4nv' in name or 'fp8e4nv' in name: return torch.float8_e4m3fn
    if 'float8e5' in name or 'fp8e5' in name: return torch.float8_e5m2
    
    return None

def _convert_float_original(input, input_dtype: tl.dtype, output_dtype: tl.dtype, rounding_mode):
    """
    Original Triton implementation of convert_float for comparison.
    """
    input_uint_dtype = getattr(np, f"uint{input_dtype.primitive_bitwidth}")
    output_unint_dtype = getattr(np, f"uint{output_dtype.primitive_bitwidth}")
    input_bin = np.frombuffer(input.tobytes(), dtype=input_uint_dtype)
    sign = (input_bin >> (input_dtype.primitive_bitwidth - 1)) & 0x01
    input_exponent_width = input_dtype.primitive_bitwidth - input_dtype.fp_mantissa_width - 1
    output_exponent_width = output_dtype.primitive_bitwidth - output_dtype.fp_mantissa_width - 1
    significand = input_bin & ((1 << input_dtype.fp_mantissa_width) - 1)
    bias_input = input_dtype.exponent_bias
    bias_output = output_dtype.exponent_bias
    exponent = ((input_bin >> input_dtype.fp_mantissa_width) & ((1 << input_exponent_width) - 1)).astype(np.int32)
    subnormal_index = exponent == 0
    if np.any(subnormal_index):
        # Credit to Phil: phil@openai.com
        # subnormal repr: ((-1.0)**sign) * (2.0**(1 - exp_bias)) * (2^(m0) + 2^(m1) + ... + 2^(mn))
        # where m0, m1, ..., mn are the 1-bit of the mantissa
        # convert it to normal repr: ((-1.0)**sign) * (2.0**(1 + m0 - exp_bias)) * (1 + 2^(m1 - m0) + ... + 2^(mn - m0))
        bit_pos = np.zeros_like(input_bin, dtype=np.int32)
        # Find the most significant bit of the mantissa in the significand
        for i in range(input_dtype.fp_mantissa_width):
            bit_index = ((significand >> i) & 0x01)
            # pos should be >= 1
            bit_pos[bit_index == 1] = input_dtype.fp_mantissa_width - i
        zero_significand_index = significand == 0
        exponent[subnormal_index] = 1 - bit_pos[subnormal_index]
        # 0 significand and subnormal should be treated as 0
        exponent[zero_significand_index & subnormal_index] = bias_input - bias_output
        significand[subnormal_index] = (significand[subnormal_index] << bit_pos[subnormal_index]) & (
            (1 << input_dtype.fp_mantissa_width) - 1)
    # Prevent overflow and underflow
    exponent_output = np.maximum(0, np.minimum((exponent - bias_input + bias_output), (1 << output_exponent_width) - 1))
    exponent_output = exponent_output.astype(output_unint_dtype)
    sign_output = sign.astype(output_unint_dtype)
    if input_dtype.primitive_bitwidth > output_dtype.primitive_bitwidth:  # Downcast
        significand_output = (significand >> (input_dtype.fp_mantissa_width - output_dtype.fp_mantissa_width)) & (
            (1 << output_dtype.fp_mantissa_width) - 1)
        if rounding_mode == _ir.ROUNDING_MODE.RTNE:  # Round to nearst even
            # find the cut-off bit
            cut_off = significand & (1 << (input_dtype.fp_mantissa_width - output_dtype.fp_mantissa_width - 1))
            significand_output = significand_output + (cut_off > 0)
        significand_output = significand_output.astype(output_unint_dtype)
    else:  # Upcast
        significand_output = (significand.astype(output_unint_dtype) <<
                              (output_dtype.fp_mantissa_width - input_dtype.fp_mantissa_width)) & (
                                  (1 << output_dtype.fp_mantissa_width) - 1)
    subnormal_index = exponent_output == 0
    if np.any(subnormal_index):  # underflow
        # normal repr: ((-1.0)**sign) * (2.0**(exp - exp_bias_input)) * (1 + 2^(m0) + 2^(m1) + ... + 2^(mn))
        # where m0, m1, ..., mn are the 1-bit of the mantissa
        # shift = (1 - exp_bias_output) - (exp - exp_bias_input)
        # convert it to subnormal repr: ((-1.0)**sign) * (2.0**(1 - exp_bias_output)) * (2^(-shift) + 2^(m0 - shift) + 2^(m1 - shift) + ... + 2^(mn - shift))
        exponent = ((input_bin >> input_dtype.fp_mantissa_width) & ((1 << input_exponent_width) - 1)).astype(np.int32)
        non_zero_exponent_index = exponent != 0
        # If the original exponent is not zero, we still need to shift the significand and consider the 1.0 part in mantissa
        subnormal_index = subnormal_index & non_zero_exponent_index
        shift = np.zeros_like(input_bin, dtype=np.int32)
        shift[subnormal_index] = (1 - bias_output) - (exponent[subnormal_index] - bias_input)
        significand_output[subnormal_index] = (significand_output[subnormal_index] >> shift[subnormal_index]) | (
            1 << (output_dtype.fp_mantissa_width - shift[subnormal_index]))
    output = (sign_output << (output_dtype.primitive_bitwidth - 1)) | (
        exponent_output << output_dtype.fp_mantissa_width) | significand_output
    return output.reshape(input.shape)

def get_dtypes_to_test():
    dtypes = []
    dtypes.append(tl.float32)
    dtypes.append(tl.float16)
    dtypes.append(tl.bfloat16)
    dtypes.append(tl.float64)
    
    for name in ['float8e5', 'float8e4nv', 'float8e4b8', 'float8e4b15', 'float8e5b16']:
        dtypes.append(getattr(tl, name))
            
    # Create pairs (src, dst)
    pairs = []
    for src in dtypes:
        for dst in dtypes:
            if src == dst: continue
            # Filter out incompatible conversions if any known constraints
            pairs.append((src, dst))
            
    return pairs

def analyze_mismatch(ref_flat, act_flat, dtype):
    """
    Analyze mismatches between reference and actual output.
    Returns a dict with error statistics.
    """
    total = ref_flat.size
    diff_mask = ref_flat != act_flat
    error_count = np.sum(diff_mask)
    
    if error_count == 0:
        return {'status': 'PASS', 'accuracy': 100.0}
        
    accuracy = 100.0 * (1.0 - (error_count / total))
    
    # Bit analysis
    width = dtype.primitive_bitwidth
    mantissa_width = dtype.fp_mantissa_width
    # exponent_width = width - mantissa_width - 1 (sign bit)
    
    diff_bits = ref_flat[diff_mask] ^ act_flat[diff_mask]
    
    sign_mask = 1 << (width - 1)
    mantissa_mask = (1 << mantissa_width) - 1
    exponent_mask = ((1 << width) - 1) ^ (sign_mask | mantissa_mask)
    
    sign_errors = np.sum((diff_bits & sign_mask) != 0)
    exponent_errors = np.sum((diff_bits & exponent_mask) != 0)
    mantissa_errors = np.sum((diff_bits & mantissa_mask) != 0)
    
    return {
        'status': f"FAIL ({error_count})",
        'accuracy': accuracy,
        'error_count': error_count,
        'sign_errors': sign_errors,
        'exponent_errors': exponent_errors,
        'mantissa_errors': mantissa_errors
    }

def benchmark_all():
    pairs = get_dtypes_to_test()
    print(f"Found {len(pairs)} conversion pairs to test.")
    
    results = []
    
    for src, dst in pairs:
        res = benchmark_pair(src, dst)
        if res:
            results.append(res)
            
    # Aggregation stats
    agg_stats = {
        'C++': {'time': 0.0, 'count': 0, 'acc_sum': 0.0, 'acc_count': 0, 'na_count': 0},
        'Py': {'time': 0.0, 'count': 0, 'acc_sum': 0.0, 'acc_count': 0, 'na_count': 0},
        'PyOrig': {'time': 0.0, 'count': 0, 'acc_sum': 0.0, 'acc_count': 0, 'na_count': 0},
        'Torch': {'time': 0.0, 'count': 0, 'acc_sum': 0.0, 'acc_count': 0, 'na_count': 0, 'low_acc_count': 0}
    }

    # Print results line by line
    print("\nBenchmark Results:")
    for res in results:
        in_type = res['Input']
        out_type = res['Output']
        
        print(f"[{in_type} -> {out_type}]")
        
        # C++
        cpp_time_str = res['C++ (s)']
        if cpp_time_str != 'nan':
            cpp_time = float(cpp_time_str)
            agg_stats['C++']['time'] += cpp_time
            agg_stats['C++']['count'] += 1
            agg_stats['C++']['acc_sum'] += 100.0
            agg_stats['C++']['acc_count'] += 1
            print(f" C++: {cpp_time_str}s, acc: 100.00%")
        else:
            agg_stats['C++']['na_count'] += 1
            print(f" C++: N/A")

        # Py
        py_time_str = res['Py (s)']
        py_spd = res['Speedup (C++/Py)']
        py_acc_str = "N/A"
        if py_time_str != 'nan':
            agg_stats['Py']['time'] += float(py_time_str)
            agg_stats['Py']['count'] += 1
            
            p_match = res.get('Py Match Stats')
            if p_match:
                acc = p_match['accuracy']
                agg_stats['Py']['acc_sum'] += acc
                agg_stats['Py']['acc_count'] += 1
                py_acc_str = f"{acc:.2f}%"
                if p_match['status'] != 'PASS':
                    py_acc_str += f" (Err: {p_match['error_count']})"
            
            print(f" Py: {py_time_str}s ({py_spd}), acc: {py_acc_str}")
        else:
             agg_stats['Py']['na_count'] += 1
             print(f" Py: N/A")

        # PyOrig
        py_orig_time_str = res['PyOrig (s)']
        py_orig_spd = res['Speedup (C++/PyOrig)']
        py_orig_acc_str = "N/A"
        if py_orig_time_str != 'nan':
            agg_stats['PyOrig']['time'] += float(py_orig_time_str)
            agg_stats['PyOrig']['count'] += 1
            
            po_match = res.get('PyOrig Match Stats')
            if po_match:
                acc = po_match['accuracy']
                agg_stats['PyOrig']['acc_sum'] += acc
                agg_stats['PyOrig']['acc_count'] += 1
                py_orig_acc_str = f"{acc:.2f}%"
                if po_match['status'] != 'PASS':
                    py_orig_acc_str += f" (Err: {po_match['error_count']})"
            elif res['PyOrig Match'] == 'PASS':
                 py_orig_acc_str = "100.00%"
            else:
                 py_orig_acc_str = res['PyOrig Match']
            
            print(f" PyOrig: {py_orig_time_str}s ({py_orig_spd}), acc: {py_orig_acc_str}")
        else:
            agg_stats['PyOrig']['na_count'] += 1
            print(f" PyOrig: N/A")

        # Torch
        torch_time_str = res['Torch (s)']
        torch_acc_str = "N/A"
        
        if res['Torch Error']:
             agg_stats['Torch']['na_count'] += 1
             print(f" Torch: N/A (Error: {res['Torch Error']})")
        elif torch_time_str != 'nan':
            agg_stats['Torch']['time'] += float(torch_time_str)
            agg_stats['Torch']['count'] += 1
            
            t_match = res.get('Torch Match Stats')
            if t_match:
                acc = t_match['accuracy']
                agg_stats['Torch']['acc_sum'] += acc
                agg_stats['Torch']['acc_count'] += 1
                torch_acc_str = f"{acc:.2f}%"
                if acc < TORCH_MIN_ACCURACY:
                    torch_acc_str += " [LOW]"
                    agg_stats['Torch']['low_acc_count'] += 1
                if t_match['status'] != 'PASS':
                    torch_acc_str += f" (Err: {t_match['error_count']})"
            elif res['Torch Match'] != 'N/A':
                 torch_acc_str = res['Torch Match']
            
            print(f" Torch: {torch_time_str}s, acc: {torch_acc_str}")
        else:
            agg_stats['Torch']['na_count'] += 1
            print(f" Torch: N/A")
            
        print("") # Empty line between cases

    # Print Summary
    print("-" * 55)
    print("Summary:")
    print(f"{'Backend':<10} | {'Avg Time (s)':<12} | {'Avg Accuracy':<12} | {'N/A Count':<10}")
    print("-" * 55)
    
    for backend in ['C++', 'Py', 'PyOrig', 'Torch']:
        stats = agg_stats[backend]
        avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0.0
        avg_acc = stats['acc_sum'] / stats['acc_count'] if stats['acc_count'] > 0 else 0.0
        na_count = stats['na_count']
        
        # PyOrig stats might be incomplete if we didn't update benchmark_pair
        if backend == 'PyOrig' and stats['acc_count'] == 0:
            acc_str = "N/A"
        else:
            acc_str = f"{avg_acc:.2f}%"
            
        print(f"{backend:<10} | {avg_time:<12.6f} | {acc_str:<12} | {na_count:<10}")

    torch_low_acc = agg_stats['Torch']['low_acc_count']
    if torch_low_acc > 0:
        print(f"Torch accuracy < {TORCH_MIN_ACCURACY:.0f}%: {torch_low_acc} cases")

def benchmark_pair(in_dtype, out_dtype, size=100000):
    # print(f"\nBenchmarking {in_dtype} -> {out_dtype} (size={size})")
    
    in_name = getattr(in_dtype, '__name__', str(in_dtype))
    out_name = getattr(out_dtype, '__name__', str(out_dtype))
    
    res_entry = {
        'Input': in_name,
        'Output': out_name,
        'Py (s)': None,
        'PyOrig (s)': None,
        'C++ (s)': None,
        'Torch (s)': None,
        'Speedup (C++/Py)': None,
        'Speedup (C++/PyOrig)': None,
        'Py Match': 'N/A',
        'PyOrig Match': 'N/A',
        'Torch Match': 'N/A',
        'Torch Error': None, # Added field
        'Note': ''
    }
    
    try:
        in_width = in_dtype.primitive_bitwidth
        out_width = out_dtype.primitive_bitwidth
        
        if in_width == 8:
            input_uint = np.random.randint(0, 256, size, dtype=np.uint8)
            input_data = input_uint.view(np.uint8)
        elif in_width == 16:
            input_uint = np.random.randint(0, 65536, size, dtype=np.uint16)
            input_data = input_uint.view(np.uint16)
        elif in_width == 32:
            input_uint = np.random.randint(0, 0xFFFFFFFF, size, dtype=np.uint32)
            input_data = input_uint.view(np.float32)
        elif in_width == 64:
            input_uint = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size, dtype=np.uint64)
            input_data = input_uint.view(np.float64)
        else:
            return res_entry

        input_uint64 = input_uint.astype(np.uint64)
        
        in_w = in_dtype.primitive_bitwidth
        in_m = in_dtype.fp_mantissa_width
        in_b = in_dtype.exponent_bias
        
        out_w = out_dtype.primitive_bitwidth
        out_m = out_dtype.fp_mantissa_width
        out_b = out_dtype.exponent_bias
        
        # Define output width for torch verification
        out_width = out_w

        rounding_mode = _ir.ROUNDING_MODE.RTNE
        
        # Benchmark Python
        start_time = time.perf_counter()
        try:
            res_py = _convert_float(input_data, in_dtype, out_dtype, rounding_mode)
            py_time = time.perf_counter() - start_time
            res_entry['Py (s)'] = f"{py_time:.6f}"
        except Exception:
            res_py = None
            py_time = 0
            res_entry['Py (s)'] = "nan"
        
        # Benchmark Python Original
        start_time = time.perf_counter()
        try:
            res_py_orig = _convert_float_original(input_data, in_dtype, out_dtype, rounding_mode)
            py_orig_time = time.perf_counter() - start_time
            res_entry['PyOrig (s)'] = f"{py_orig_time:.6f}"
        except Exception:
            res_py_orig = None
            py_orig_time = 0
            res_entry['PyOrig (s)'] = "nan"

        # Benchmark C++ Generic
        start_time = time.perf_counter()
        try:
            res_cpp = _interpreter.convert_float(input_uint64, in_w, in_m, in_b, out_w, out_m, out_b)
            cpp_time = time.perf_counter() - start_time
            res_entry['C++ (s)'] = f"{cpp_time:.6f}"
        except Exception:
            res_cpp = None
            cpp_time = 0
            res_entry['C++ (s)'] = "nan"

        # Benchmark C++ Specialized (only for float32 -> float16)
        cpp_spec_time = 0
        if in_dtype == tl.float32 and out_dtype == tl.float16:
            try:
                start_time = time.perf_counter()
                res_cpp_spec = _interpreter.convert_float32_to_float16(input_data)
                cpp_spec_time = time.perf_counter() - start_time
                res_entry['C++ Spec (s)'] = f"{cpp_spec_time:.6f}"
                
                # Verify Spec
                if res_cpp is not None:
                    res_cpp_flat = res_cpp.flatten()
                    res_cpp_spec_flat = res_cpp_spec.flatten().astype(np.uint64)
                    mismatch = np.sum(res_cpp_flat != res_cpp_spec_flat)
            except Exception:
                pass
        
        # Benchmark Torch
        torch_in_dtype = get_torch_dtype(in_dtype)
        torch_out_dtype = get_torch_dtype(out_dtype)
        res_torch = None
        torch_time = 0
        
        if torch_in_dtype and torch_out_dtype:
            try:
                # IMPORTANT: We must treat input bits as correct for the SOURCE type
                # input_data is a numpy array. 
                # If in_width is 8, it is uint8 view.
                # If in_width is 32, it is float32 view (which is standard IEEE754).
                
                # Torch doesn't support .view() from integer types to float types directly for all types
                # or vice versa in the same way numpy does sometimes.
                # We need to be careful about how we construct the tensor.
                
                # Step 1: Create tensor with integer bits
                if in_width == 8:
                    t_bits = torch.from_numpy(input_data.view(np.uint8))
                    t_in = t_bits.view(torch_in_dtype)
                elif in_width == 16:
                    t_bits = torch.from_numpy(input_data.view(np.int16)) # Torch uses int16 for half/bfloat storage view
                    t_in = t_bits.view(torch_in_dtype)
                elif in_width == 32:
                    t_bits = torch.from_numpy(input_data.view(np.int32))
                    t_in = t_bits.view(torch_in_dtype)
                elif in_width == 64:
                    t_bits = torch.from_numpy(input_data.view(np.int64))
                    t_in = t_bits.view(torch_in_dtype)
                else:
                    # Should not happen based on earlier checks
                    t_in = None

                if t_in is not None:
                    # Debug print
                    # print(f"DEBUG: t_in type: {type(t_in)}, torch_out_dtype: {torch_out_dtype}, type: {type(torch_out_dtype)}")
                    
                    start_time = time.perf_counter()
                    t_out = t_in.to(torch_out_dtype)
                    torch_time = time.perf_counter() - start_time
                    res_entry['Torch (s)'] = f"{torch_time:.6f}"
                    
                    if out_width == 8:
                        res_torch = t_out.view(torch.uint8).numpy()
                    elif out_width == 16:
                        res_torch = t_out.view(torch.int16).numpy().view(np.uint16)
                    elif out_width == 32:
                        res_torch = t_out.view(torch.int32).numpy().view(np.uint32)
                    elif out_width == 64:
                        res_torch = t_out.view(torch.int64).numpy().view(np.uint64)
            except Exception as e:
                # print(f"DEBUG EXCEPTION: {e}")
                # print(f"DEBUG: in_width={in_width}, torch_in_dtype={torch_in_dtype}, type={type(torch_in_dtype)}")
                res_entry['Torch (s)'] = "nan"
                res_entry['Torch Error'] = f"{e} (in_width={in_width}, torch_in={torch_in_dtype}, torch_out={torch_out_dtype})"
        else:
             res_entry['Torch (s)'] = "nan" # No dtype support
             if not torch_in_dtype:
                 res_entry['Torch Error'] = f"Unsupported input dtype: {in_dtype}"
             elif not torch_out_dtype:
                 res_entry['Torch Error'] = f"Unsupported output dtype: {out_dtype}"

        # Speedups
        if cpp_time > 0 and py_time > 0:
            res_entry['Speedup (C++/Py)'] = f"{py_time / cpp_time:.2f}x"
        if cpp_time > 0 and py_orig_time > 0:
            res_entry['Speedup (C++/PyOrig)'] = f"{py_orig_time / cpp_time:.2f}x"
            
        # Verification
        if res_cpp is not None:
            res_cpp_flat = res_cpp.flatten()
            
            if res_py is not None:
                res_py_flat = res_py.flatten().astype(np.uint64)
                
                # Use analyze_mismatch
                p_stats = analyze_mismatch(res_py_flat, res_cpp_flat, out_dtype)
                res_entry['Py Match Stats'] = p_stats
                res_entry['Py Match'] = p_stats['status']
                
            if res_py_orig is not None:
                res_py_orig_flat = res_py_orig.flatten().astype(np.uint64)
                
                # Use analyze_mismatch for PyOrig as well
                po_stats = analyze_mismatch(res_py_orig_flat, res_cpp_flat, out_dtype)
                res_entry['PyOrig Match Stats'] = po_stats
                res_entry['PyOrig Match'] = po_stats['status']
                
            if res_torch is not None:
                res_torch_flat = res_torch.flatten().astype(np.uint64)
                
                # Use analyze_mismatch
                t_stats = analyze_mismatch(res_torch_flat, res_cpp_flat, out_dtype)
                res_entry['Torch Match Stats'] = t_stats
                res_entry['Torch Match'] = t_stats['status']
        
        return res_entry
            
    except Exception:
        return res_entry

def benchmark_conversion():
    # This function was for initial specialized testing, merging it into main flow or skipping output
    pass

if __name__ == "__main__":
    # benchmark_conversion()
    benchmark_all()
