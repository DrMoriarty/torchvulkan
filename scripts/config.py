"""
Which shader variants are compiled ahead of time.

Each Kernel names a Slang module and its generic entry point; each Entry lists
the dtypes to specialise it for. One variant is built as:

    slangc ops/binary.slang -entry binary_main -specialize float32_t -specialize 4

The host looks a variant up by ShaderKey{Kernel::BINARY, dtype}. To add a
kernel, write its module in src/shaders/ops/ and add a Kernel here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name:  str   # short name; part of the generated .spv filename
    slang: str   # the Slang type the entry point is specialised with
    bytes: int   # size in bytes
    aten:  tuple # the c10::ScalarType values this variant serves


U64 = DType('u64', 'uint64_t', 8, ('UInt64',))
U32 = DType('u32', 'uint32_t', 4, ('UInt32',))
U16 = DType('u16', 'uint16_t', 2, ('UInt16',))
U8  = DType('u8',  'uint8_t',  1, ('Byte', 'Bool'))

I64 = DType('i64', 'int64_t', 8, ('Long',))
I32 = DType('i32', 'int32_t', 4, ('Int',))
I16 = DType('i16', 'int16_t', 2, ('Short',))
I8  = DType('i8',  'int8_t',  1, ('Char',))

F64 = DType('f64', 'float64_t', 8, ('Double',))
F32 = DType('f32', 'float32_t', 4, ('Float',))
F16 = DType('f16', 'float16_t', 2, ('Half',))

BF16 = DType('bf16', 'BFloat16', 2, ('BFloat16',))

UNSIGNED_INTEGERS = [U64, U32, U16, U8]
SIGNED_INTEGERS = [I64, I32, I16, I8]
INTEGERS = UNSIGNED_INTEGERS + SIGNED_INTEGERS
BUILTIN_FLOATS = [F64, F32, F16]
FLOATS = BUILTIN_FLOATS + [BF16]
DTYPES = INTEGERS + FLOATS

BYTE_WIDTHS = [
    DType('16', 'uint4',    16, ('ComplexDouble',)),
    DType('8',  'uint64_t', 8,  ('UInt64',)),
    DType('4',  'uint32_t', 4,  ('UInt32',)),
    DType('2',  'uint16_t', 2,  ('UInt16',)),
    DType('1',  'uint8_t',  1,  ('Byte',)),
]


def vec_size(nbytes):
    """
    How many elements a thread handles per step: as many as fit in 128 bits,
    capped at 4. The host reads it from VEC_SIZE_FOR_BYTES in shader_kernels.h.
    """
    return int(min(16 / nbytes, 4))


# The integer arguments an entry point takes after its type arguments, derived
# from the dtype's size. For a pairwise entry point they come from the wider of
# the two dtypes, since both tensors are stepped by the same amount.
VALUE_ARGS = {
    'vec': vec_size,
    'wide_vec': lambda nbytes: int(min(16 / nbytes, 8)),  # full 128-bit loads
}


@dataclass(frozen=True)
class Entry:
    function: str          # the entrypoint in the kernel's module
    dtypes: list           # the dtypes to specialise it for
    value_args: tuple = () # names from VALUE_ARGS, in the order the entry point declares them
    constants: tuple = ()  # literal integer arguments that follow the value args
    pairwise: bool = False # specialise over every (src, dst) pair rather than one dtype


@dataclass(frozen=True)
class Kernel:
    name: str
    module: str # path relative to src/shaders
    entries: list 


KERNELS = [
    Kernel('binary', 'ops/binary.slang', [
        Entry('binary_main', DTYPES, ('vec',)),
    ]),
    Kernel('unary', 'ops/unary.slang', [
        Entry('unary_main', DTYPES, ('vec',)),
    ]),
    # wide fp16 pointwise variants (A1, training_speedup.md): full 128-bit
    # uint4 loads/stores = 8 halfs per access instead of VEC_SIZE scalar
    # 2-byte ones; the host routes only contiguous, 16-byte-aligned shapes
    Kernel('binary_wide', 'ops/binary.slang', [
        Entry('binary_main', [F16], ('wide_vec',)),
    ]),
    Kernel('unary_wide', 'ops/unary.slang', [
        Entry('unary_main', [F16], ('wide_vec',)),
    ]),
    Kernel('compare', 'ops/compare.slang', [
        Entry('compare_main', DTYPES, ('vec',)),
    ]),
    Kernel('integer_binary', 'ops/integer_binary.slang', [
        Entry('integer_binary_main', INTEGERS),
    ]),
    Kernel('special', 'ops/special.slang', [
        Entry('special_main', FLOATS),
    ]),
    Kernel('fill', 'ops/fill.slang', [
        Entry('fill_main', DTYPES, ('vec',)),
    ]),
    Kernel('where', 'ops/where.slang', [
        Entry('where_main', DTYPES),
    ]),
    Kernel('where_wide', 'ops/where.slang', [
        Entry('where_main_w', [F16], ('wide_vec',)),
    ]),
    Kernel('addc', 'ops/addc.slang', [
        Entry('addc_main', DTYPES),
    ]),
    Kernel('reduce', 'ops/reduce.slang', [
        Entry('reduce_main', DTYPES),
    ]),
    Kernel('reduce_subgroup', 'ops/reduce_subgroup.slang', [
        Entry('reduce_subgroup_main', [F32, F16]),
    ]),
    Kernel('arg_reduce', 'ops/arg_reduce.slang', [
        Entry('arg_reduce_main', DTYPES),
    ]),
    # two-phase tree sum/prod/amax/amin for long reduce dims on contiguous
    # fp16/fp32 (full-tensor sum, batch_norm train stats, wgrad chains): the
    # base reduce scans per-thread chunks with uncoalesced lanes (~4 GB/s);
    # fp32 accumulation for every input dtype, partials/output fp32 (A2,
    # training_speedup.md)
    Kernel('reduce_tree', 'ops/reduce_tree.slang', [
        Entry('reduce_tree_main', [F16, F32]),
    ]),
    # fused native batch_norm train fwd / backward (A4, training_speedup.md):
    # three kernels per call instead of the 15-20-kernel aten composition.
    # bn_partials: per (channel, chunk) fp32 (sum, sumsq) [fwd] or
    # (sum_dy, sum_dy*xhat) [backward]; bn_finalize: per-channel fold -> mean/
    # rstd/saves/running stats [fwd, T = running dtype] or grad_bias/grad_weight
    # [backward, T = parameter dtype]; bn_apply: normalization / grad_input.
    Kernel('bn_partials', 'ops/batch_norm.slang', [
        Entry('bn_partials_main', [F16, F32]),
    ]),
    Kernel('bn_finalize', 'ops/batch_norm.slang', [
        Entry('bn_finalize_main', [F16, F32]),
    ]),
    Kernel('bn_apply', 'ops/batch_norm.slang', [
        Entry('bn_apply_main', [F16, F32]),
    ]),
    Kernel('scan', 'ops/scan.slang', [
        Entry('scan_main', DTYPES),
    ]),
    Kernel('scan_arg', 'ops/scan_arg.slang', [
        Entry('scan_arg_main', DTYPES),
    ]),
    Kernel('nllloss', 'ops/nllloss.slang', [
        Entry('nllloss_main', FLOATS),
    ]),
    Kernel('copy', 'ops/copy.slang', [
        Entry('copy_main', BYTE_WIDTHS),
    ]),
    # wide strided copy for 2-byte element tensors: uint4 loads/stores along a
    # contiguous innermost dimension (host-checked strides and alignment)
    Kernel('copy_wide', 'ops/copy.slang', [
        Entry('copy_main_w', [U16], ()),
    ]),
    Kernel('index_gather', 'ops/index_gather.slang', [
        Entry('index_gather_main', BYTE_WIDTHS),
    ]),
    Kernel('embedding_backward', 'ops/embedding_backward.slang', [
        Entry('embedding_backward_main', FLOATS),
    ]),
    Kernel('sort', 'ops/sort.slang', [
        Entry('sort_main', DTYPES),
    ]),
    Kernel('scatter', 'ops/scatter.slang', [
        Entry('scatter_main', DTYPES),
    ]),
    Kernel('searchsorted', 'ops/searchsorted.slang', [
        Entry('searchsorted_main', DTYPES),
    ]),
    Kernel('pool', 'ops/pool.slang', [
        Entry('pool_main', DTYPES),
    ]),
    Kernel('resample', 'ops/resample.slang', [
        Entry('resample_main', [F64, F32]),
    ]),
    Kernel('resample_u8', 'ops/resample_u8.slang', [
        Entry('resample_u8_main', [U8]),
    ]),
    Kernel('grid_sample', 'ops/grid_sample.slang', [
        Entry('grid_sample_main', [F64, F32]),
    ]),
    Kernel('ctc', 'ops/ctc.slang', [
        Entry('ctc_main', [F64, F32]),
    ]),
    Kernel('cholesky', 'ops/cholesky.slang', [
        Entry('cholesky_main', [F64, F32]),
    ]),
    Kernel('lu', 'ops/lu.slang', [
        Entry('lu_main', [F64, F32]),
    ]),
    Kernel('triangular_solve', 'ops/triangular_solve.slang', [
        Entry('triangular_solve_main', [F64, F32]),
    ]),
    Kernel('householder', 'ops/householder.slang', [
        Entry('householder_main', [F64, F32]),
    ]),
    Kernel('jacobi', 'ops/jacobi.slang', [
        Entry('jacobi_main', [F64, F32]),
    ]),
    Kernel('ldl', 'ops/ldl.slang', [
        Entry('ldl_main', [F64, F32]),
    ]),
    Kernel('eig', 'ops/eig.slang', [
        Entry('eig_main', [F64, F32]),
    ]),
    Kernel('complex_arc', 'ops/complex_arc.slang', [
        Entry('complex_arc_main', [F32]),
    ]),
    Kernel('fft', 'ops/fft.slang', [
        Entry('fft_main', [F64, F32]),
    ]),
    Kernel('cast', 'ops/cast.slang', [
        Entry('cast_main', DTYPES, ('vec',), pairwise=True),
    ]),
    Kernel('matmul_simd_128', 'ops/matmul_simd.slang', [
        Entry('matmul_simd_main', [d for d in DTYPES if d.bytes < 8], ('vec',), constants=(128,)),
    ]),
    Kernel('matmul_simd_64', 'ops/matmul_simd.slang', [
        Entry('matmul_simd_main', DTYPES, ('vec',), constants=(64,)),
    ]),
    Kernel('matmul_gemv', 'ops/matmul_gemv.slang', [
        Entry('matmul_gemv_main', FLOATS, ('wide_vec',)),
    ]),
    # fused residual add + relu (training hotpath, resnet's `relu(out + shortcut)`):
    # one kernel, two outputs — the pre-activation (relu's backward needs it) and
    # the activated result
    Kernel('add_relu', 'ops/add_relu.slang', [
        Entry('add_relu_main', [F16, F32]),
    ]),
    # weight-only quantized GEMV (M == 1): fp16/fp32 activations, int8/int4
    # weights dequantized in registers; PACK is the bits-per-weight constant
    Kernel('quant_gemv_w8', 'ops/quant_gemv.slang', [
        Entry('quant_gemv_main', [F16, F32], (), constants=(8,)),
    ]),
    Kernel('quant_gemv_w4', 'ops/quant_gemv.slang', [
        Entry('quant_gemv_main', [F16, F32], (), constants=(4,)),
    ]),
    # fused RMSNorm (statistics in fp32, one subgroup per row, weight applied
    # in-kernel) and fused rotate-half RoPE for precomputed full-size tables:
    # one kernel instead of the 9- and 4-kernel aten compositions
    # two-phase tree argmax/argmin for long reduce dims (vocab-size argmax in
    # LLM decode): the base arg_reduce scans sequentially with one thread and
    # costs ~30 ms on a 151936-element fp16 row
    Kernel('arg_reduce_tree', 'ops/arg_reduce_tree.slang', [
        Entry('arg_reduce_tree_main', [F16, F32]),
    ]),
    Kernel('rms_norm', 'ops/rms_norm.slang', [
        Entry('rms_norm_main', [F16, F32]),
    ]),
    # fused residual-add + RMSNorm with two outputs (sum + normed row) for the
    # decode hotpath: one kernel instead of aten add + rms_norm
    Kernel('rms_norm_add', 'ops/rms_norm.slang', [
        Entry('rms_norm_add_main', [F16, F32]),
    ]),
    Kernel('rope_half', 'ops/rope_half.slang', [
        Entry('rope_half_main', [F16, F32]),
    ]),
    # fused per-row RMSNorm + rotate-half RoPE (decode QK stage): one workgroup
    # of 128 threads per row, rrms in fp32 через groupshared-редукцию
    Kernel('rms_rope', 'ops/rms_rope.slang', [
        Entry('rms_rope_main', [F16, F32]),
    ]),
    # fused decode attention (T == 1, GQA): QK^T + softmax + P·V in one kernel,
    # one workgroup of 128 threads per output row, scores in groupshared memory
    # (cap 4096); WG is a compile-time constant (phase-B reductions walk it)
    Kernel('attn_decode', 'ops/attn_decode.slang', [
        Entry('attn_decode_main', [F16], (), constants=(128,)),
    ]),
    Kernel('conv_128', 'ops/conv.slang', [
        Entry('conv_main', [d for d in FLOATS if d.bytes < 8], ('vec',), constants=(128,)),
    ]),
    Kernel('conv_64', 'ops/conv.slang', [
        Entry('conv_main', FLOATS, ('vec',), constants=(64,)),
    ]),
    Kernel('matmul_coop_8', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(8,)),
    ]),
    Kernel('matmul_coop_16', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(16,)),
    ]),
    Kernel('matmul_coop_32', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(32,)),
    ]),
    Kernel('matmul_coop_64', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(64,)),
    ]),
    Kernel('random', 'ops/random.slang', [
        Entry('random_main', DTYPES, ('vec',)),
    ]),
    Kernel('dropout', 'ops/dropout.slang', [
        Entry('dropout_main', FLOATS, ('vec',)),
    ]),
    Kernel('multinomial', 'ops/multinomial.slang', [
        Entry('multinomial_main', [F64, F32]),
    ]),
    Kernel('distribution', 'ops/distribution.slang', [
        Entry('distribution_main', FLOATS),
    ]),
]