use paired::Engine;
use ff::PrimeField;
use itertools::join;

static DEFS_SRC : &str = include_str!("common/defs.cl");
static FIELD_SRC : &str = include_str!("common/field.cl");
static FFT_SRC : &str = include_str!("fft/fft.cl");

static EXP_SRC : &str = include_str!("multiexp/exp.cl");
static FIELD2_SRC : &str = include_str!("multiexp/field2.cl");
static EC_SRC : &str = include_str!("multiexp/ec.cl");
static MULTIEXP_SRC : &str = include_str!("multiexp/multiexp.cl");

fn limbs_of<T>(value: &T) -> &[u64] {
    unsafe {
        std::slice::from_raw_parts(value as *const T as *const u64, std::mem::size_of::<T>() / 8)
    }
}

fn calc_inv(a: u64) -> u64 {
    let mut inv = 1u64;
    for _ in 0..63 {
        inv = inv.wrapping_mul(inv);
        inv = inv.wrapping_mul(a);
    }
    return inv.wrapping_neg();
}

fn params<F>(name: &str) -> String where F: PrimeField {
    let one = F::one(); let one = limbs_of(&one);
    let p = F::char(); let p = limbs_of(&p);
    let limbs = one.len();
    let inv = calc_inv(p[0]);
    let limbs_def = format!("#define {}_LIMBS {}", name, limbs);
    let p_def = format!("#define {}_P (({}){{ {{ {} }} }})", name, name, join(p, ", "));
    let one_def = format!("#define {}_ONE (({}){{ {{ {} }} }})", name, name, join(one, ", "));
    let zero_def = format!("#define {}_ZERO (({}){{ {{ {} }} }})", name, name, join(vec![0u32; limbs], ", "));
    let inv_def = format!("#define {}_INV {}", name, inv);
    return format!("{}\n{}\n{}\n{}\n{}", limbs_def, one_def, p_def, zero_def, inv_def);
}

fn exponent<F>(name: &str) -> String where F: PrimeField {
    return format!("{}\n{}\n",
        format!("#define {}_LIMBS {}", name, limbs_of(&F::one()).len()),
        String::from(EXP_SRC).replace("EXPONENT", name));
}

fn field<F>(name: &str) -> String where F: PrimeField {
    return format!("{}\n{}\n",
        params::<F>(name),
        String::from(FIELD_SRC).replace("FIELD", name));
}

fn field2(field2: &str, field: &str) -> String {
    return String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field);
}

fn fft(field: &str) -> String {
    return String::from(FFT_SRC)
        .replace("FIELD", field);
}

fn ec(field: &str, point: &str) -> String {
    return String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point);
}

fn multiexp(point: &str, exp: &str) -> String {
    return String::from(MULTIEXP_SRC)
        .replace("POINT", point)
        .replace("EXPONENT", exp);;
}

pub fn fft_kernel<F>() -> String where F: PrimeField {
    return String::from(format!("{}\n{}\n{}",
        DEFS_SRC,
        field::<F>("Fr"), fft("Fr")));
}

pub fn multiexp_kernel<E>() -> String where E: Engine {
    return String::from(format!("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        DEFS_SRC,
        exponent::<E::Fr>("Exp"),
        field::<E::Fq>("Fq"), ec("Fq", "G1"), multiexp("G1", "Exp"),
        field2("Fq2", "Fq"), ec("Fq2", "G2"), multiexp("G2", "Exp")));
}
