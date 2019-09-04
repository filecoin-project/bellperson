/* Bellman multiexp */

__kernel void POINT_bellman_multiexp(
    __global POINT_affine *bases,
    __global POINT_projective *buckets,
    __global POINT_projective *results,
    __global EXPONENT *exps,
    __global bool *dm,
    uint skip,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  uint gid = get_global_id(0);
  if(gid > num_windows * num_groups) return;

  uint bucket_len = ((1 << window_size) - 1);
  bases += skip;
  buckets += bucket_len * gid;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = POINT_ZERO;

  uint len = (uint)ceil(n / (float)num_groups);
  uint nstart = len * (gid / num_windows);
  uint nend = min(nstart + len, n);

  uint bits = (gid % num_windows) * window_size;
  ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_projective res = POINT_ZERO;
  for(uint i = nstart; i < nend; i++) {
    if(dm[i]) {
      uint ind = EXPONENT_get_bits(exps[i], bits, w);
      if(ind == 1) res = POINT_add_mixed(res, bases[i]);
      else if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    }
  }

  POINT_projective acc = POINT_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = POINT_add(acc, buckets[j]);
    res = POINT_add(res, acc);
  }

  results[gid] = res;
}
