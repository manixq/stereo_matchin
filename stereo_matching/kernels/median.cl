__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

//for sorting rows and columns
void sort(float4 *x, float4 *y, float4 *z)
{
 float4 temp = *x;
 *x = select(*y, *x, isless(*x, *y));
 *y = select(temp, *y, isless(temp, *y));

 temp = *y;
 *y = select(*z, *y, isless(*y, *z));
 *z = select(temp, *z, isless(temp, *z));

 temp = *x;
 *x = select(*y, *x, isless(*x, *y));
 *y = select(temp, *y, isless(temp, *y));
}

__kernel void Median(
 __read_only image2d_t input,
 __write_only image2d_t output
 )
{
 //actual(THIS) position of kernel
 const int2 pos = { get_global_id(0), get_global_id(1) };
 /*
 s0, s1, s2
 s3, s4, s5
 s6, s7, s8
 */
 float4 s[9];
 s[0] = read_imagef(input, sampler, pos + (int2)(-1, -1));
 s[1] = read_imagef(input, sampler, pos + (int2)(0, -1));
 s[2] = read_imagef(input, sampler, pos + (int2)(1, -1));
 s[3] = read_imagef(input, sampler, pos + (int2)(-1, 0));
 s[4] = read_imagef(input, sampler, pos + (int2)(0, 0));
 s[5] = read_imagef(input, sampler, pos + (int2)(1, 0));
 s[6] = read_imagef(input, sampler, pos + (int2)(-1, 1));
 s[7] = read_imagef(input, sampler, pos + (int2)(0, 1));
 s[8] = read_imagef(input, sampler, pos + (int2)(1, 1));
 //rows
 sort(&s[0], &s[1], &s[2]);
 sort(&s[3], &s[4], &s[5]);
 sort(&s[6], &s[7], &s[8]);
 //columns
 sort(&s[0], &s[3], &s[6]);
 sort(&s[1], &s[4], &s[7]);
 sort(&s[2], &s[5], &s[8]);
 //diagonal
 sort(&s[0], &s[4], &s[8]);
 //result
 
 write_imagef(output, (int2)(pos.x, pos.y), s[4]);
}