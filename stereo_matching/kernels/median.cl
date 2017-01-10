
/*
Based on:
3x3 Median
Morgan McGuire and Kyle Whitson
http://graphics.cs.williams.edu
*/

void s2(float4 *a, float4 *b)	
{
 float4 temp = *a; 
 *a = min(*a, *b); 
 *b = max(temp, *b);
}

void mn3(float4 *a, float4 *b, float4 *c)
{
 s2(a, b); 
 s2(a, c);
}

void mx3(float4 *a, float4 *b, float4 *c)
{
 s2(b, c); 
 s2(a, c);
}

void mnmx3(float4 *a, float4 *b, float4 *c)
{
 mx3(a, b, c); 
 s2(a, b);
}

void mnmx4(float4 *a, float4 *b, float4 *c, float4 *d)
{
 s2(a, b); 
 s2(c, d); 
 s2(a, c); 
 s2(b, d);
}

void mnmx5(float4 *a, float4 *b, float4 *c, float4 *d, float4 *e)
{
 s2(a, b); 
 s2(c, d); 
 mn3(a, c, e); mx3(b, d, e);
}

void mnmx6(float4 *a, float4 *b, float4 *c, float4 *d, float4 *e, float4 *f)
{
 s2(a, d);
 s2(b, e); 
 s2(c, f); 
 mn3(a, b, c); 
 mx3(d, e, f);
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

 mnmx6(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5]);
 mnmx5(&s[1], &s[2], &s[3], &s[4], &s[6]);
 mnmx4(&s[2], &s[3], &s[4], &s[7]);
 mnmx3(&s[3], &s[4], &s[8]);
 
 write_imagef(output, (int2)(pos.x, pos.y), s[4]);
}