__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


float FilterValue (__constant const float* filterWeights,	const int x, const int y)
{
	return filterWeights[(x+FILTER_SIZE) + (y+FILTER_SIZE)*(FILTER_SIZE*2 + 1)];
}
int ispixeleq(float4 a,float4 b)
{
 if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
 {
  return 1;
 }
 return 0;
}
__kernel void Filter (
	__read_only image2d_t input,
 __read_only image2d_t input2,
	__constant float* filterWeights,
	__write_only image2d_t output)
{

    const int2 pos = {get_global_id(0), get_global_id(1)};

    float4 sum = (float4)(0.0f);
    for(int x = 0; x <= 50; x++) {
     float4 a = read_imagef(input, sampler, pos);
     float4 b = read_imagef(input2, sampler, pos + (int2)(x, 0));
     if(ispixeleq(a,b))
      sum += (float4)(0.0f, 0.5f, 0.0f, 1.0f)+ read_imagef(input2, sampler, pos+(int2)(x,0));
     
    }

    write_imagef (output, (int2)(pos.x, pos.y), sum);
}