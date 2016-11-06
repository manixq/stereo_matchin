__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_MIRRORED_REPEAT
| CLK_FILTER_NEAREST;

__kernel void Aggregation (
	__read_only image2d_t input_l,
 __read_only image2d_t input_r,
	__global int* input_cross_l,
 __global int* input_cross_r,
 __write_only image2d_t output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    write_imagef(output, (int2)(pos.x, pos.y), (float4)(0.5f,0.0f,0.0f,1.0f));
}