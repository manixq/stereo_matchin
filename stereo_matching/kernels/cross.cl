__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_MIRRORED_REPEAT
| CLK_FILTER_NEAREST;

__kernel void Cross (
	__read_only image2d_t input,
	__write_only image3d_t output
 )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    int4 coords = { pos.x, pos.y, 1, 0 };
    float4 color_data = read_imagef(input, sampler, pos);
    write_imagef(output, coords, (float4)(0.0f,0.0f,5.0f,1.0f));

}