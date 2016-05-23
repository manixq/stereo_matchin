
__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


__kernel void Filter (
	__read_only image2d_t input,
	__write_only image2d_t output
 )
{

    const int2 pos = {get_global_id(0), get_global_id(1)};

     float4 sum = (float4)(0.0f);
     float4 piksel = read_imagef(input, sampler, pos);
     for (int i = 0; i < 4; i++)
      write_imagef(output, (int2)(pos.x*4+i, pos.y),piksel);
   

}