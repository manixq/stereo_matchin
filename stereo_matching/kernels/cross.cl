__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_MIRRORED_REPEAT
| CLK_FILTER_NEAREST;

__kernel void Decimate (
	__read_only image2d_t input_l,
 __read_only image2d_t input_r,
	__write_only image3d_t output_l,
 __write_only image3d_t output_r
 )
{
     //actual(THIS) position of kernel
     const int2 pos = {get_global_id(0), get_global_id(1)};
     
     write_imagef(image3d_t  image,    int4  coord,    float4  color)
}