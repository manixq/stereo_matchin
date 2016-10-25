__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_MIRRORED_REPEAT
| CLK_FILTER_NEAREST;


__kernel void Decimate (
	__read_only image2d_t input,
	__write_only image2d_t output
 )
{
     //actual(THIS) position of kernel
     const int2 pos = {get_global_id(0), get_global_id(1)};
     //size of our input image
     const int2 base_size = get_image_dim(input);
     float4 sum = (float4)(0.0f);
     float4 piksele = (float4)(0.0f);
     
     piksele += read_imagef(input, sampler, pos * 2 + (int2)(0, 0));
     piksele += read_imagef(input, sampler, pos * 2 + (int2)(0, 1));
     piksele += read_imagef(input, sampler, pos * 2 + (int2)(1, 0));
     piksele += read_imagef(input, sampler, pos * 2 + (int2)(1, 1));
     sum = piksele / 4;
     write_imagef(output, (int2)(pos.x, pos.y), sum);
}