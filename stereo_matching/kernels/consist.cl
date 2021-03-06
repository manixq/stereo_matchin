

__kernel void Constistency(
 __read_only image2d_t ref,
 __read_only image2d_t tar,
 __global float* confidence_ref,
 __global float* confidence_tar,
 __write_only image2d_t output,
 __write_only image2d_t output_red
 )
{
 
 //actual(THIS) position of kernel
 const int2 pos = { get_global_id(0), get_global_id(1) };
 const int2 dim = get_image_dim(output);

 float4 pixel_ref;
 float4 pixel_tar;

 pixel_ref = read_imagef(ref, sampler, pos) * 60;
 pixel_tar = read_imagef(tar, sampler, pos) * 60;

 float4 result = (float4)(1.0f, 0.0f, 0.0f, 1.0f);

 result =  select(result, pixel_ref / 60, isless(fabs(pixel_tar - pixel_ref), 1.001f));

 confidence_ref[pos.x + pos.y * dim.x] = select(0.0f, confidence_ref[pos.x + pos.y * dim.x], isless(fabs(pixel_tar.x - pixel_ref.x), 1.001f));
 confidence_tar[pos.x + pos.y * dim.x] = select(0.0f, confidence_tar[pos.x + pos.y * dim.x], isless(fabs(pixel_tar.x - pixel_ref.x), 1.001f));
 
 float4 result_reff = select(pixel_tar / 60, pixel_ref / 60, isless(fabs(pixel_tar - pixel_ref), 1.001f));

 write_imagef(output, (int2)(pos.x, pos.y),result_reff);
 write_imagef(output_red, (int2)(pos.x, pos.y), result );
}