

__kernel void Constistency(
 __read_only image2d_t ref,
 __read_only image2d_t tar,
 __write_only image2d_t output
 )
{
 //actual(THIS) position of kernel
 const int2 pos = { get_global_id(0), get_global_id(1) };

 float4 pixel_ref;
 float4 pixel_tar;

 pixel_ref = read_imagef(ref, sampler, pos);
 pixel_tar = read_imagef(tar, sampler, pos);

 float4 result = (float4)(1.0f, 0.0f, 0.0f, 1.0f);

 result =  select(result, pixel_ref, isless(fabs(pixel_tar - pixel_ref), 0.1f));
 write_imagef(output, (int2)(pos.x, pos.y), result);
}