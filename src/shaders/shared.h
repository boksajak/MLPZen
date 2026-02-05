
#if __cplusplus
#pragma once

	// Typedefs for sharing types between C++ (using GLM) and HLSL
	#include "..\common.h"
	typedef uint32_t uint;

	typedef glm::uint16_t half;

	typedef glm::vec2 float2;
	typedef glm::vec3 float3;
	typedef glm::vec4 float4;

	typedef glm::uvec2 uint2;
	typedef glm::uvec3 uint3;
	typedef glm::uvec4 uint4;

	typedef glm::ivec2 int2;
	typedef glm::ivec3 int3;
	typedef glm::ivec4 int4;

	typedef glm::mat2 float2x2;
	typedef glm::mat3 float3x3;
	typedef glm::mat4 float4x4;

#endif

#define MAX_TEXTURES 4
#define MAX_MLPS 8

#define MAX_LAYERS 6
#define HG_MAX_LEVELS 8

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

#define BACKPROP_THREADGROUP_SIZE 32
#define INFERENCE_THREADGROUP_SIZE 8
#define REFOUTPUT_THREADGROUP_SIZE 8
#define INIT_THREADGROUP_SIZE 32
#define OPTIMIZATION_THREADGROUP_SIZE 32
	
struct NNData
{
	uint frameNumber;
	uint outputWidth;
	uint outputHeight;
	uint mlpCount;

	float mlpGradientScaler;
	float hgGradientScaler;
	uint mainTextureIndex;
	float bubblesZ;
};

struct AdamData
{
	float4 mean;
	float4 variance;
};

#define FLOAT_PACKING_CONSTANT_POSITIVE 16384.0f
static uint packFloatPositive(float x)
{
	return uint(x * FLOAT_PACKING_CONSTANT_POSITIVE);
}

static float unpackFloatPositive(uint x)
{
	return float(x) / FLOAT_PACKING_CONSTANT_POSITIVE;
}