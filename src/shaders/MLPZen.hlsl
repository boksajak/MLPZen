#include "shared.h"

// =========================================================================
//   Network configuration
// =========================================================================

static const uint neuronsPerLayer[MAX_LAYERS] = {
    NEURONS_PER_LAYER_0, NEURONS_PER_LAYER_1, NEURONS_PER_LAYER_2, NEURONS_PER_LAYER_3, NEURONS_PER_LAYER_4, NEURONS_PER_LAYER_5
};

static const uint neuronQuartetsPerLayer[MAX_LAYERS] = {
    NEURON_QUARTETS_PER_LAYER_0, NEURON_QUARTETS_PER_LAYER_1, NEURON_QUARTETS_PER_LAYER_2, NEURON_QUARTETS_PER_LAYER_3, NEURON_QUARTETS_PER_LAYER_4, NEURON_QUARTETS_PER_LAYER_5
};

// Hashgrid levels resolutions
static const uint hgLevelResolutions[HG_MAX_LEVELS * 2] = {
    HG_LEVEL_RES_0, HG_LEVEL_RES_1, HG_LEVEL_RES_2, HG_LEVEL_RES_3, HG_LEVEL_RES_4, HG_LEVEL_RES_5, HG_LEVEL_RES_6, HG_LEVEL_RES_7,
    HG_LEVEL_RES_8, HG_LEVEL_RES_9, HG_LEVEL_RES_10, HG_LEVEL_RES_11, HG_LEVEL_RES_12, HG_LEVEL_RES_13, HG_LEVEL_RES_14, HG_LEVEL_RES_15
};

// Hashgrid levels offsets of feature vectors
static const uint hgLevelOffsets[HG_MAX_LEVELS * 2] = {
    HG_LEVEL_OFFSET_0, HG_LEVEL_OFFSET_1, HG_LEVEL_OFFSET_2, HG_LEVEL_OFFSET_3, HG_LEVEL_OFFSET_4, HG_LEVEL_OFFSET_5, HG_LEVEL_OFFSET_6, HG_LEVEL_OFFSET_7,
    HG_LEVEL_OFFSET_8, HG_LEVEL_OFFSET_9, HG_LEVEL_OFFSET_10, HG_LEVEL_OFFSET_11, HG_LEVEL_OFFSET_12, HG_LEVEL_OFFSET_13, HG_LEVEL_OFFSET_14, HG_LEVEL_OFFSET_15
};

// Flags indicating whether given hashgrid level is hashed
static const bool hgLevelHashed[HG_MAX_LEVELS * 2] = {
    HG_LEVEL_HASHED_0, HG_LEVEL_HASHED_1, HG_LEVEL_HASHED_2, HG_LEVEL_HASHED_3, HG_LEVEL_HASHED_4, HG_LEVEL_HASHED_5, HG_LEVEL_HASHED_6, HG_LEVEL_HASHED_7,
    HG_LEVEL_HASHED_8, HG_LEVEL_HASHED_9, HG_LEVEL_HASHED_10, HG_LEVEL_HASHED_11, HG_LEVEL_HASHED_12, HG_LEVEL_HASHED_13, HG_LEVEL_HASHED_14, HG_LEVEL_HASHED_15
};

#if MLP_FP16_STORAGE
    #define MLP_FLOAT_TYPE half
    #define MLP_FLOAT4_TYPE half4
#else
    #define MLP_FLOAT_TYPE float
    #define MLP_FLOAT4_TYPE float4
#endif

#if HG_FP16_STORAGE
    #define HG_FLOAT4_STORAGE_TYPE half4
#else
    #define HG_FLOAT4_STORAGE_TYPE float4
#endif

// =========================================================================
//   Resources
// =========================================================================

// Constant buffer with global data
cbuffer NNDataCB : register(b0)
{
    NNData gData;
}

// These are per-training iteration settings which might be changed several times per frame
struct RootConstants
{
    uint trainingStep;
    float adamEpsilon;
    float adamBeta1;
    float adamBeta2;
    float adamBeta1T;
    float adamBeta2T;
    float learningRate;
    uint useNormalDistribution;
    uint useHeStrategy;
    uint useXavierStrategy;
    uint freezeMLP;
    uint freezeHashgrid;
};
cbuffer RootConstantsCB : register(b1)
{
    RootConstants gRootConstants;
}

// Output and reference output images
RWTexture2D<float4> Output : register(u0);
RWTexture2D<float4> Reference : register(u1);

// Loss Data
RWStructuredBuffer<unsigned int> lossData : register(u2);

// Reference texture (input image)
Texture2D<float4> targetTexture[MAX_TEXTURES] : register(t0, space1);

// MLP buffers
ByteAddressBuffer nnParametersInputBuffer[MAX_MLPS] : register(t0, space0);
StructuredBuffer<float4> nnParametersBackpropInputBuffer[MAX_MLPS] : register(t0, space2);
ByteAddressBuffer nnHashgridInputBuffer[MAX_MLPS] : register(t0, space3);
RWStructuredBuffer<MLP_FLOAT4_TYPE> nnParametersOutputBuffer[MAX_MLPS] : register(u0, space1);
RWStructuredBuffer<AdamData> nnAdamDataBuffer[MAX_MLPS] : register(u0, space2);
RWStructuredBuffer<int4> nnGradientBuffer[MAX_MLPS] : register(u0, space3);
RWStructuredBuffer<float> nnParametersBackpropOutputBuffer[MAX_MLPS] : register(u0, space4);
RWStructuredBuffer<HG_FLOAT4_STORAGE_TYPE> nnHashgridOutputBuffer[MAX_MLPS] : register(u0, space5);
StructuredBuffer<uint4> nnMemoryLayoutMap  : register(t0, space4);

// -------------------------------------------------------------------------
//    RNG
// -------------------------------------------------------------------------

// 32-bit Xorshift random number generator
inline uint xorshift32(inout uint rngState)
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return rngState;
}

// PCG Hash Function
// Source: https://jcgt.org/published/0009/03/02/
uint pcgHash(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
// Explanation: https://vectrx.substack.com/p/lcg-xs-fast-gpu-rng
float uintToFloat(const uint x)
{
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

// Initialize RNG
uint initRNG(const uint linearIndex, const uint frameNumber) {
    return pcgHash(linearIndex ^ pcgHash(frameNumber));
}

// Initialize RNG for given pixel and frame
uint initRNG(const uint2 pixelCoords, const uint2 resolution, const uint frameNumber) {
    return initRNG(dot(pixelCoords, uint2(1, resolution.x)), frameNumber);
}

// Generate random float in <0; 1) range
float rand(inout uint rngState) {
    return uintToFloat(xorshift32(rngState));
}

// Generate a random float in the range <-x; x)
float randInRange(inout uint rng, float x)
{
    return (rand(rng) * 2.0f - 1.0f) * x;
}

// Precise inverse error function approximation using polynomials
// Source: "Approximating the erfinv function"
// https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
float MBG_erfinv(float x)
{
    float w, p;
    w = -log((1.0f - x) * (1.0f + x));
    if (w < 5.000000f)
    {
        w = w - 2.500000f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    }
    else
    {
        w = sqrt(w) - 3.000000f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

// Source: "Sampling From a Normal (Gaussian) Distribution on GPUs"
// https://gpuopen.com/learn/sampling-normal-gaussian-distribution-gpus/
float gaussianDistribution(const float u, float mean, float standardDeviation)
{
    return mean + sqrt(2) * standardDeviation * MBG_erfinv(2 * u - 1);
}

// =========================================================================
//   Bubbles demo
// =========================================================================

struct Bubble
{
    float3 center;
    float3 color;
    float radius;
};

float3 getBubble(const float3 coords)
{
    static const int nBubbles = 9;
    static Bubble bubbles[nBubbles] =
    {
        { float3(0.5, 0.5, 0.5), float3(1, 1, 0.3), 0.3f },
        { float3(0.2, 0.1, 0.5), float3(0.3, 0.3, 0.9), 0.2f },
        { float3(0.9, 0.7, 0.4), float3(0.3, 1, 0.3), 0.3f },
        { float3(0.6, 0.3, 0.2), float3(0.3, 0.8, 0.3), 0.3f },
        { float3(0.0, 0.2, 0.7), float3(0.3, 1, 1), 0.5f },
        { float3(0.1, 0.4, 0.2), float3(1, 0.3, 0.3), 0.3f },
        { float3(0.4, 0.9, 0.8), float3(1, 0.3, 1), 0.15f },
        { float3(0.9, 0.4, 0.2), float3(1, 0.3, 1), 0.25f },
        { float3(0.8, 0.1, 0.9), float3(1, 0.3, 0.3), 0.3f },
    };
    
    for (int i = 0; i < nBubbles; i++)
    {
        const float distance = length(coords - bubbles[i].center);
        if (distance < bubbles[i].radius) return bubbles[i].color;
    }
    
    return 0;
}

// =========================================================================
//   Activation functions
// =========================================================================

#define LEAKY_RELU_SLOPE 0.01

template<typename T>
T leakyRelu(T x)
{
    return (x >= T(0.0)) ? x : (x * T(LEAKY_RELU_SLOPE));
}

template<typename T>
T leakyReluDeriv(T x)
{
    return (x <= T(0.0)) ? T(LEAKY_RELU_SLOPE) : T(1.0);
}

template<typename T>
T sigmoid(T x)
{
    return T(1.0) / (T(1.0) + exp(-x));
}

template<typename T>
T sigmoidDeriv(T x)
{
    return x * (T(1.0) - x);
}

MLP_FLOAT_TYPE activationFunction(MLP_FLOAT_TYPE x)
{
    return leakyRelu<MLP_FLOAT_TYPE>(x);
}

MLP_FLOAT4_TYPE activationFunction(MLP_FLOAT4_TYPE v)
{
    return MLP_FLOAT4_TYPE(
        activationFunction(v.x),
        activationFunction(v.y),
        activationFunction(v.z),
        activationFunction(v.w)
    );
}

MLP_FLOAT_TYPE activationFunctionOutput(MLP_FLOAT_TYPE x)
{
#if OUTPUT_LEAKY_RELU
    return leakyRelu<MLP_FLOAT_TYPE>(x);
#elif OUTPUT_SIGMOID
    return sigmoid<MLP_FLOAT_TYPE>(x);
#elif OUTPUT_NONE
    return x;
#endif
}

MLP_FLOAT4_TYPE activationFunctionOutput(MLP_FLOAT4_TYPE v)
{
    return MLP_FLOAT4_TYPE(
        activationFunctionOutput(v.x),
        activationFunctionOutput(v.y),
        activationFunctionOutput(v.z),
        activationFunctionOutput(v.w)
    );
}

float activationFunctionFP32(float x)
{
    return leakyRelu<float>(x);
}

float4 activationFunctionFP32(float4 v)
{
    return float4(
        activationFunctionFP32(v.x),
        activationFunctionFP32(v.y),
        activationFunctionFP32(v.z),
        activationFunctionFP32(v.w)
    );
}

float activationFunctionOutputFP32(float x)
{
#if OUTPUT_LEAKY_RELU
    return leakyRelu<float>(x);
#elif OUTPUT_SIGMOID
    return sigmoid<float>(x);
#elif OUTPUT_NONE
    return x;
#endif
}

float4 activationFunctionOutputFP32(float4 v)
{
    return float4(
        activationFunctionOutputFP32(v.x),
        activationFunctionOutputFP32(v.y),
        activationFunctionOutputFP32(v.z),
        activationFunctionOutputFP32(v.w)
    );
}

float activationFunctionDeriv(float x)
{
    return leakyReluDeriv(x);
}

float4 activationFunctionDeriv(float4 v)
{
    return float4(
        activationFunctionDeriv(v.x),
        activationFunctionDeriv(v.y),
        activationFunctionDeriv(v.z),
        activationFunctionDeriv(v.w)
    );
}

float activationFunctionOutputDeriv(float x)
{
#if OUTPUT_LEAKY_RELU
    return leakyReluDeriv<float>(x);
#elif OUTPUT_SIGMOID
    return sigmoidDeriv<float>(x);
#elif OUTPUT_NONE
    return 1.0f;
#endif
}

float4 activationFunctionOutputDeriv(float4 v)
{
    return float4(
        activationFunctionOutputDeriv(v.x),
        activationFunctionOutputDeriv(v.y),
        activationFunctionOutputDeriv(v.z),
        activationFunctionOutputDeriv(v.w)
    );
}

// =========================================================================
//   Helper functions
// =========================================================================

#define FLOAT4_PACKING_CONSTANT 65536.0f // Maximum for 64k records of magnitude 0.5
float4 unpackFloat4(int4 x)
{
    return float4(x) / FLOAT4_PACKING_CONSTANT;
}

int4 packFloat4(float4 x)
{
    return int4(x * FLOAT4_PACKING_CONSTANT);
}

float4 unpackFloat4HP(int4 x)
{
    // Higher precision version (8x)
    return float4(x) / (FLOAT4_PACKING_CONSTANT * 8);
}

int4 packFloat4HP(float4 x)
{
    // Higher precision version (8x)
    return int4(x * FLOAT4_PACKING_CONSTANT * 8);
}

float4 unpackFloat4MLP(int4 x)
{
#if HIGH_PRECISION_GRADIENT
    return unpackFloat4HP(x);
#else
    return unpackFloat4(x);
#endif
}

void accumulateGradient(RWStructuredBuffer<int4> gradientTarget, const uint gradientIndex, float4 gradient)
{

#if CLIP_GRADIENT
    gradient = clamp(gradient, -0.5f, 0.5f);
#endif

    // Pack to integer and accumulate
    const int4 packed = packFloat4(gradient);
    InterlockedAdd(gradientTarget[gradientIndex].x, packed.x);
    InterlockedAdd(gradientTarget[gradientIndex].y, packed.y);
    InterlockedAdd(gradientTarget[gradientIndex].z, packed.z);
    InterlockedAdd(gradientTarget[gradientIndex].w, packed.w);
}

// Wave-based version with higher gradient precision
// This pre-calculates average gradient within a wave enabling greater precision of float packing and higher clipping magnitude
// You can only used this for MLP parameters which are coherent within a wave.
// Hashgrid weights have different gradient index in every thread, so you need to use a thread-based version for those.
void accumulateGradientWave(RWStructuredBuffer<int4> gradientTarget, const uint gradientIndex, float4 gradient)
{
    if (WaveIsFirstLane())
    {
        const float4 totalWaveGradient = WaveActiveSum(gradient);
        float4 avgWaveGradient = totalWaveGradient / float(BACKPROP_THREADGROUP_SIZE);

#if CLIP_GRADIENT
        avgWaveGradient = clamp(avgWaveGradient, -2.0f, 2.0f);
#endif

        const int4 packed = packFloat4HP(avgWaveGradient);

        InterlockedAdd(gradientTarget[gradientIndex].x, packed.x);
        InterlockedAdd(gradientTarget[gradientIndex].y, packed.y);
        InterlockedAdd(gradientTarget[gradientIndex].z, packed.z);
        InterlockedAdd(gradientTarget[gradientIndex].w, packed.w);
    }
}

void accumulateGradientMLP(RWStructuredBuffer<int4> gradientTarget, const uint gradientIndex, float4 gradient)
{
    // For accumulation of MLP parameters gradient, choose between high or low precision version here
#if HIGH_PRECISION_GRADIENT
    accumulateGradientWave(gradientTarget, gradientIndex, gradient);
#else
    accumulateGradient(gradientTarget, gradientIndex, gradient);
#endif
}

// Helper for the forward pass memory layout
// Checks whether parameter quartet stored at linearIndex is valid, reads out layer index and whether it stores connection weights or biases
bool getParamQuartetInfo(const uint linearIndex, inout uint layer, inout bool isWeight)
{
    layer = (linearIndex / MAX_PARAM_QUARTETS_PER_LAYER) + 1;
    const uint paramsQuartet = linearIndex % MAX_PARAM_QUARTETS_PER_LAYER;

    if (layer >= LAYER_COUNT) return false;

    const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    const uint thisParamQuartetsPerLayer = (neuronCountCurrentLayer * neuronCountPreviousLayer + neuronCountCurrentLayer) / 4;

    if (paramsQuartet >= thisParamQuartetsPerLayer)
        return false;

    const uint paramsGroupSize = neuronCountPreviousLayer + 1;
    const uint paramsGroup = paramsQuartet / paramsGroupSize;
    const uint paramsQuartetInGroup = paramsQuartet % paramsGroupSize;

    // Figure out if this is weight or bias
    isWeight = (paramsQuartetInGroup < (paramsGroupSize - 1));

    return true;
}

half2 UnpackFloat16(const uint a)
{
    return half2(f16tof32(uint2(a & 0xFFFF, a >> 16)));
}

#if MLP_FP16_STORAGE
    half4 loadNNParameter(const uint index, inout ByteAddressBuffer nnParameters)
    {
        const uint2 raw = nnParameters.Load2(index * 8);
        return min16float4(UnpackFloat16(raw.x), UnpackFloat16(raw.y));
    }
#else
    float4 loadNNParameter(const uint index, inout ByteAddressBuffer nnParameters)
    {
        return asfloat(nnParameters.Load4(index * 16));
    }
#endif

#if HG_FP16_STORAGE
    half4 loadHGParameter(const uint index, inout ByteAddressBuffer hgParameters)
    {
        const uint2 raw = hgParameters.Load2(index * 8);
        return min16float4(UnpackFloat16(raw.x), UnpackFloat16(raw.y));
    }
#else
    float4 loadHGParameter(const uint index, inout ByteAddressBuffer hgParameters)
    {
        return asfloat(hgParameters.Load4(index * 16));
    }
#endif

// =========================================================================
//   Input Encoding functions
// =========================================================================

struct Hashgrid2DEncodingData
{
    float2 weight[HG_LEVELS];
    uint4 indices[HG_LEVELS];
};

struct Hashgrid3DEncodingData
{
    float3 weight[HG_LEVELS];
    uint4 indicesA[HG_LEVELS];
    uint4 indicesB[HG_LEVELS];
};

struct InputEncodingData
{
#if INPUT_LENGTH == 2
    Hashgrid2DEncodingData hgData;
#elif INPUT_LENGTH == 3
    Hashgrid3DEncodingData hgData;
#else
    uint pad;
#endif
};

// Helper for oneblob encoding
inline float quarticCdf(const float x, const float invRadius)
{
    const float u = x * invRadius;
    const float u2 = u * u;
    const float u4 = u2 * u2;
    return saturate((15.0f / 16.0f) * u * (1.0f - (2.0f / 3.0f) * u2 + (1.0f / 5.0f) * u4) + 0.5f);
}

// Source: "Neural Importance Sampling"
// https://arxiv.org/pdf/1808.03856
template<typename InputType, typename T>
void oneBlobEncoding(const InputType input, const int inputCount, inout uint activationIndex, inout T activations)
{
    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
    {
        const float currentInput = input[inputIndex];

        for (int binIndex = 0; binIndex < NUM_ONEBLOB_BINS / 4; binIndex++)
        {
            MLP_FLOAT4_TYPE encodedValue;
            for (int i = 0; i < 4; i++)
            {
                const float leftBoundary = ((binIndex * 4) + i) * ONEBLOB_ZETA;
                const float leftCdf = quarticCdf(leftBoundary - currentInput, NUM_ONEBLOB_BINS);

                const float rightBoundary = ((binIndex * 4) + i + 1) * ONEBLOB_ZETA;
                const float rightCdf = quarticCdf(rightBoundary - currentInput, NUM_ONEBLOB_BINS);

                encodedValue[i] = MLP_FLOAT_TYPE(rightCdf - leftCdf);
            }

            activations[activationIndex++] = encodedValue;
        }
    }
}

// Source: "Positional Encoding" from "NeRF: Representing Scenes as Neural RadianceFields for View Synthesis"
// https://arxiv.org/pdf/2003.08934
template<typename InputType, typename T>
void frequencyEncoding(const InputType input, const int inputCount, inout uint activationIndex, inout T activations)
{
    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
    {
        const float p = PI * input[inputIndex];
        int modifier = 1;
        
        for (int f = 0; f < NUM_FREQUENCIES / 2; f++)
        {
            MLP_FLOAT4_TYPE encoded = 0.0;
            {
                const float x = modifier * p;
                encoded[0] = MLP_FLOAT_TYPE(sin(x));
                encoded[1] = MLP_FLOAT_TYPE(cos(x));
                modifier *= 2;
            }
            {
                const float x = modifier * p;
                encoded[2] = MLP_FLOAT_TYPE(sin(x));
                encoded[3] = MLP_FLOAT_TYPE(cos(x));
                modifier *= 2;
            }
            
            activations[activationIndex++] = encoded;
        }
    }
}

// Translates grid coordinates into array index, with optional hashing
// 2D version
uint getHashgridIndex(const uint levelOffset, const uint levelResolution, const bool isLevelHashed, const uint2 coords)
{
    // Linearize coordinates
    uint index = coords.x + coords.y * levelResolution;

    // Hash the index and map it to the hash table
    if (isLevelHashed) {
        index = pcgHash(index) % HG_MAP_SIZE;
    }

    return levelOffset + index;
}

// Translates grid coordinates into array index, with optional hashing
// 3D version
uint getHashgridIndex(const uint levelOffset, const uint levelResolution, const bool isLevelHashed, const uint3 coords)
{
    // Linearize coordinates
    uint index = coords.x + coords.y * levelResolution + coords.z * levelResolution * levelResolution;

    // Hash the index and map it to the hash table
    if (isLevelHashed)
    {
        index = pcgHash(index) % HG_MAP_SIZE;
    }

    return levelOffset + index;
}

template<typename T>
float4 loadHashgrid(const uint mlpIndex, const uint levelOffset, const uint levelResolution, const bool isLevelHashed, const T coords, inout uint index)
{
    index = getHashgridIndex(levelOffset, levelResolution, isLevelHashed, coords);
    return loadHGParameter(index, nnHashgridInputBuffer[mlpIndex]);
}

template<typename T>
float4 loadHashgrid(const uint mlpIndex, const uint levelOffset, const uint levelResolution, const bool isLevelHashed, const T coords)
{
    uint index;
    return loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, coords, index);
}

// Source: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
// https://arxiv.org/abs/2201.05989
template<typename T>
void hashgridEncoding2D(const uint mlpIndex, const float2 input, inout uint activationIndex, inout T activations, inout Hashgrid2DEncodingData inputEncodingData)
{
    for (uint level = 0; level < HG_LEVELS; level++)
    {
        const uint levelResolution = hgLevelResolutions[level];
        const float maxCoordinate = levelResolution - 1;
        const uint levelOffset = hgLevelOffsets[level];
        const bool isLevelHashed = hgLevelHashed[level];

        // Map input point to cell of this grid
        float2 weight = min(maxCoordinate, clamp(input, 0.0f, 0.9999f) * maxCoordinate);
        const uint2 gridCoordinates = floor(weight);
        weight -= float2(gridCoordinates);

        // Load nearest feature vectors
        uint4 indices;
        float4 a = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint2(gridCoordinates.x + 0, gridCoordinates.y + 0), indices.x);
        float4 b = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint2(gridCoordinates.x + 0, gridCoordinates.y + 1), indices.y);
        float4 c = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint2(gridCoordinates.x + 1, gridCoordinates.y + 0), indices.z);
        float4 d = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint2(gridCoordinates.x + 1, gridCoordinates.y + 1), indices.w);

        const float4 featureVector =
            (1.0f - weight.x) * (1.0f - weight.y) * a +
            (1.0f - weight.x) * weight.y          * b +
            weight.x          * (1.0f - weight.y) * c +
            weight.x          * weight.y          * d;

        activations[activationIndex++] = MLP_FLOAT4_TYPE(featureVector);

        inputEncodingData.weight[level] = weight;
        inputEncodingData.indices[level] = HG_OFFSET + indices;
    }
}

// Source: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
// https://arxiv.org/abs/2201.05989
template<typename T>
void hashgridEncoding3D(const uint mlpIndex, const float3 input, inout uint activationIndex, inout T activations, inout Hashgrid3DEncodingData inputEncodingData)
{
    for (uint level = 0; level < HG_LEVELS; level++)
    {
        const uint levelResolution = hgLevelResolutions[level];
        const float maxCoordinate = levelResolution - 1;
        const uint levelOffset = hgLevelOffsets[level];
        const bool isLevelHashed = hgLevelHashed[level];

        // Map input point to cell of this grid
        float3 weight = min(maxCoordinate, clamp(input, 0.0f, 0.9999f) * maxCoordinate);
        const uint3 gridCoordinates = floor(weight);
        weight -= float3(gridCoordinates);

        // Load nearest feature vectors
        uint4 indicesA;
        uint4 indicesB;
        float4 a = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 0, gridCoordinates.y + 0, gridCoordinates.z + 0), indicesA.x);
        float4 b = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 0, gridCoordinates.y + 1, gridCoordinates.z + 0), indicesA.y);
        float4 c = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 1, gridCoordinates.y + 0, gridCoordinates.z + 0), indicesA.z);
        float4 d = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 1, gridCoordinates.y + 1, gridCoordinates.z + 0), indicesA.w);
        float4 e = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 0, gridCoordinates.y + 0, gridCoordinates.z + 1), indicesB.x);
        float4 f = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 0, gridCoordinates.y + 1, gridCoordinates.z + 1), indicesB.y);
        float4 g = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 1, gridCoordinates.y + 0, gridCoordinates.z + 1), indicesB.z);
        float4 h = loadHashgrid(mlpIndex, levelOffset, levelResolution, isLevelHashed, uint3(gridCoordinates.x + 1, gridCoordinates.y + 1, gridCoordinates.z + 1), indicesB.w);

        const float4 featureVector =
            (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * a +
            (1.0f - weight.x) * weight.y          * (1.0f - weight.z) * b +
            weight.x          * (1.0f - weight.y) * (1.0f - weight.z) * c +
            weight.x          * weight.y          * (1.0f - weight.z) * d +
            (1.0f - weight.x) * (1.0f - weight.y) * weight.z          * e +
            (1.0f - weight.x) * weight.y          * weight.z          * f +
            weight.x          * (1.0f - weight.y) * weight.z          * g +
            weight.x          * weight.y          * weight.z          * h;

        activations[activationIndex++] = MLP_FLOAT4_TYPE(featureVector);

        inputEncodingData.weight[level] = weight;
        inputEncodingData.indicesA[level] = HG_OFFSET + indicesA;
        inputEncodingData.indicesB[level] = HG_OFFSET + indicesB;
    }
}

void hashgridEncoding2DBackprop(const uint mlpIndex, const Hashgrid2DEncodingData inputEncodingData, inout float4 errors[LAYER_COUNT * MAX_NEURON_QUARTETS_PER_LAYER])
{
    uint errorIndex = 0;

    for (uint level = 0; level < HG_LEVELS; level++)
    {
        const float4 neuronGrad = errors[errorIndex++];
        const float2 weight = inputEncodingData.weight[level];
        
        const float4 gradientA = neuronGrad * (1.0f - weight.x) * (1.0f - weight.y);
        const float4 gradientB = neuronGrad * (1.0f - weight.x) * weight.y;
        const float4 gradientC = neuronGrad * weight.x          * (1.0f - weight.y);
        const float4 gradientD = neuronGrad * weight.x          * weight.y;

        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indices[level].x, gradientA);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indices[level].y, gradientB);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indices[level].z, gradientC);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indices[level].w, gradientD);
    }
}

void hashgridEncoding3DBackprop(const uint mlpIndex, const Hashgrid3DEncodingData inputEncodingData, inout float4 errors[LAYER_COUNT * MAX_NEURON_QUARTETS_PER_LAYER])
{
    uint errorIndex = 0;

    for (uint level = 0; level < HG_LEVELS; level++)
    {
        const float4 neuronGrad = errors[errorIndex++];
        const float3 weight = inputEncodingData.weight[level];
        
        const float4 gradientA = neuronGrad * (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z);
        const float4 gradientB = neuronGrad * (1.0f - weight.x) * weight.y          * (1.0f - weight.z);
        const float4 gradientC = neuronGrad * weight.x          * (1.0f - weight.y) * (1.0f - weight.z);
        const float4 gradientD = neuronGrad * weight.x          * weight.y          * (1.0f - weight.z);
        const float4 gradientE = neuronGrad * (1.0f - weight.x) * (1.0f - weight.y) * weight.z;
        const float4 gradientF = neuronGrad * (1.0f - weight.x) * weight.y          * weight.z;
        const float4 gradientG = neuronGrad * weight.x          * (1.0f - weight.y) * weight.z;
        const float4 gradientH = neuronGrad * weight.x          * weight.y          * weight.z;

        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesA[level].x, gradientA);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesA[level].y, gradientB);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesA[level].z, gradientC);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesA[level].w, gradientD);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesB[level].x, gradientE);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesB[level].y, gradientF);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesB[level].z, gradientG);
        accumulateGradient(nnGradientBuffer[mlpIndex], inputEncodingData.indicesB[level].w, gradientH);
    }
}

template<typename EntryType, typename InputType, typename T>
void encodeInput(const uint mlpIndex, const InputType input, const int inputLength, inout uint activationIndex, inout T activations, inout InputEncodingData inputEncodingData)
{
#if USE_IDENTITY_ENCODING

    if (inputLength == 1)
    {
        activations[activationIndex++] = EntryType(input[0], 0.0, 0.0, 0.0);
    }
    else if (inputLength == 2)
    {
        activations[activationIndex++] = EntryType(input[0], input[1], 0.0, 0.0);
    }
    else if (inputLength == 3)
    {
        activations[activationIndex++] = EntryType(input[0], input[1], input[2], 0.0);
    }
    
#elif USE_FREQUENCY_ENCODING

    frequencyEncoding(input, inputLength, activationIndex, activations);

#elif USE_ONEBLOB_ENCODING

    oneBlobEncoding(input, inputLength, activationIndex, activations);

#elif USE_HASHGRID_ENCODING

    #if INPUT_LENGTH == 2
        hashgridEncoding2D(mlpIndex, input, activationIndex, activations, inputEncodingData.hgData);
    #endif
    #if INPUT_LENGTH == 3
        hashgridEncoding3D(mlpIndex, input, activationIndex, activations, inputEncodingData.hgData);
    #endif
    
#endif
}

// =========================================================================
//   Inference
// =========================================================================

void evalLayer(
    inout MLP_FLOAT4_TYPE previousActivations[MAX_NEURON_QUARTETS_PER_LAYER],
    inout MLP_FLOAT4_TYPE currentActivations[MAX_NEURON_QUARTETS_PER_LAYER],
    inout ByteAddressBuffer nnParameters,
    uint paramOffset,
    const uint neuronQuartetCountCurrentLayer,
    const uint neuronQuartetCountPreviousLayer,
    const uint layerType)
{
    for (int neuronQuartet = 0; neuronQuartet < neuronQuartetCountCurrentLayer; neuronQuartet++)
    {
        MLP_FLOAT4_TYPE neuronValue = 0.0f;
        
        for (uint previousNeuronQuartet = 0; previousNeuronQuartet < neuronQuartetCountPreviousLayer; previousNeuronQuartet++)
        {
            const MLP_FLOAT4_TYPE previousActivationsQuartet = previousActivations[previousNeuronQuartet];
            neuronValue.x += dot(loadNNParameter(paramOffset++, nnParameters), previousActivationsQuartet);
            neuronValue.y += dot(loadNNParameter(paramOffset++, nnParameters), previousActivationsQuartet);
            neuronValue.z += dot(loadNNParameter(paramOffset++, nnParameters), previousActivationsQuartet);
            neuronValue.w += dot(loadNNParameter(paramOffset++, nnParameters), previousActivationsQuartet);
        }
        
        const MLP_FLOAT4_TYPE bias = MLP_FLOAT4_TYPE(loadNNParameter(paramOffset++, nnParameters));

        if (layerType == HIDDEN_LAYER)
        {
            currentActivations[neuronQuartet] = activationFunction(neuronValue + bias);
        }
        else // if OUTPUT_LAYER
        {
            currentActivations[neuronQuartet] = activationFunctionOutput(neuronValue + bias);
        }
    }
}

void evalLayerActivations(
    inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK],
    inout ByteAddressBuffer nnParameters,
    uint neuronWeightsOffset,
    uint previousNeuronOffset,
    uint currentNeuronOffset,
    const uint neuronQuartetCountCurrentLayer,
    const uint neuronQuartetCountPreviousLayer,
    const uint layerType)
{
    for (int neuronQuartet = 0; neuronQuartet < neuronQuartetCountCurrentLayer; neuronQuartet++)
    {
        float4 neuronValue = 0.0f;

        for (uint previousNeuronQuartet = previousNeuronOffset; previousNeuronQuartet < (previousNeuronOffset + neuronQuartetCountPreviousLayer); previousNeuronQuartet++)
        {
            const float4 previousActivationsQuartet = activations[previousNeuronQuartet];
            neuronValue.x += dot(float4(loadNNParameter(neuronWeightsOffset++, nnParameters)), previousActivationsQuartet);
            neuronValue.y += dot(float4(loadNNParameter(neuronWeightsOffset++, nnParameters)), previousActivationsQuartet);
            neuronValue.z += dot(float4(loadNNParameter(neuronWeightsOffset++, nnParameters)), previousActivationsQuartet);
            neuronValue.w += dot(float4(loadNNParameter(neuronWeightsOffset++, nnParameters)), previousActivationsQuartet);
        }

        const float4 bias = float4(loadNNParameter(neuronWeightsOffset++, nnParameters));

        if (layerType == HIDDEN_LAYER)
        {
            activations[currentNeuronOffset++] = activationFunctionFP32(neuronValue + bias);
        }
        else // if OUTPUT_LAYER
        {
            activations[currentNeuronOffset++] = activationFunctionOutputFP32(neuronValue + bias);
        }
    }
}

template<typename InputType>
float3 inference(const uint mlpIndex, const InputType input)
{
    MLP_FLOAT4_TYPE activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    MLP_FLOAT4_TYPE activationsB[MAX_NEURON_QUARTETS_PER_LAYER];
    
    // Encode input into first array
    InputEncodingData inputEncodingData;
    uint activationIndex = 0;
    encodeInput<MLP_FLOAT4_TYPE>(mlpIndex, input, INPUT_LENGTH, activationIndex, activationsA, inputEncodingData);
    
    // Eval layer 1
    evalLayer(activationsA, activationsB, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 0, NEURON_QUARTETS_PER_LAYER_1, NEURON_QUARTETS_PER_LAYER_0, LAYER_TYPE_1);

#if (LAYER_COUNT > 2)
    evalLayer(activationsB, activationsA, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 1, NEURON_QUARTETS_PER_LAYER_2, NEURON_QUARTETS_PER_LAYER_1, LAYER_TYPE_2);
#endif

#if (LAYER_COUNT > 3)
    evalLayer(activationsA, activationsB, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 2, NEURON_QUARTETS_PER_LAYER_3, NEURON_QUARTETS_PER_LAYER_2, LAYER_TYPE_3);
#endif

#if (LAYER_COUNT > 4)
    evalLayer(activationsB, activationsA, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 3, NEURON_QUARTETS_PER_LAYER_4, NEURON_QUARTETS_PER_LAYER_3, LAYER_TYPE_4);
#endif

#if (LAYER_COUNT > 5)
    evalLayer(activationsA, activationsB, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 4, NEURON_QUARTETS_PER_LAYER_5, NEURON_QUARTETS_PER_LAYER_4, LAYER_TYPE_5);
#endif

    return float3(((LAYER_COUNT % 2) == 0) ? activationsB[0].rgb : activationsA[0].rgb);
}

template<typename InputType>
float3 inference(const InputType input)
{
    return inference(0, input);
}

template<typename InputType>
void forwardActivations(const uint mlpIndex, const InputType input, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK], inout InputEncodingData inputEncodingData)
{
    // Encode input
    uint activationIndex = 0;
    encodeInput<float4>(mlpIndex, input, INPUT_LENGTH, activationIndex, activations, inputEncodingData);

    // Eval layer 1
    evalLayerActivations(activations, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 0,
        MAX_NEURON_QUARTETS_PER_LAYER * 0, MAX_NEURON_QUARTETS_PER_LAYER * 1, NEURON_QUARTETS_PER_LAYER_1, NEURON_QUARTETS_PER_LAYER_0, LAYER_TYPE_1);

#if (LAYER_COUNT > 2)
    evalLayerActivations(activations, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 1,
        MAX_NEURON_QUARTETS_PER_LAYER * 1, MAX_NEURON_QUARTETS_PER_LAYER * 2, NEURON_QUARTETS_PER_LAYER_2, NEURON_QUARTETS_PER_LAYER_1, LAYER_TYPE_2);
#endif

#if (LAYER_COUNT > 3)
    evalLayerActivations(activations, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 2,
        MAX_NEURON_QUARTETS_PER_LAYER * 2, MAX_NEURON_QUARTETS_PER_LAYER * 3, NEURON_QUARTETS_PER_LAYER_3, NEURON_QUARTETS_PER_LAYER_2, LAYER_TYPE_3);
#endif

#if (LAYER_COUNT > 4)
    evalLayerActivations(activations, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 3,
        MAX_NEURON_QUARTETS_PER_LAYER * 3, MAX_NEURON_QUARTETS_PER_LAYER * 4, NEURON_QUARTETS_PER_LAYER_4, NEURON_QUARTETS_PER_LAYER_3, LAYER_TYPE_4);
#endif

#if (LAYER_COUNT > 5)
    evalLayerActivations(activations, nnParametersInputBuffer[mlpIndex], MAX_PARAM_QUARTETS_PER_LAYER * 4,
        MAX_NEURON_QUARTETS_PER_LAYER * 4, MAX_NEURON_QUARTETS_PER_LAYER * 5, NEURON_QUARTETS_PER_LAYER_5, NEURON_QUARTETS_PER_LAYER_4, LAYER_TYPE_5);
#endif

}

template<typename InputType>
void forwardActivations(const InputType input, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK], inout InputEncodingData inputEncodingData)
{
    forwardActivations(0, input, activations, inputEncodingData);
}

// =========================================================================
//   Inference
// =========================================================================

[numthreads(INFERENCE_THREADGROUP_SIZE, INFERENCE_THREADGROUP_SIZE, 1)]
void Inference(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.x >= gData.outputWidth || LaunchIndex.y >= gData.outputHeight)
        return;
    
#if LEARN_COSINE

    const float2 inputXY = float2(LaunchIndex + 0.5) / float2(gData.outputWidth, gData.outputHeight);

    const float x = LaunchIndex.x / float(gData.outputWidth - 1);
    const float y = 1.0f - inference((float2(inputXY.x, 0.0f) * 2.0f) % 1.0f).x;

    if (abs(y - (inputXY.y * 2) + 0.5) < 0.005)
    {
        Output[LaunchIndex] = 1;
    }
    else
    {
        Output[LaunchIndex] = 0;
    }

#elif LEARN_IMAGE

    // Figure out UV coordinates for this pixel
    const float2 uvs = float2(LaunchIndex + 0.5f) / float2(gData.outputWidth, gData.outputHeight);
    Output[LaunchIndex] = float4(inference(uvs), 1);

#elif LEARN_FOUR_IMAGES

    const int halfWidth = gData.outputWidth / 2;
    const int halfHeight = gData.outputHeight / 2;

    // Figure out which texture to infer
    const int x = LaunchIndex.y / halfWidth;
    const int y = LaunchIndex.x / halfHeight;
    const int imageIndex = x * 2 + y;

    // Figure out UV coordinates for this pixel
    const float2 uvs = float2(
            ((LaunchIndex.x % (halfWidth)) + 0.5f) / float(halfWidth),
            ((LaunchIndex.y % (halfHeight)) + 0.5f) / float(halfHeight)
        );

    Output[LaunchIndex] = float4(inference(imageIndex, uvs), 1);

#elif LEARN_BUBBLES

    const float3 xyz = float3(float2(LaunchIndex) / float2(gData.outputWidth - 1, gData.outputHeight - 1), gData.bubblesZ);
    Output[LaunchIndex] = float4(inference(xyz), 1);

#endif

}

// =========================================================================
//   Network initialization
// =========================================================================

// Source: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
// https://arxiv.org/pdf/1502.01852
float heStandardDeviation(const uint nInputs)
{
    return sqrt(2.0f / nInputs);
}

float heUniformScale(const uint nInputs)
{
    return sqrt(6.0f / nInputs);
}

// Source: "Normalized Initialization" from "Understanding the difficulty of training deep feedforward neural networks"
// https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
float xavierUniformScale(const uint nInputs, const uint nOutputs)
{
    return sqrt(6.0f / (nInputs + nOutputs));
}

float xavierStandardDeviation(const uint nInputs, const uint nOutputs)
{
    return sqrt(2.0f / (nInputs + nOutputs));
}

// Source: "Efficient BackProp"
// https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf
float lecunStandardDeviation(const uint nInputs)
{
    return sqrt(1.0f / nInputs);
}

float lecunUniformScale(const uint nInputs)
{
    return sqrt(3.0f / nInputs);
}

[numthreads(INIT_THREADGROUP_SIZE, 1, 1)]
void Initialize(
    int2 groupID : SV_GroupID,
    int2 groupThreadID : SV_GroupThreadID,
    int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.y >= gData.mlpCount) return;
    const uint linearIndex = LaunchIndex.x;
    const uint mlpIndex = LaunchIndex.y;

    // Initialize RNG
    uint rng = initRNG(linearIndex, gData.frameNumber);

    // Initialize hash-grid
    {
#if USE_HASHGRID_ENCODING

        if (linearIndex < HG_TOTAL_QUARTETS)
        {
            // Set initial hash-grid entry to small number centered around zero
            const float hgInitialScale = 0.001f;
            nnHashgridOutputBuffer[mlpIndex][linearIndex] = HG_FLOAT4_STORAGE_TYPE(
                randInRange(rng, hgInitialScale),
                randInRange(rng, hgInitialScale),
                randInRange(rng, hgInitialScale),
                randInRange(rng, hgInitialScale)
            );

            // Clear gradients and Adam parameters to zeros
            const uint hgIndex = linearIndex + HG_OFFSET;
            nnGradientBuffer[mlpIndex][hgIndex] = 0;
            nnAdamDataBuffer[mlpIndex][hgIndex].mean = 0.0f;
            nnAdamDataBuffer[mlpIndex][hgIndex].variance = 0.0f;
        }

#endif
    }

    // Initialize the MLP
    {
        // Figure out details about this parameter quartet
        uint layer;
        bool isWeight;
        if (!getParamQuartetInfo(linearIndex, layer, isWeight))
        {
            return;
        }

        if (isWeight)
        {
            const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
            const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];

            MLP_FLOAT4_TYPE initialWeight;

            if (gRootConstants.useNormalDistribution)
            {
                // Initialize weights according to normal (Gaussian) distribution
                float sigma;
                if (gRootConstants.useHeStrategy)
                {
                    // He strategy standard deviation
                    sigma = heStandardDeviation(neuronCountPreviousLayer);
                }
                else if (gRootConstants.useXavierStrategy)
                {
                    // Xavier strategy standard deviation
                    sigma = xavierStandardDeviation(neuronCountPreviousLayer, neuronCountCurrentLayer);
                }
                else
                {
                    // LeCun strategy standard deviation
                    sigma = lecunStandardDeviation(neuronCountPreviousLayer);
                }
                
                const float minWeight = -3.0f * sigma;
                const float maxWeight = 3.0f * sigma;

                initialWeight = MLP_FLOAT4_TYPE(
                    clamp(gaussianDistribution(rand(rng), 0.0, sigma), minWeight, maxWeight),
                    clamp(gaussianDistribution(rand(rng), 0.0, sigma), minWeight, maxWeight),
                    clamp(gaussianDistribution(rand(rng), 0.0, sigma), minWeight, maxWeight),
                    clamp(gaussianDistribution(rand(rng), 0.0, sigma), minWeight, maxWeight)
                );
            }
            else
            {
                // Initialize weights according to uniform distribution
                float scale;
                if (gRootConstants.useHeStrategy)
                {
                    // He strategy uniform distribution range
                    scale = heUniformScale(neuronCountPreviousLayer);
                }
                else if (gRootConstants.useXavierStrategy)
                {
                    // Xavier strategy uniform distribution range
                    scale = xavierUniformScale(neuronCountPreviousLayer, neuronCountCurrentLayer);
                }
                else
                {
                    // LeCun strategy uniform distribution range
                    scale = lecunUniformScale(neuronCountPreviousLayer);
                }
                
                initialWeight = MLP_FLOAT4_TYPE(
                    randInRange(rng, scale),
                    randInRange(rng, scale),
                    randInRange(rng, scale),
                    randInRange(rng, scale)
                );
            }

            nnParametersOutputBuffer[mlpIndex][linearIndex] = initialWeight;
        }
        else
        {
            // Initialize bias to zero
            nnParametersOutputBuffer[mlpIndex][linearIndex] = 0.0;
        }

        // Clear gradients and Adam parameters to zeros
        nnGradientBuffer[mlpIndex][linearIndex] = 0;
        nnAdamDataBuffer[mlpIndex][linearIndex].mean = 0.0f;
        nnAdamDataBuffer[mlpIndex][linearIndex].variance = 0.0f;
    }
}

// =========================================================================
//   Network Training
// =========================================================================

void backpropLayer(const uint mlpIndex, const float3 target, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK],
    inout float4 errors[LAYER_COUNT * MAX_NEURON_QUARTETS_PER_LAYER],
    const uint currentLayer, const uint layerType)
{
    const uint previousLayer = (layerType == INPUT_LAYER) ? 0 : (currentLayer - 1);
    const uint nextLayer = (layerType == OUTPUT_LAYER) ? 0 : currentLayer + 1;
    const uint neuronQuartetCountCurrentLayer = neuronQuartetsPerLayer[currentLayer];
    const uint neuronQuartetCountPreviousLayer = neuronQuartetsPerLayer[previousLayer];
    const uint neuronQuartetCountNextLayer = neuronQuartetsPerLayer[nextLayer];
    const uint currentLayerActivationsOffset = MAX_NEURON_QUARTETS_PER_LAYER * currentLayer;
    const uint previousLayerActivationsOffset = MAX_NEURON_QUARTETS_PER_LAYER * previousLayer;
    const uint nextLayerActivationsOffset = MAX_NEURON_QUARTETS_PER_LAYER * nextLayer;
    uint gradientIndex = MAX_PARAM_QUARTETS_PER_LAYER * previousLayer;
    uint nextLayerWeightsIndex = (currentLayer * MAX_NEURONS_PER_LAYER * MAX_NEURONS_PER_LAYER) / 4;
    float totalLoss = 0.0f;

    for (uint neuron = currentLayerActivationsOffset; neuron < currentLayerActivationsOffset + neuronQuartetCountCurrentLayer; neuron++)
    {
        const float4 neuronActivation = activations[neuron];
        float4 dCost_O = 0.0f;

        if (layerType == OUTPUT_LAYER)
        {
            // For output layer, calculate derivative of the loss function
            const float4 l2LossDeriv = (neuronActivation - float4(target, 0.0f));
            dCost_O = l2LossDeriv;

            #if CALCULATE_GLOBAL_LOSS
                totalLoss += dot(l2LossDeriv, l2LossDeriv);
            #endif
        }
        else
        {
            // For hidden and input layer, calculate derivative based on next layer errors
            for (uint nextNeuron = nextLayerActivationsOffset; nextNeuron < nextLayerActivationsOffset + neuronQuartetCountNextLayer; nextNeuron++)
            {
                const float4 neuronError = errors[nextNeuron];
                dCost_O.x += dot(neuronError, nnParametersBackpropInputBuffer[mlpIndex][nextLayerWeightsIndex++]);
                dCost_O.y += dot(neuronError, nnParametersBackpropInputBuffer[mlpIndex][nextLayerWeightsIndex++]);
                dCost_O.z += dot(neuronError, nnParametersBackpropInputBuffer[mlpIndex][nextLayerWeightsIndex++]);
                dCost_O.w += dot(neuronError, nnParametersBackpropInputBuffer[mlpIndex][nextLayerWeightsIndex++]);
            }
        }

        if (layerType == INPUT_LAYER)
        {
            // Derivative of input layer is equal to neuron error
            errors[neuron] = dCost_O;
        }
        else
        {
            // Derivative of hidden and output layer takes activation function derivative into account
            const float4 dO_Z = ((layerType == HIDDEN_LAYER) ? activationFunctionDeriv(neuronActivation) : activationFunctionOutputDeriv(neuronActivation));
            const float4 dCost_Z = dCost_O * dO_Z;
            errors[neuron] = dCost_Z;

            // Calculate weights gradient
            for (uint previousNeuron = previousLayerActivationsOffset; previousNeuron < previousLayerActivationsOffset + neuronQuartetCountPreviousLayer; previousNeuron++)
            {
                const float4 dCost_weightX = dCost_Z.x * activations[previousNeuron];
                const float4 dCost_weightY = dCost_Z.y * activations[previousNeuron];
                const float4 dCost_weightZ = dCost_Z.z * activations[previousNeuron];
                const float4 dCost_weightW = dCost_Z.w * activations[previousNeuron];

                accumulateGradientMLP(nnGradientBuffer[mlpIndex], gradientIndex++, dCost_weightX);
                accumulateGradientMLP(nnGradientBuffer[mlpIndex], gradientIndex++, dCost_weightY);
                accumulateGradientMLP(nnGradientBuffer[mlpIndex], gradientIndex++, dCost_weightZ);
                accumulateGradientMLP(nnGradientBuffer[mlpIndex], gradientIndex++, dCost_weightW);
            }

            // Calculate bias gradient
            accumulateGradientMLP(nnGradientBuffer[mlpIndex], gradientIndex++, dCost_Z);
        }
    }

    if (layerType == OUTPUT_LAYER)
    {
        #if CALCULATE_GLOBAL_LOSS
            totalLoss /= float(neuronQuartetCountCurrentLayer * 4);
            InterlockedAdd(lossData[mlpIndex * 2 + 0], packFloatPositive(totalLoss));
            InterlockedAdd(lossData[mlpIndex * 2 + 1], 1);
        #endif
    }
}

void backpropagation(const uint mlpIndex, const float3 target, float4 activations[ACTIVATION_QUARTETS_PER_NETWORK], const InputEncodingData inputEncodingData)
{
    float4 errors[ACTIVATION_QUARTETS_PER_NETWORK];

    // Output layer derivatives
    backpropLayer(mlpIndex, target, activations, errors, (LAYER_COUNT - 1), OUTPUT_LAYER);

    // Hidden layer derivatives
#if (LAYER_COUNT > 2)
    backpropLayer(mlpIndex, target, activations, errors, (LAYER_COUNT - 2), HIDDEN_LAYER);
#endif

#if (LAYER_COUNT > 3)
    backpropLayer(mlpIndex, target, activations, errors, (LAYER_COUNT - 3), HIDDEN_LAYER);
#endif  

#if (LAYER_COUNT > 4)
    backpropLayer(mlpIndex, target, activations, errors, (LAYER_COUNT - 4), HIDDEN_LAYER);
#endif  

#if (LAYER_COUNT > 5)
    backpropLayer(mlpIndex, target, activations, errors, (LAYER_COUNT - 5), HIDDEN_LAYER);
#endif  

    // Input layer derivatives
#if USE_HASHGRID_ENCODING
    backpropLayer(mlpIndex, target, activations, errors, 0, INPUT_LAYER);
    
    #if INPUT_LENGTH == 2
        hashgridEncoding2DBackprop(mlpIndex, inputEncodingData.hgData, errors);
    #elif INPUT_LENGTH == 3
        hashgridEncoding3DBackprop(mlpIndex, inputEncodingData.hgData, errors);
    #endif
#endif  

}

// Runs backpropagation (calculates gradients based on training inputs and reference output)
[numthreads(BACKPROP_THREADGROUP_SIZE, 1, 1)]
void Backpropagation(
    int2 groupID : SV_GroupID,
    int2 groupThreadID : SV_GroupThreadID,
    int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.y >= gData.mlpCount) return;
    const uint linearIndex = LaunchIndex.x;
    const uint mlpIndex = LaunchIndex.y;

    // Initialize random numbers generator
    uint rng = initRNG(linearIndex, gRootConstants.trainingStep);

#if LEARN_COSINE

    const float2 input = float2(rand(rng), 0.0f);
    const float3 target = (cos(input.x * 2 * PI).xxx + 1) * 0.5;
    
#elif LEARN_IMAGE

    // Generate a random input (UV coordinates in the image)
    const float2 input = float2(rand(rng), rand(rng));

    // Load target value to learn for this input from reference image
    const uint2 inputCoords = min(input * float2(gData.outputWidth, gData.outputHeight), float2(gData.outputWidth - 1, gData.outputHeight - 1));
    const float3 target = targetTexture[gData.mainTextureIndex][inputCoords].rgb;

#elif LEARN_FOUR_IMAGES

    // Generate a random input (UV coordinates in the image)
    const float2 input = float2(rand(rng), rand(rng));

    // Load target value to learn for this input from reference image
    const uint2 inputCoords = min(input * float2(gData.outputWidth, gData.outputHeight), float2(gData.outputWidth - 1, gData.outputHeight - 1));
    const float3 target = targetTexture[mlpIndex][inputCoords].rgb;

#elif LEARN_BUBBLES

    const float3 input = float3(rand(rng), rand(rng), rand(rng));
    const float3 target = getBubble(input);

#endif

    // First run forward pass to evaluate network activations for given input
    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    InputEncodingData inputEncodingData;
    forwardActivations(mlpIndex, input, activations, inputEncodingData);

    // Run backpropagation on current network state
    backpropagation(mlpIndex, target, activations, inputEncodingData);
}

// Source: "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
// https://arxiv.org/pdf/1412.6980
float4 AdamOptimizer(const uint mlpIndex, const float4 gradient, const uint dataIndex)
{
    // Load gradient, mean and variance
    AdamData adamData = nnAdamDataBuffer[mlpIndex][dataIndex];
    
    // Update mean and variance for this training step
    adamData.mean = lerp(gradient, adamData.mean, gRootConstants.adamBeta1);
    adamData.variance = lerp((gradient * gradient), adamData.variance, gRootConstants.adamBeta2);

    // Store updated mean and variance
    nnAdamDataBuffer[mlpIndex][dataIndex] = adamData;

    // Calculate weight adjustment
    const float4 correctedMean = adamData.mean / (1.0f - gRootConstants.adamBeta1T);
    const float4 correctedVariance = adamData.variance / (1.0f - gRootConstants.adamBeta2T);

    // Update the NN parameter
    return -gRootConstants.learningRate * (correctedMean * rsqrt(correctedVariance + gRootConstants.adamEpsilon));
}

[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void Optimization(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.y >= gData.mlpCount) return;
    const uint linearIndex = LaunchIndex.x;
    const uint mlpIndex = LaunchIndex.y;

#if USE_HASHGRID_ENCODING

    if (linearIndex >= HG_OFFSET && !gRootConstants.freezeHashgrid)
    {
        // Update hashgrid params
        const uint hgLinearQuartetIndex = linearIndex - HG_OFFSET;
        if (hgLinearQuartetIndex < HG_TOTAL_QUARTETS)
        {
            const float4 gradient = unpackFloat4(nnGradientBuffer[mlpIndex][linearIndex]) * gData.hgGradientScaler;

            float4 hgParameter = nnHashgridOutputBuffer[mlpIndex][hgLinearQuartetIndex];
#if USE_SGD_OPTIMIZER
            hgParameter += -gRootConstants.learningRate * gradient;
#else
            hgParameter += AdamOptimizer(mlpIndex, gradient, linearIndex);
#endif
            // Update the parameter
            nnHashgridOutputBuffer[mlpIndex][hgLinearQuartetIndex] = HG_FLOAT4_STORAGE_TYPE(hgParameter);

            // Clear the gradient
            nnGradientBuffer[mlpIndex][linearIndex] = 0;
        }
    }

#endif

    // Update MLP params
    uint layer;
    bool isWeight;
    if (getParamQuartetInfo(linearIndex, layer, isWeight) && !gRootConstants.freezeMLP)
    {
        const float4 gradient = unpackFloat4MLP(nnGradientBuffer[mlpIndex][linearIndex]) * gData.mlpGradientScaler;

        float4 nnParameter = nnParametersOutputBuffer[mlpIndex][linearIndex];
#if USE_SGD_OPTIMIZER
        nnParameter += -gRootConstants.learningRate * gradient;
#else
        nnParameter += AdamOptimizer(mlpIndex, gradient, linearIndex);
#endif

        // Update the parameter
        nnParametersOutputBuffer[mlpIndex][linearIndex] = MLP_FLOAT4_TYPE(nnParameter);

        // Clear the gradient
        nnGradientBuffer[mlpIndex][linearIndex] = 0;

        // Write out weights for backprop pass
        {
            const uint4 targetIndices = nnMemoryLayoutMap[linearIndex];
            for (int i = 0; i < 4; i++)
            {
                const uint targetIndex = targetIndices[i];
                if (targetIndex != uint(-1))
                {
                    nnParametersBackpropOutputBuffer[mlpIndex][targetIndex] = nnParameter[i];
                }
            }
        }
    }

    // Clear global loss
    {
        #if CALCULATE_GLOBAL_LOSS
            lossData[mlpIndex * 2 + 0] = 0;
            lossData[mlpIndex * 2 + 1] = 0;
        #endif
    }
}

// =========================================================================
//   Reference Display
// =========================================================================

[numthreads(REFOUTPUT_THREADGROUP_SIZE, REFOUTPUT_THREADGROUP_SIZE, 1)]
void OutputReference(
    int2 groupID : SV_GroupID,
    int2 groupThreadID : SV_GroupThreadID,
    int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.x >= gData.outputWidth || LaunchIndex.y >= gData.outputHeight)
        return;

    const float2 input = float2(LaunchIndex) / float2(gData.outputWidth, gData.outputHeight);

#if LEARN_COSINE

    const float y = 1.0f - ((cos(input.x * 4.0f * PI) + 1.0f) * 0.5f);

    if (abs(y - (input.y * 2.0f) + 0.5f) < 0.005f)
    {
        Reference[LaunchIndex] = 1;
    }
    else
    {
        Reference[LaunchIndex] = 0;
    }

#elif LEARN_IMAGE

    const uint2 inputCoords = min(input * float2(gData.outputWidth, gData.outputHeight), float2(gData.outputWidth - 1, gData.outputHeight - 1));
    Reference[LaunchIndex] = float4(targetTexture[gData.mainTextureIndex][inputCoords].rgb, 1);

#elif LEARN_FOUR_IMAGES

    const int halfWidth = gData.outputWidth / 2;
    const int halfHeight = gData.outputHeight / 2;
 
    // Figure out which texture to display
    const int x = LaunchIndex.y / halfWidth;
    const int y = LaunchIndex.x / halfHeight;
    const int imageIndex = x * 2 + y;

    // Figure out UV coordinates for this pixel
    const float2 uvs = float2(
        ((LaunchIndex.x % (halfWidth)) + 0.5f) / float(halfWidth),
        ((LaunchIndex.y % (halfHeight)) + 0.5f) / float(halfHeight)
    );
 
    const uint2 inputCoords = min(uvs * float2(gData.outputWidth, gData.outputHeight), float2(gData.outputWidth - 1, gData.outputHeight - 1));
    Reference[LaunchIndex] = float4(targetTexture[imageIndex][inputCoords].rgb, 1);

#elif LEARN_BUBBLES
    
    Reference[LaunchIndex] = float4(getBubble(float3(input, gData.bubblesZ)), 1);
    
#endif
}