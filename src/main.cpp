#include "common.h"
#include "utils.h"
#include "shaders/shared.h"

// Windows DPI Scaling
#include <ShellScalingApi.h>
#pragma comment(lib, "shcore.lib")

const unsigned int frameWidth = 1820;
const unsigned int frameHeight = 980;

class DxcShaderCompiler
{
public:

	void Initialize()
	{
		HRESULT hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
		utils::validate(hr, L"Failed to create IDxcCompiler3 instance!");

		hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
		utils::validate(hr, L"Failed to create IDxcUtils instance!");
	}

	IDxcBlob* CompileShader(LPCWSTR filename, LPCWSTR entryPoint, LPCWSTR targetProfile, std::vector<LPCWSTR> compilerFlags = {})
	{
		IDxcResult* result = nullptr;
		HRESULT hr = S_OK;

		bool shaderCompiled = false;
		while (!shaderCompiled) {

			// Load the shader source file
			IDxcBlobEncoding* shaderSource = nullptr;
			hr = dxcUtils->LoadFile(filename, nullptr, &shaderSource);
			utils::validate(hr, L"Error: failed to create blob from shader file!");

			// Turn the shader source into a buffer object
			DxcBuffer sourceBuffer = {};
			sourceBuffer.Ptr = shaderSource->GetBufferPointer();
			sourceBuffer.Size = shaderSource->GetBufferSize();
			sourceBuffer.Encoding = DXC_CP_ACP;

			// Create the compiler include handler
			IDxcIncludeHandler* dxcIncludeHandler;
			hr = dxcUtils->CreateDefaultIncludeHandler(&dxcIncludeHandler);
			utils::validate(hr, L"Error: failed to create include handler");

			// Additional compiler flags (always present)
			compilerFlags.push_back(DXC_ARG_ALL_RESOURCES_BOUND);
			compilerFlags.push_back(DXC_ARG_WARNINGS_ARE_ERRORS); //< -WX
			compilerFlags.push_back(L"-enable-16bit-types");
			compilerFlags.push_back(L"-encoding utf8"); //< Force outputs to be in UTF-8
			compilerFlags.push_back(L"-Qstrip_reflect");
			compilerFlags.push_back(L"-HV 2021");

#if _DEBUG
			compilerFlags.push_back(DXC_ARG_DEBUG); //< -Zi
#else
			compilerFlags.push_back(L"-Qstrip_debug");
#endif

			// Build arguments for the compile command
			IDxcCompilerArgs* compilerArgs = nullptr;
			dxcUtils->BuildArguments(filename, entryPoint, targetProfile,
				compilerFlags.data(), (UINT32)compilerFlags.size(),
				nullptr, 0, &compilerArgs);

			// Compile the shader
			hr = dxcCompiler->Compile(&sourceBuffer, compilerArgs->GetArguments(), compilerArgs->GetCount(), dxcIncludeHandler, IID_PPV_ARGS(&result));
			utils::validate(hr, L"Error: failed to compile shader!");

			// Release the compile operation resources
			shaderSource->Release();
			dxcIncludeHandler->Release();
			compilerArgs->Release();

			// Verify the result 
			result->GetStatus(&hr);
			if (FAILED(hr))
			{
				// Read errors
				IDxcBlobUtf8* compileErrors;
				hr = result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&compileErrors), nullptr);
				utils::validate(hr, L"Error: failed to retrieve DXC compilation errors!");

				// Ask user if he wants to try recompiling again?
				bool retry = (MessageBox(nullptr, utils::stringToWstring(compileErrors->GetStringPointer()).c_str(), L"DXC Shader Compiler Error", MB_RETRYCANCEL) == IDRETRY);
				compileErrors->Release();

				if (retry)
				{
					continue;
				}
				else
				{
					return nullptr;
				}
			}

			// Successful compilation
			shaderCompiled = true;
		}

		IDxcBlob* compiledShaderBlob = nullptr;
		hr = result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&compiledShaderBlob), nullptr);
		utils::validate(hr, L"Error: failed to get compiled shader blob!");

		compiledShaders.push_back(result);

		return compiledShaderBlob;
	}

	void Release()
	{
		SAFE_RELEASE(dxcCompiler);
		SAFE_RELEASE(dxcUtils);

		for (auto& shader : compiledShaders)
		{
			SAFE_RELEASE(shader);
		}

		compiledShaders.clear();

	}

private:

	IDxcCompiler3* dxcCompiler = nullptr;
	IDxcUtils* dxcUtils = nullptr;

	std::vector<IDxcResult*> compiledShaders;
};

class Profiler
{
public:

	void Initialize(ID3D12Device5* device) {

		// Create query heap
		{
			SAFE_RELEASE(mQueryHeap);

			D3D12_QUERY_HEAP_DESC heapDesc = { };
			heapDesc.Count = cMaxQueryCount * 2;
			heapDesc.NodeMask = 0;
			heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
			HRESULT hr = device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&mQueryHeap));
			utils::validate(hr, L"Error: failed to create profiling query heap!");
		}

		// Create readback heap for query results
		{
			SAFE_RELEASE(mReadbackBuffer);

			D3D12_RESOURCE_DESC resourceDesc = {};
			resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			resourceDesc.Width = uint32_t(cMaxQueryCount * 2 * sizeof(uint64_t));
			resourceDesc.Height = 1;
			resourceDesc.DepthOrArraySize = 1;
			resourceDesc.MipLevels = 1;
			resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
			resourceDesc.SampleDesc.Count = 1;
			resourceDesc.SampleDesc.Quality = 0;
			resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			resourceDesc.Alignment = 0;

			HRESULT	hr = device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
				D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mReadbackBuffer));
			utils::validate(hr, L"Error: failed to create profiling readback heap!");
		}
	}

	std::string BeginFrame(ID3D12GraphicsCommandList4* cmdList, ID3D12CommandQueue* cmdQueue) {

		std::string result = "";

		// Get resolved timings from previous frame
		if (mQueryCount > 0) {

			// Download queries results
			D3D12_RANGE range;
			range.Begin = SIZE_T(0);
			range.End = SIZE_T(mQueryCount * 2 * sizeof(uint64_t));
			void* mappedData = nullptr;
			mReadbackBuffer->Map(0, &range, &mappedData);

			uint64_t* queryData = ((uint64_t*)mappedData);

			for (size_t i = 0; i < mQueryCount; i++)
			{
				uint64_t startTime = queryData[i * 2];
				uint64_t endTime = queryData[i * 2 + 1];

				uint64_t delta = endTime - startTime;
				double frequency = double(mLastGpuFrequency);
				float queryTimeMs = (delta / frequency) * 1000.0;

				float cachedTimeMs = 0.0f;
				auto cachedTimeItem = mCachedTimes.find(mQueryNames[i]);
				if (cachedTimeItem != mCachedTimes.end())
				{
					cachedTimeMs = cachedTimeItem->second;
				}

				const float alpha = 0.99f;
				cachedTimeMs = alpha * cachedTimeMs + (1.0f - alpha) * queryTimeMs;
				mCachedTimes[mQueryNames[i]] = cachedTimeMs;

				char temp[256];
				snprintf(temp, 256, "%-16ls: %6.4fms (%.2fms)\n", mQueryNames[i].c_str(), queryTimeMs, cachedTimeMs);
				result += temp;
			}

			mReadbackBuffer->Unmap(0, nullptr);
		}

		mCmdList = cmdList;
		mQueryCount = 0;

		cmdQueue->GetTimestampFrequency(&mLastGpuFrequency);

		return result;
	}

	unsigned int StartEvent(std::wstring name) {

		unsigned int queryIndex = mQueryCount++;
		mQueryNames[queryIndex] = name;

		if (queryIndex >= cMaxQueryCount) {
			utils::validate(E_FAIL, L"Only 'cMaxQueryCount' profiles are supported, this is one too many!");
			return -1;
		}

		// Timestamp query only supports EndQuery method (so we call it during StartEvent as well as EndEvent)
		mCmdList->EndQuery(mQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex * 2);

		return queryIndex;
	}

	void StopEvent(unsigned int queryIndex)
	{
		mCmdList->EndQuery(mQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex * 2 + 1);
		mCmdList->ResolveQueryData(mQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex * 2, 2, mReadbackBuffer, queryIndex * 2 * sizeof(uint64_t));
	}

private:
	ID3D12GraphicsCommandList4* mCmdList = nullptr;
	ID3D12QueryHeap* mQueryHeap = nullptr;
	ID3D12Resource* mReadbackBuffer = nullptr;
	std::map<std::wstring, float> mCachedTimes;

	unsigned int mQueryCount = 0;
	static const unsigned int cMaxQueryCount = 32;
	std::wstring mQueryNames[cMaxQueryCount];
	uint64_t mLastGpuFrequency = 0;
};

class ProfileEvent
{
public:

	ProfileEvent(Profiler* profiler, std::wstring name) {
		mId = profiler->StartEvent(name);
		mProfiler = profiler;
	}

	~ProfileEvent() {
		mProfiler->StopEvent(mId);
	}
private:

	unsigned int mId;
	Profiler* mProfiler;
};

#define TOKENPASTE(x, y) x ## y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)
#define PROFILE(_name) ProfileEvent TOKENPASTE2(profilingContext, __COUNTER__)(&mProfiler, _name);

class MLPZen
{
public:
	void Initialize(HWND hwnd)
	{
		mDpiScale = utils::getDpiScale(hwnd);
		mNNData.frameNumber = 0;
		mReloadShaders = false;
		mNNNeedsInitialization = true;

		mShaderCompiler.Initialize();

		initializeDx12(hwnd);
		initImGui(hwnd);

		mProfiler.Initialize(mDevice);
	}

	void ReloadShaders() {
		mReloadShaders = true;
	}

	void TrainingStep()
	{
		mTrainingStep = true;
	}

	bool Update(HWND hwnd, const float elapsedTime)
	{
		std::string profiling = mProfiler.BeginFrame(mCmdList, mCmdQueue);

		// Start the Dear ImGui frame
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		// Position the ImGui window on start to the right
		if (mNNData.frameNumber == 0) {
			ImGui::SetNextWindowPos(ImVec2(frameWidth - 450 - 20, 20));
			ImGui::SetNextWindowSize(ImVec2(450, frameHeight - 40.f));
		}

		float learningRate = 0.0f;
		
		if (mTargetImageLoaded)
		{

			if (mNNNeedsInitialization)
			{
				mTrainingSteps = 0;
				mRestarts = 0;
				mTCur = 0;
			}

			// Update constant buffer
			{
				mNNData.outputWidth = mTargetWidth;
				mNNData.outputHeight = mTargetHeight;

				const size_t gradientRecordsPerThread = mEnableHighPrecisionGradient ? (mBatchSize / BACKPROP_THREADGROUP_SIZE) : mBatchSize;
				mNNData.mlpGradientScaler = 1.0f / float(gradientRecordsPerThread);
				updateHashgridGradientScaler();
				mNNData.hgGradientScaler = mHashgridGradientScaler;

				mNNData.mainTextureIndex = mTextureToLearn;
				mNNData.bubblesZ = mBubblesZPlane;
				mNNData.mlpCount = mMlpCount;
				
				uploadConstantBuffer();

				mNNData.frameNumber++;
			}

			// Setup root signature
			{
				ID3D12DescriptorHeap* ppHeaps[] = { mDescriptorHeap };
				mCmdList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
				mCmdList->SetComputeRootSignature(mGlobalRootSignature);
				mCmdList->SetComputeRootDescriptorTable(UINT(RootParameterIndex::CbvSrvUavs), mDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
			}

			// Clear the screen
			D3D12_CPU_DESCRIPTOR_HANDLE destination = getCurrentBackBufferView();
			const glm::vec4 black = glm::vec4(0, 0, 0, 0);
			transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
			mCmdList->ClearRenderTargetView(destination, &black.x, 0, nullptr);

			// Specify the buffers we are going to render to - destination (back buffer)
			D3D12_CPU_DESCRIPTOR_HANDLE depthStencilBufferViewHandle = mDsvHeap->GetCPUDescriptorHandleForHeapStart();
			mCmdList->OMSetRenderTargets(1, &destination, true, &depthStencilBufferViewHandle);

			// Run neural network
			{
				PROFILE(L"NN Total");

				// Show reference image
				{
					uint32_t dispatchWidth = utils::divRoundUp(mTargetWidth, REFOUTPUT_THREADGROUP_SIZE);
					uint32_t dispatchHeight = utils::divRoundUp(mTargetHeight, REFOUTPUT_THREADGROUP_SIZE);

					dispatchCompute2D(mReferenceOutputPSO, dispatchWidth, dispatchHeight);
				}

				// Initialize network weights
				if (mNNNeedsInitialization)
				{
					mNNNeedsInitialization = false;

					{
						PROFILE(L"Initialize");

						transitionBarrier(mNNParametersBuffer, mMlpCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
						transitionBarrier(mNNParametersBackpropBuffer, mMlpCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
						transitionBarrier(mNNHashgridBuffer, mMlpCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
						uavBarrier(mNNGradientBuffer, mMlpCount);
						uavBarrier(mNNAdamDataBuffer, mMlpCount);

						uint32_t useNormalDistribution = 0;
						uint32_t useHeStrategy = 0;
						uint32_t useXavierStrategy = 0;
						uint32_t useLeCunStrategy = 0;

						if (mInitializationType == InitializationType::HeGaussian)
						{
							useNormalDistribution = 1;
							useHeStrategy = 1;
						}
						else if (mInitializationType == InitializationType::HeUniform)
						{
							useNormalDistribution = 0;
							useHeStrategy = 1;
						}
						else if (mInitializationType == InitializationType::XavierGaussian)
						{
							useNormalDistribution = 1;
							useXavierStrategy = 1;
						}
						else if (mInitializationType == InitializationType::XavierUniform)
						{
							useNormalDistribution = 0;
							useXavierStrategy = 1;
						}
						else if (mInitializationType == InitializationType::LeCunGaussian)
						{
							useNormalDistribution = 1;
							useLeCunStrategy = 1;
						}
						else if (mInitializationType == InitializationType::LeCunUniform)
						{
							useNormalDistribution = 0;
							useLeCunStrategy = 1;
						}

						uint32_t rootConstants[12] = { 0, 0, 0, 0, 0, 0, 0, useNormalDistribution, useHeStrategy, useXavierStrategy, 0, 0 };
						{
							mCmdList->SetComputeRoot32BitConstants(1, _countof(rootConstants), &rootConstants, 0);
						}

						size_t hashgridEncodingQuartetCount = (mInputEncodingType == InputEncodingType::HashGrid ? (mHashgridTotalParameters / 4) : 0);
						const uint32_t dispatchWidth = utils::divRoundUp(glm::max(mNNParametersQuartetCount, hashgridEncodingQuartetCount), INIT_THREADGROUP_SIZE);
						const uint32_t dispatchHeight = utils::divRoundUp(mMlpCount, 1);

						dispatchCompute2D(mInitializationPSO, dispatchWidth, dispatchHeight);

						// Return trainable data to the SRV state used by inference and backpropagation.
						transitionBarrier(mNNParametersBuffer, mMlpCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
						transitionBarrier(mNNParametersBackpropBuffer, mMlpCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
						transitionBarrier(mNNHashgridBuffer, mMlpCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
					}
				}

				// Run training
				if (mEnableTraining || mTrainingStep)
				{
					PROFILE(L"Training");

					if (!mLimitTrainingSteps || mTrainingSteps < mMaxTrainingSteps)
					{
						for (int step = 0; step < mTrainingStepsPerFrame; step++)
						{
							uint32_t rootConstants[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mFreezeMLP, mFreezeHashgrid };
							{
								// Set training step index
								rootConstants[0] = mTrainingSteps;

								// Set Adam parameters
								const float adamEpsilon = 0.00000001f;
								const float adamBeta1 = 0.9f;
								const float adamBeta2 = 0.999f;
								const float adamBeta1T = glm::pow(adamBeta1, mTrainingSteps + 1);
								const float adamBeta2T = glm::pow(adamBeta2, mTrainingSteps + 1);
								memcpy(&rootConstants[1], &adamEpsilon, sizeof(float));
								memcpy(&rootConstants[2], &adamBeta1, sizeof(float));
								memcpy(&rootConstants[3], &adamBeta2, sizeof(float));
								memcpy(&rootConstants[4], &adamBeta1T, sizeof(float));
								memcpy(&rootConstants[5], &adamBeta2T, sizeof(float));

								// Set learning rate
								learningRate = getLearningRate();
								memcpy(&rootConstants[6], &learningRate, sizeof(float));

								mCmdList->SetComputeRoot32BitConstants(1, _countof(rootConstants), &rootConstants, 0);
							}

							{
								PROFILE(L" Backpropagation");

								uavBarrier(mNNGradientBuffer, mMlpCount);
								if (mCalculateGlobalLoss)
								{
									uavBarrier(mLossDataBuffer);
								}

								const uint32_t dispatchWidth = utils::divRoundUp(mBatchSize, BACKPROP_THREADGROUP_SIZE);
								const uint32_t dispatchHeight = utils::divRoundUp(mMlpCount, 1);

								dispatchCompute2D(mBackpropagationPSO, dispatchWidth, dispatchHeight);

								// Copy loss data to readback buffer
								if (mCalculateGlobalLoss)
								{
									transitionBarrier(mLossDataBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
									mCmdList->CopyResource(mLossDataReadbackBuffer, mLossDataBuffer);
									transitionBarrier(mLossDataBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
								}
							}

							// Optimize network (apply gradients)
							{
								PROFILE(L" Optimization");

								uavBarrier(mNNGradientBuffer, mMlpCount);
								uavBarrier(mNNAdamDataBuffer, mMlpCount);
								transitionBarrier(mNNParametersBuffer, mMlpCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
								transitionBarrier(mNNParametersBackpropBuffer, mMlpCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
								transitionBarrier(mNNHashgridBuffer, mMlpCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

								size_t hashgridEncodingQuartetCount = (mInputEncodingType == InputEncodingType::HashGrid ? (mHashgridTotalParameters / 4) : 0);
								const uint32_t dispatchWidth = utils::divRoundUp(mNNParametersQuartetCount + hashgridEncodingQuartetCount, OPTIMIZATION_THREADGROUP_SIZE);
								const uint32_t dispatchHeight = utils::divRoundUp(mMlpCount, 1);

								dispatchCompute2D(mOptimizationPSO, dispatchWidth, dispatchHeight);

								transitionBarrier(mNNParametersBuffer, mMlpCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
								transitionBarrier(mNNParametersBackpropBuffer, mMlpCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
								transitionBarrier(mNNHashgridBuffer, mMlpCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
								uavBarrier(mNNAdamDataBuffer, mMlpCount);
							}
							mTrainingSteps++;
						}

						mTrainingStep = false;
					}
				}

				// Run inference
				{
					PROFILE(L"Inference");

					uint32_t dispatchWidth = utils::divRoundUp(mTargetWidth, INFERENCE_THREADGROUP_SIZE);
					uint32_t dispatchHeight = utils::divRoundUp(mTargetHeight, INFERENCE_THREADGROUP_SIZE);

					dispatchCompute2D(mInferencePSO, dispatchWidth, dispatchHeight);
				}
			}

			// Copy the final output and target texture to the back buffer
			{
				transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
				transitionBarrier(mOutputBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
				transitionBarrier(mReferenceBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

				// Copy Final output (inference result)
				{
					CD3DX12_TEXTURE_COPY_LOCATION dest(mBackBuffer[mCurrentFrameIndex]);
					CD3DX12_TEXTURE_COPY_LOCATION src(mOutputBuffer);
					CD3DX12_BOX box(0, 0, mTargetWidth, mTargetHeight);
					mCmdList->CopyTextureRegion(&dest, 512 + 250, 100, 0, &src, &box);
				}

				// Copy Target texture (training image)
				{
					CD3DX12_TEXTURE_COPY_LOCATION dest(mBackBuffer[mCurrentFrameIndex]);
					CD3DX12_TEXTURE_COPY_LOCATION src(mReferenceBuffer);
					CD3DX12_BOX box(0, 0, mTargetWidth, mTargetHeight);
					mCmdList->CopyTextureRegion(&dest, 100, 100, 0, &src, &box);
				}

				transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
				transitionBarrier(mOutputBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
				transitionBarrier(mReferenceBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			}
		}

		// Fetch and calculate NN loss
		float currentLoss[MAX_MLPS];
		unsigned int totalLossRecords[MAX_MLPS];
		memset(currentLoss, 0, sizeof(currentLoss));
		memset(totalLossRecords, 0, sizeof(totalLossRecords));
		if (mCalculateGlobalLoss)
		{
			D3D12_RANGE range;
			range.Begin = SIZE_T(0);
			range.End = SIZE_T(mLossBufferSize * sizeof(int));
			void* mappedData = nullptr;
			mLossDataReadbackBuffer->Map(0, &range, &mappedData);
			unsigned int* outputData = ((unsigned int*)mappedData);

			for (int mlp = 0; mlp < mMlpCount; mlp++)
			{
				const float totalLoss = unpackFloatPositive(outputData[mlp * 2 + 0]);
				totalLossRecords[mlp] = outputData[mlp * 2 + 1];
				if (totalLossRecords[mlp] > 0)
				{
					currentLoss[mlp] = totalLoss / float(totalLossRecords[mlp]);
				}
			}
			mLossDataReadbackBuffer->Unmap(0, nullptr);
		}

		// Render ImGUI
		{
			imgui(profiling, elapsedTime, totalLossRecords, currentLoss, learningRate);
			mCmdList->SetDescriptorHeaps(1, &imguiSrvDescHeap);
			ImGui::Render();
			ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), mCmdList);
		}

		transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);

		// End frame - process cmd. list and move to next frame
		submitCmdList();
		waitForGPU();
		present();
		moveToNextFrame();
		resetCommandList();

		// Reload shaders here if needed
		if (mReloadShaders) {
			createComputePasses();
			createNNBuffers();
			mReloadShaders = false;
		}

		return true;
	}

	void Cleanup()
	{
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}

private:

	void initializeDx12(HWND hwnd) {

		for (int i = 0; i < kMaxFramesInFlight; i++) {
			mFenceValues[i] = 0;
			mBackBuffer[i] = nullptr;
			mCmdAlloc[i] = nullptr;
		}

		// Create DXGI factory
		HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&mDxgiFactory));
		utils::validate(hr, L"Error: failed to create DXGI factory!");

		const D3D_FEATURE_LEVEL kDx12FeatureLevel = D3D_FEATURE_LEVEL_12_1;

#ifdef _DEBUG
		// Enable the D3D12 debug layer.
		{
			if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&mDebugController))))
			{
				mDebugController->EnableDebugLayer();
			}
		}
#endif

		// Find a suitable adapter to run D3D
		int i = 0;
		IDXGIAdapter1* adapter = nullptr;
		DXGI_ADAPTER_DESC1 selectedAdapterDesc = {};
		while (SUCCEEDED(mDxgiFactory->EnumAdapters1(i, &adapter)))
		{
			ID3D12Device5* tempDevice = nullptr;

			if (SUCCEEDED(D3D12CreateDevice(adapter, kDx12FeatureLevel, IID_PPV_ARGS(&tempDevice))))
			{
				D3D12_FEATURE_DATA_D3D12_OPTIONS5 features = {};
				HRESULT hr = tempDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features, sizeof(features));
				if (SUCCEEDED(hr))
				{
					DXGI_ADAPTER_DESC1 desc;
					adapter->GetDesc1(&desc);

					if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
						selectedAdapterDesc = desc;
						SAFE_RELEASE(tempDevice);
						SAFE_RELEASE(adapter);
						break;
					}
				}
			}

			SAFE_RELEASE(tempDevice);
			SAFE_RELEASE(adapter);
			i++;
		}

		// Create D3D Device
		{
			HRESULT hr = mDxgiFactory->EnumAdapterByLuid(selectedAdapterDesc.AdapterLuid, IID_PPV_ARGS(&mAdapter));
			utils::validate(hr, L"Error: failed to enumerate selected adapter by luid!");

			hr = D3D12CreateDevice(mAdapter, kDx12FeatureLevel, IID_PPV_ARGS(&mDevice));
			utils::validate(hr, L"Error: failed to create D3D device!");

			mAdapterName = selectedAdapterDesc.Description;
		}

		// Create command queue
		{
			D3D12_COMMAND_QUEUE_DESC desc = {};
			desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
			desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

			HRESULT hr = mDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&mCmdQueue));
			utils::validate(hr, L"Error: failed to create command queue!");
		}

		// Create a command allocator for each frame
		{
			for (UINT n = 0; n < kMaxFramesInFlight; n++)
			{
				HRESULT hr = mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCmdAlloc[n]));
				utils::validate(hr, L"Error: failed to create the command allocator!");
			}
		}

		// Create Command List
		{
			// Create the command list
			HRESULT hr = mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCmdAlloc[mCurrentFrameIndex], nullptr, IID_PPV_ARGS(&mCmdList));
			hr = mCmdList->Close();
			utils::validate(hr, L"Error: failed to create the command list!");

			resetCommandList();
		}

		// Create fence
		{
			HRESULT hr = mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence));
			utils::validate(hr, L"Error: failed to create fence!");

			mFenceValues[mCurrentFrameIndex]++;

			// Create an event handle to use for frame synchronization
			mFenceEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);
			if (mFenceEvent == nullptr)
			{
				hr = HRESULT_FROM_WIN32(GetLastError());
				utils::validate(hr, L"Error: failed to create fence event!");
			}
		}

		// Create swap chain
		{
			// Check for tearing support
			BOOL allowTearing = FALSE;
			HRESULT hr = mDxgiFactory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing));
			utils::validate(hr, L"Error: failed to create DXGI factory!");

			mIsTearingSupport = SUCCEEDED(hr) && allowTearing;

			// Describe the swap chain
			DXGI_SWAP_CHAIN_DESC1 desc = {};
			desc.BufferCount = kMaxFramesInFlight;
			desc.Width = frameWidth;
			desc.Height = frameHeight;
			desc.Format = mBackBufferFormat;
			desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
			desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
			desc.SampleDesc.Count = 1;
			desc.SampleDesc.Quality = 0;
			desc.Flags = mIsTearingSupport ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

			IDXGISwapChain1* tempSwapChain;

			// Create the swap chain
			hr = mDxgiFactory->CreateSwapChainForHwnd(mCmdQueue, hwnd, &desc, 0, 0, &tempSwapChain);
			utils::validate(hr, L"Error: failed to create swap chain!");

			if (mIsTearingSupport)
			{
				// When tearing support is enabled we will handle ALT+Enter key presses in the
				// window message loop rather than let DXGI handle it by calling SetFullscreenState.
				mDxgiFactory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
				utils::validate(hr, L"Error: failed to make window association!");
			}

			// Get the swap chain interface
			hr = tempSwapChain->QueryInterface(IID_PPV_ARGS(&mSwapChain));
			utils::validate(hr, L"Error: failed to cast swap chain!");

			SAFE_RELEASE(tempSwapChain);
			mCurrentFrameIndex = mSwapChain->GetCurrentBackBufferIndex();
		}

		// Create RTV heap
		{
			// Describe the RTV heap
			D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
			rtvDesc.NumDescriptors = kMaxFramesInFlight;
			rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

			// Create the RTV heap
			HRESULT hr = mDevice->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&mRtvHeap));
			utils::validate(hr, L"Error: failed to create RTV descriptor heap!");

			mRtvDescSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}

		// Create DSV heap
		{
			D3D12_DESCRIPTOR_HEAP_DESC heapDescription;
			heapDescription.NumDescriptors = 1;
			heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
			heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			heapDescription.NodeMask = 0;

			HRESULT result = mDevice->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&mDsvHeap));
			utils::validate(result, L"Error: failed to create DSV heap!");
		}

		// Create back buffer
		{
			HRESULT hr;
			D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();

			// Create a RTV for each back buffer
			for (unsigned int n = 0; n < kMaxFramesInFlight; n++)
			{
				hr = mSwapChain->GetBuffer(n, IID_PPV_ARGS(&mBackBuffer[n]));
				utils::validate(hr, L"Error: failed to get swap chain buffer!");

				mDevice->CreateRenderTargetView(mBackBuffer[n], nullptr, rtvHandle);

				rtvHandle.ptr += mRtvDescSize;
			}
		}

		// Create depth stencil buffer
		{
			// Create the depth/stencil buffer and view.
			D3D12_RESOURCE_DESC depthStencilDesc;
			depthStencilDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			depthStencilDesc.Alignment = 0;
			depthStencilDesc.Width = frameWidth;
			depthStencilDesc.Height = frameHeight;
			depthStencilDesc.DepthOrArraySize = 1;
			depthStencilDesc.MipLevels = 1;
			depthStencilDesc.Format = mDepthBufferFormat;
			depthStencilDesc.SampleDesc.Count = 1;
			depthStencilDesc.SampleDesc.Quality = 0;
			depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

			D3D12_CLEAR_VALUE optClear;
			optClear.Format = mDepthBufferFormat;
			optClear.DepthStencil.Depth = 1.0f;
			optClear.DepthStencil.Stencil = 0;

			mDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_NONE,
				&depthStencilDesc,
				D3D12_RESOURCE_STATE_COMMON,
				&optClear,
				IID_PPV_ARGS(&mDepthStencilBuffer));

			// Create descriptor to mip level 0 of entire resource	using the format of the resource.
			D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
			dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
			dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Format = mDepthBufferFormat;
			dsvDesc.Texture2D.MipSlice = 0;

			CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(mDsvHeap->GetCPUDescriptorHandleForHeapStart());

			mDevice->CreateDepthStencilView(mDepthStencilBuffer, &dsvDesc, hDescriptor);

			// Transition the resource from its initial state to be used as a depth buffer.
			transitionBarrier(mDepthStencilBuffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
		}

		// Create root signature
		{
			mGlobalRootSignature = createGlobalRootSignature();
		}

		// Create descriptor heap
		{
			// Describe the CBV/SRV/UAV heap
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.NumDescriptors = UINT(DescriptorHeapConstants::Total);
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

			// Create the descriptor heap
			HRESULT hr = mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mDescriptorHeap));
			utils::validate(hr, L"Error: failed to create descriptor heap!");

			// Get the descriptor heap handle and increment size
			mCbvSrvUavDescSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		// Disable FP16 if not supported
		{
			// Check if the adapter supports 16-bit types
			D3D12_FEATURE_DATA_D3D12_OPTIONS4 features = {};
			HRESULT hr = mDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &features, sizeof(features));

			if (!features.Native16BitShaderOpsSupported)
			{
				mUseFP16Hashgrid = false;
				mUseFP16MLP = false;
			}
		}

		// Create shaders and compute passes
		{
			createComputePasses();
		}

		// Create output and reference buffer
		{
			// Max size is 512x512 pixels
			createTexture(mDevice, 512, 512, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mOutputBuffer);
			createTexture(mDevice, 512, 512, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mReferenceBuffer);
		}

		// Create constant buffer
		{
			mNNDataCBSize = ALIGN(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, sizeof(mNNData));
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, mNNDataCBSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, &mNNDataCB);

			UINT64 uploadBufferSize = GetRequiredIntermediateSize(mNNDataCB, 0, 1);
			uploadBufferSize = ALIGN(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, uploadBufferSize);
			createBuffer(mDevice, D3D12_HEAP_TYPE_UPLOAD, 0, uploadBufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &mNNDataCBUpload);
		}

		// Create buffer for loss
		{
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, mLossBufferSize * sizeof(int), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mLossDataBuffer);
			createBuffer(mDevice, D3D12_HEAP_TYPE_READBACK, 0, mLossBufferSize * sizeof(int), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &mLossDataReadbackBuffer);

			// The UI can read this buffer before the first GPU copy has completed.
			D3D12_RANGE readRange = { 0, 0 };
			void* mappedData = nullptr;
			HRESULT hr = mLossDataReadbackBuffer->Map(0, &readRange, &mappedData);
			utils::validate(hr, L"Error: failed to initialize loss readback buffer!");
			if (SUCCEEDED(hr))
			{
				memset(mappedData, 0, mLossBufferSize * sizeof(int));
				D3D12_RANGE writtenRange = { 0, SIZE_T(mLossBufferSize * sizeof(int)) };
				mLossDataReadbackBuffer->Unmap(0, &writtenRange);
			}
		}

		// Create target image used for training
		{
			std::wstring assetsFolder = utils::getExePath() + L"assets\\";

			if (!utils::directoryExists(utils::wstringToString(assetsFolder)))
			{
				assetsFolder = utils::stringToWstring(ASSETS_DIR);
			}

			loadTargetImage(assetsFolder + L"a1.jpg", 0);
			loadTargetImage(assetsFolder + L"a3.jpg", 1);
			loadTargetImage(assetsFolder + L"a4.jpg", 2);
			loadTargetImage(assetsFolder + L"mandrill.png", 3);
		}

		// Fill descriptor heap
		{
			// Create the NNData CBV
			{
				D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
				cbvDesc.SizeInBytes = mNNDataCBSize;
				cbvDesc.BufferLocation = mNNDataCB->GetGPUVirtualAddress();
				mDevice->CreateConstantBufferView(&cbvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNDataCB)));
			}

			// Create UAV for loss data buffer
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.Buffer.NumElements = mLossBufferSize;
				uavDesc.Buffer.StructureByteStride = sizeof(int);
				uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

				mDevice->CreateUnorderedAccessView(mLossDataBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::LossData)));
			}

			// Create the DXR output buffer UAV
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				mDevice->CreateUnorderedAccessView(mOutputBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::Output)));
			}
			
			// Create the reference buffer UAV
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				mDevice->CreateUnorderedAccessView(mReferenceBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::Reference)));
			}

		}

		createNNBuffers();
	}

	int getHashgridSizeBytes()
	{
		return (mHashgridTotalAllocatedParameters * (mUseFP16Hashgrid ? 2 : 4));
	}

	int getExactMLPSizeBytes()
	{
		int neuronsPerLayer[MAX_LAYERS];
		getNeuronsPerLayer(mInputDimensions, neuronsPerLayer);

		int total = 0;
		for (int i = 1; i < mLayerCount; i++)
		{
			// Add weights to previous layer
			total += (neuronsPerLayer[i] * neuronsPerLayer[i - 1]);

			// Add biases
			total += neuronsPerLayer[i];
		}

		return total * (mUseFP16MLP ? 2 : 4);
	}

	void imgui(const std::string profiling, const float elapsedTime, const uint totalLossRecords[MAX_MLPS], const float currentLoss[MAX_MLPS], const float learningRate)
	{
		ImGui::Begin("MLP Zen");

		ImGui::Text("Adapter: %ls", mAdapterName.c_str());

		static float elapsedTimeWeighted = 0.0f;
		static const float alpha = 0.95f;
		elapsedTimeWeighted = alpha * elapsedTimeWeighted + (1.0f - alpha) * elapsedTime;
		ImGui::Text("Frame Time: %.02fms", elapsedTimeWeighted);

		ImGui::Checkbox("VSync", &mEnableVSync);
		ImGui::SameLine();
		if (ImGui::Button("[F5]Reload Shaders")) mReloadShaders = true;

		// Example app settings ===============================
		{
			ImGui::Separator();
			ImGui::Text("Example settings:");
			ImGui::Separator();
			ImGui::Indent(16.0f);

			if (ImGui::Combo("Example", (int*)&mExampleType, "Learn Cosine\0Learn Image\0Learn 4 Images\0Bubbles 3D\0\0")) {
				mReloadShaders = true;
				mNNNeedsInitialization = true;
				mMlpCount = ((mExampleType == ExampleType::LearnFourImages) ? 4 : 1);

				if (mExampleType == ExampleType::LearnCosine)
				{
					mInputEncodingType = InputEncodingType::OneBlob;
				}
				else
				{
					mInputEncodingType = InputEncodingType::HashGrid;
				}

				if (mExampleType == ExampleType::Bubbles3D)
				{
					mHashgridMapSize = 32768;
					mInputDimensions = 3;
					mOutputActivationFunctionType = ActivationFunctionType::LeakyRelu;
				}
				else
				{
					mHashgridMapSize = 4096;
					mInputDimensions = 2;
					mOutputActivationFunctionType = ActivationFunctionType::Sigmoid;
				}
			}

			if (mExampleType == ExampleType::LearnImage)
			{
				if (ImGui::SliderInt("Texture to Learn", &mTextureToLearn, 0, 3)) {
					mNNNeedsInitialization = true;
				}
			}

			if (mExampleType == ExampleType::Bubbles3D)
			{
				if (ImGui::SliderFloat("Z Plane", &mBubblesZPlane, 0.0f, 1.0f));
			}

			ImGui::Indent(-16.0f);
		}

		// Training settings ===============================
		{
			ImGui::Separator();
			ImGui::Text("Training:");
			ImGui::Separator();
			ImGui::Indent(16.0f);

			ImGui::Checkbox("Enable Training", &mEnableTraining);
			if (!mEnableTraining) {
				if (ImGui::Button("Do Training Step [F6]")) mTrainingStep = true;
			}

			ImGui::SliderInt("Training Steps per Frame", &mTrainingStepsPerFrame, 1, 4);
			ImGui::Checkbox("Limit Training Steps", &mLimitTrainingSteps);
			if (mLimitTrainingSteps)
			{
				ImGui::SliderInt("Max. Training Steps", &mMaxTrainingSteps, 1, 64 * 1024);

			}
			ImGui::Text("Training Steps: %i", mTrainingSteps);

			ImGui::Checkbox("Freeze MLP", &mFreezeMLP);
			ImGui::Checkbox("Freeze Hash-grid", &mFreezeHashgrid);
			if (ImGui::Button("Initialize Weights")) mNNNeedsInitialization = true;

			ImGui::SliderInt("Batch Size", &mBatchSize, 1, 64 * 1024);

			ImGui::SliderFloat("Learning Rate", &mGlobalLearningRate, 0.0f, mOptimizerType == OptimizerType::Adam ? 0.01f : 0.1f);

			ImGui::Text("Current Learning Rate: %.06f", learningRate);
			if (ImGui::Combo("Learning Rate Schedule", (int*)&mLearningRateSchedule, "Fixed\0Linear\0Cosine Annealing\0\0")) {
				mNNNeedsInitialization = true;
			}

			if (mLearningRateSchedule != LearningRateScheduleType::Fixed)
			{
				ImGui::SliderFloat("Max. Learning Rate", &mMaxLearningRate, 0.0f, 0.05f);
				if (mLearningRateSchedule == LearningRateScheduleType::Linear)
				{
					ImGui::SliderInt("Decay Steps", &mDecaySteps, 1, 1000);
				}
				else if (mLearningRateSchedule == LearningRateScheduleType::CosineAnnealing)
				{
					ImGui::SliderInt("Base Ti", &mBaseTi, 1, 1000);
				}
			}
			if (ImGui::Checkbox("Clip Gradient", &mEnableGradientClipping))
			{
				mReloadShaders = true;
			}
			if (ImGui::Checkbox("High-precision Gradient for MLP (Wave ops)", &mEnableHighPrecisionGradient))
			{
				mReloadShaders = true;
			}

			if (ImGui::Combo("Initialization", (int*)&mInitializationType, "He Gaussian\0He Uniform\0Xavier Gaussian\0Xavier Uniform\0LeCun Gaussian\00LeCun Uniform\0\0")) {
				mNNNeedsInitialization = true;
			}

			if (ImGui::Combo("Optimizer", (int*)&mOptimizerType, "SGD\0Adam\0\0")) {
				mReloadShaders = true;
			}

			ImGui::Indent(-16.0f);
		}

		// MLP architecture settings ===============================
		{
			ImGui::Separator();
			ImGui::Text("MLP Architecture:");
			ImGui::Separator();
			ImGui::Indent(16.0f);

			if (ImGui::Checkbox("Use FP16 for MLP", &mUseFP16MLP))
			{
				mReloadShaders = true;
				mNNNeedsInitialization = true;
			}

			if (ImGui::Combo("Output Activation Function", (int*)&mOutputActivationFunctionType, "Leaky ReLU\0Sigmoid\0None\0\0")) {
				mReloadShaders = true;
				mNNNeedsInitialization = true;
			}

			if (ImGui::SliderInt("Layer Count", &mNewLayerCount, 2, MAX_LAYERS)) {
				mNNArchitectureDirty = true;
			}

			if (ImGui::SliderInt("Neurons Per Layer", &mNewNeuronsPerLayer, 4, mMaxNeuronsPerLayerLimit)) {
				mNewNeuronsPerLayer = ALIGN(4, mNewNeuronsPerLayer);
				mNNArchitectureDirty = true;
			}

			ImGui::Text("MLP size: %.01fkB", mExactMlpSizeBytes / 1024.0f);

			ImGui::Indent(-16.0f);
		}

		// Input encoding settings ===============================
		{
			ImGui::Separator();
			ImGui::Text("Input Encoding:");
			ImGui::Separator();
			ImGui::Indent(16.0f);

			if (ImGui::Combo("Input Encoding", (int*)&mInputEncodingType, "Identity\0Frequency\0OneBlob\0HashGrid\0\0")) {
				mReloadShaders = true;
				mNNNeedsInitialization = true;
			}

			if (mInputEncodingType == InputEncodingType::Frequency) {
				if (ImGui::SliderInt("Frequencies", &mFrequencies, 2, mFrequenciesLimit)) {
					mFrequencies = ALIGN(2, mFrequencies);
					mNNArchitectureDirty = true;
				}
			}

			if (mInputEncodingType == InputEncodingType::OneBlob) {
				if (ImGui::SliderInt("OneBlob Bins", &mOneBlobBins, 4, mOneBlobBinsLimit)) {
					mOneBlobBins = ALIGN(4, mOneBlobBins);
					mNNArchitectureDirty = true;
				}
			}

			if (mInputEncodingType == InputEncodingType::HashGrid) {
				if (ImGui::Checkbox("Use FP16 for hash-grid", &mUseFP16Hashgrid))
				{
					mNNArchitectureDirty = true;
					mHashgridSettingsChanged = true;
				}
				if (ImGui::SliderInt("Levels", &mHashgridLevels, 1, HG_MAX_LEVELS)) {
					mNNArchitectureDirty = true;
					mHashgridSettingsChanged = true;
				}
				if (ImGui::Combo("Feature Vector Length", (int*)&mHashgridFeatureVectorLength, "Four\0Eight\0\0")) {
					mNNArchitectureDirty = true;
					mHashgridSettingsChanged = true;
				}
				if (ImGui::SliderInt("Base Resolution", &mHashgridBaseResolution, 2, 32)) {
					mNNArchitectureDirty = true;
					mHashgridSettingsChanged = true;
				}
				if (ImGui::SliderInt("Max. Level Feature Vectors", &mHashgridMapSize, 16, 64 * 1024)) {
					mNNArchitectureDirty = true;
					mHashgridSettingsChanged = true;
				}
				const float hashGridSizekB = getHashgridSizeBytes() / 1024.0f;
				ImGui::Text("Hash-grid size: %.01fkB", hashGridSizekB);
			}

			ImGui::Indent(-16.0f);
		}
		
		if (mNNArchitectureDirty) {
			ImGui::Text("Press Apply for changes to take effect");
			if (ImGui::Button("Apply"))
			{
				mReloadShaders = true;
				mNNNeedsInitialization = true;
				mNNArchitectureDirty = false;
				mLayerCount = mNewLayerCount;
				mNeuronsPerLayer = ALIGN(4, mNewNeuronsPerLayer);
			}
		}

		// Loss reporting ===============================
		{
			ImGui::Separator();
			ImGui::Text("Loss:");
			ImGui::Separator();
			ImGui::Indent(16.0f);

			if (ImGui::Checkbox("Calculate Global Loss", &mCalculateGlobalLoss))
			{
				mReloadShaders = true;
			}

			if (mCalculateGlobalLoss)
			{
				ImGui::Indent(16.0f);
				for (int mlp = 0; mlp < mMlpCount; mlp++)
				{
					ImGui::Text("Training Records Created %i: %u", mlp, totalLossRecords[mlp]);
					const float alpha = 0.99f;
					static float smoothLoss[MAX_MLPS];
					smoothLoss[mlp] = alpha * smoothLoss[mlp] + (1.0f - alpha) * currentLoss[mlp];
					ImGui::Text("Current Loss %i: %.05f (%.03f)", mlp, currentLoss[mlp], smoothLoss[mlp]);
				}
				ImGui::Indent(-16.0f);
			}

			ImGui::Indent(-16.0f);
		}

		// Profiler output ===============================
		{
			ImGui::Separator();
			ImGui::Text("Profiler:");
			ImGui::Separator();
			ImGui::Indent(16.0f);

			ImGui::TextUnformatted(profiling.c_str());

			ImGui::Indent(-16.0f);
		}

		if (mReloadShaders)
		{
			ImGui::SetWindowFontScale(2.0f);
			ImGui::Text("RECOMPILING SHADERS!");
			ImGui::SetWindowFontScale(1.0f);
		}

		ImGui::End();
	}

	std::vector<uint32_t> getMemoryLayoutMap(const size_t itemCount)
	{
		struct Param
		{
			Param() {
				neuronFrom = 0;
				neuronTo = 0;
				layerFrom = 0;
			}

			Param(int from, int to, int l)
			{
				neuronFrom = from;
				neuronTo = to;
				layerFrom = l;
			}

			bool operator<(const Param& other) const
			{
				if (layerFrom != other.layerFrom)
					return layerFrom < other.layerFrom;
				if (neuronFrom != other.neuronFrom)
					return neuronFrom < other.neuronFrom;
				return neuronTo < other.neuronTo;
			}

			int neuronFrom;
			int neuronTo;
			int layerFrom;
		};

		int neuronsPerLayer[MAX_LAYERS];
		getNeuronsPerLayer(mInputDimensions, neuronsPerLayer);

		// Create a map of forward pass memory layout
		std::vector<Param> forwardLayout;
		std::map<Param, uint32_t> forwardMap;
		{
			forwardLayout.resize(itemCount);
			const uint32_t paramQuartetsPerLayer = (mMaxNeuronsPerLayer * (mMaxNeuronsPerLayer + 1)) / 4;

			for (int layer = 1; layer < mLayerCount; layer++)
			{
				uint32_t paramOffset = (paramQuartetsPerLayer * (layer - 1)) * 4;
				const uint32_t neuronQuartetCountCurrentLayer = neuronsPerLayer[layer] / 4;
				const uint32_t neuronQuartetCountPreviousLayer = neuronsPerLayer[layer - 1] / 4;
				for (int neuronQuartet = 0; neuronQuartet < neuronQuartetCountCurrentLayer; neuronQuartet++)
				{
					for (uint32_t previousNeuronQuartet = 0; previousNeuronQuartet < neuronQuartetCountPreviousLayer; previousNeuronQuartet++)
					{
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								Param temp = Param(previousNeuronQuartet * 4 + j, neuronQuartet * 4 + i, layer - 1);
								forwardMap[temp] = paramOffset;
								forwardLayout[paramOffset++] = temp;
							}
						}
					}

					paramOffset += 4;
				}
			}
		}

		// Create a map of backward pass memory layout
		std::vector<uint32_t> result;
		result.resize(itemCount, uint32_t(-1));
		std::vector<Param> backwardLayout;
		{
			backwardLayout.resize(itemCount);
			for (int layer = 0; layer < (mLayerCount - 1); layer++)
			{
				const uint32_t neuronQuartetCountCurrentLayer = neuronsPerLayer[layer] / 4;
				const uint32_t neuronQuartetCountNextLayer = neuronsPerLayer[layer + 1] / 4;
				uint32_t nextLayerWeightsIndex = (layer * mMaxNeuronsPerLayer * mMaxNeuronsPerLayer);

				for (uint32_t neuron = 0; neuron < neuronQuartetCountCurrentLayer; neuron++)
				{
					for (uint32_t nextNeuron = 0; nextNeuron < neuronQuartetCountNextLayer; nextNeuron++)
					{
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								Param temp = Param(neuron * 4 + i, nextNeuron * 4 + j, layer);

								// Find a mapping from forward layout to backward layout
								auto fwd = forwardMap.find(temp);
								if (fwd == forwardMap.end())
								{
									utils::validate(E_FAIL, L"Could not create memory layout mapping, this is a bug.");
								}

								const uint32_t forwardPassIndex = fwd->second;
								result[forwardPassIndex] = nextLayerWeightsIndex;

								backwardLayout[nextLayerWeightsIndex++] = temp;
							}
						}
					}
				}
			}
		}

		return result;
	}

	void createNNBuffers()
	{
		// Figure out number of parameters of the MLP (weight + biases)
		const UINT nnParametersQuartetsCount = ((mLayerCount - 1) * (mMaxNeuronsPerLayer * (mMaxNeuronsPerLayer + 1))) / 4;
		
		// Number of entries we need for Adam optimizer data
		const UINT nnAdamDataBufferItemsCount = (nnParametersQuartetsCount + (mHashgridTotalParameters / 4)); //< One AdamData entry per quartet of parameters

		// Figure out sizes of buffer items
		const UINT nnParameterSize = mUseFP16MLP ? 2 : 4;
		const UINT nnAdamDataSize = sizeof(AdamData);
		const UINT nnGradientItemSize = sizeof(int32_t);

		// Figure out sizes of buffers needed to be allocated
		const UINT nnParametersBufferSize = nnParametersQuartetsCount * 4 * nnParameterSize;   //< One float per parameter
		const UINT nnAdamDataBufferSize = nnAdamDataBufferItemsCount * nnAdamDataSize;
		const UINT nnGradientBufferItemsCount = ((nnParametersQuartetsCount * 4) + mHashgridTotalParameters); //< One float per parameter
		const UINT nnGradientBufferSize = nnGradientBufferItemsCount * nnGradientItemSize;
		const UINT nnParametersBackpropItemCount = mMaxNeuronsPerLayer * mMaxNeuronsPerLayer * (mLayerCount - 1); //< One float per weight
		const UINT nnParametersBackpropBufferSize = sizeof(float) * nnParametersBackpropItemCount;
		const UINT inputNeurons = getInputLayerNeuronCount(mInputDimensions);

		// Check if we need to reallocate buffers due to settings changes
		bool needsReallocation = 
			(mAllocatedInputNeuronsCount != inputNeurons) || 
			(mAllocatedParamSize != nnParameterSize) || 
			(mAllocatedMlpCount != mMlpCount) || 
			(!mNNParametersBuffer[0]) || 
			(mNNParametersQuartetCount != nnParametersQuartetsCount) ||
			(mHashgridTotalAllocatedParameters != mHashgridTotalParameters);
				
		if (needsReallocation)
		{
			// Remember allocated sizes
			mAllocatedMlpCount = mMlpCount;
			mAllocatedParamSize = nnParameterSize;
			mAllocatedInputNeuronsCount = inputNeurons;
			mNNParametersQuartetCount = nnParametersQuartetsCount;

			// Allocate for all MLPs
			for (int mlp = 0; mlp < mMlpCount; mlp++)
			{
				SAFE_RELEASE(mNNParametersBuffer[mlp]);
				SAFE_RELEASE(mNNAdamDataBuffer[mlp]);
				SAFE_RELEASE(mNNGradientBuffer[mlp]);
				SAFE_RELEASE(mNNParametersBackpropBuffer[mlp]);

				// Create buffers
				createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, nnParametersBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNParametersBuffer[mlp]);
				createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, nnParametersBackpropBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNParametersBackpropBuffer[mlp]);
				createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, nnAdamDataBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mNNAdamDataBuffer[mlp]);
				createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, nnGradientBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &mNNGradientBuffer[mlp]);

				// Create UAVs for NN parameters, Adam data and gradient
				{
					D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
					uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
					uavDesc.Format = DXGI_FORMAT_UNKNOWN;
					uavDesc.Buffer.NumElements = nnParametersQuartetsCount;
					uavDesc.Buffer.StructureByteStride = nnParameterSize * 4; //< float4 or half4
					uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

					mDevice->CreateUnorderedAccessView(mNNParametersBuffer[mlp], nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNParametersOutputBuffer) + mlp));
				}

				{
					D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
					uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
					uavDesc.Format = DXGI_FORMAT_UNKNOWN;
					uavDesc.Buffer.NumElements = nnParametersBackpropItemCount;
					uavDesc.Buffer.StructureByteStride = sizeof(float); //< float
					uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

					mDevice->CreateUnorderedAccessView(mNNParametersBackpropBuffer[mlp], nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNParametersBackpropOutputBuffer) + mlp));
				}

				{
					D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
					uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
					uavDesc.Format = DXGI_FORMAT_UNKNOWN;
					uavDesc.Buffer.NumElements = nnAdamDataBufferItemsCount;
					uavDesc.Buffer.StructureByteStride = nnAdamDataSize;
					uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

					mDevice->CreateUnorderedAccessView(mNNAdamDataBuffer[mlp], nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNAdamDataBuffer) + mlp));
				}

				{
					D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
					uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
					uavDesc.Format = DXGI_FORMAT_UNKNOWN;
					uavDesc.Buffer.NumElements = nnGradientBufferItemsCount / 4;
					uavDesc.Buffer.StructureByteStride = nnGradientItemSize * 4; //< int4
					uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

					mDevice->CreateUnorderedAccessView(mNNGradientBuffer[mlp], nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNGradientBuffer) + mlp));
				}

				// Create SRV for NN parameters
				{
					D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
					srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
					srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
					srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
					srvDesc.Buffer.StructureByteStride = 0;
					srvDesc.Buffer.FirstElement = 0;
					srvDesc.Buffer.NumElements = ((nnParametersQuartetsCount * 4) * nnParameterSize) / 4; //< For raw buffer, this must be number of 32-bit values, not bytes. Therefore we divide by 4
					srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
					
					mDevice->CreateShaderResourceView(mNNParametersBuffer[mlp], &srvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNParametersInputBuffer) + mlp));
				}

				{
					D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
					srvDesc.Format = DXGI_FORMAT_UNKNOWN;
					srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
					srvDesc.Buffer.FirstElement = 0;
					srvDesc.Buffer.NumElements = nnParametersBackpropItemCount / 4;
					srvDesc.Buffer.StructureByteStride = sizeof(float) * 4; //< float4
					srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

					mDevice->CreateShaderResourceView(mNNParametersBackpropBuffer[mlp], &srvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNParametersBackpropInputBuffer) + mlp));
				}
			}
		}
	
		// Create memory layout map
		// This is the mapping from forward pass memory layout to backprop pass layout
		{
			size_t mapItems = nnParametersQuartetsCount * 4;
			size_t mapSize = mapItems * sizeof(uint32_t);

			// Allocate buffer
			{
				SAFE_RELEASE(mNNMemoryMapBuffer);
				createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, mapSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST, &mNNMemoryMapBuffer);
			}

			// Upload map data to GPU
			std::vector<uint32_t> memoryMap = getMemoryLayoutMap(mapItems);
			{
				const UINT64 uploadBufferSize = GetRequiredIntermediateSize(mNNMemoryMapBuffer, 0, 1);

				// Create the upload heap
				SAFE_RELEASE(mNNMemoryMapUploadBuffer);
				createBuffer(mDevice, D3D12_HEAP_TYPE_UPLOAD, 0, uploadBufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &mNNMemoryMapUploadBuffer);

				D3D12_SUBRESOURCE_DATA bufferData = {};
				bufferData.pData = memoryMap.data();
				bufferData.RowPitch = uploadBufferSize;
				bufferData.SlicePitch = bufferData.RowPitch;

				// Schedule a copy from the upload heap to the Texture2D resource
				UINT64 uploadedBytes = UpdateSubresources(mCmdList, mNNMemoryMapBuffer, mNNMemoryMapUploadBuffer, 0, 0, 1, &bufferData);
				HRESULT hr = (uploadBufferSize == uploadedBytes ? S_OK : E_FAIL);
				utils::validate(hr, L"Error: failed to upload data via upload heap!");

				transitionBarrier(mNNMemoryMapBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

			}

			// Create SRV for memory layout map
			{
				D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
				srvDesc.Format = DXGI_FORMAT_UNKNOWN;
				srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
				srvDesc.Buffer.FirstElement = 0;
				srvDesc.Buffer.NumElements = nnParametersQuartetsCount;
				srvDesc.Buffer.StructureByteStride = sizeof(uint32_t) * 4;
				srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

				mDevice->CreateShaderResourceView(mNNMemoryMapBuffer, &srvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNMemoryMap)));
			}
		}

		// Allocate hashgrids
		{
			bool hashgridNeedsReallocation = needsReallocation || (mHashgridTotalAllocatedFP16 != mUseFP16Hashgrid) || (mHashgridTotalAllocatedParameters != mHashgridTotalParameters);

			if (mHashgridSettingsChanged || hashgridNeedsReallocation)
			{				
				mHashgridSettingsChanged = false;
				mHashgridTotalAllocatedParameters = mHashgridTotalParameters;
				mHashgridTotalAllocatedFP16 = mUseFP16Hashgrid;
				const UINT hashgridEntrySize = mUseFP16Hashgrid ? 2 : 4;
				const size_t hashgridBufferSize = hashgridEntrySize * mHashgridTotalParameters;

				for (int mlp = 0; mlp < mMlpCount; mlp++)
				{
					SAFE_RELEASE(mNNHashgridBuffer[mlp]);

					{
						createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, hashgridBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNHashgridBuffer[mlp]);
					}

					// Create UAV for hashgrid
					{
						D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
						uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
						uavDesc.Format = DXGI_FORMAT_UNKNOWN;
						uavDesc.Buffer.NumElements = mHashgridTotalParameters / 4;
						uavDesc.Buffer.StructureByteStride = hashgridEntrySize * 4; //< float4 or half4
						uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

						mDevice->CreateUnorderedAccessView(mNNHashgridBuffer[mlp], nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNHashgridOutputBuffer) + mlp));
					}

					// Create SRV for hashgrid
					{
						D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
						srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
						srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
						srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
						srvDesc.Buffer.StructureByteStride = 0;
						srvDesc.Buffer.FirstElement = 0;
						srvDesc.Buffer.NumElements = mHashgridTotalParameters * hashgridEntrySize / 4; //< For raw buffer, this must be number of 32-bit values, not bytes. Therefore we divide by 4
						srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

						mDevice->CreateShaderResourceView(mNNHashgridBuffer[mlp], &srvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNHashgridInputBuffer) + mlp));
					}
				}
			}
		}

		// Descriptor tables cover MAX_MLPS entries. Keep the unused entries valid for
		// resource-binding tiers that require every table entry to be populated.
		const DescriptorHeapConstants mlpDescriptorRanges[] = {
			DescriptorHeapConstants::NNParametersOutputBuffer,
			DescriptorHeapConstants::NNAdamDataBuffer,
			DescriptorHeapConstants::NNGradientBuffer,
			DescriptorHeapConstants::NNParametersBackpropOutputBuffer,
			DescriptorHeapConstants::NNParametersInputBuffer,
			DescriptorHeapConstants::NNParametersBackpropInputBuffer,
			DescriptorHeapConstants::NNHashgridInputBuffer,
			DescriptorHeapConstants::NNHashgridOutputBuffer
		};
		for (int mlp = mMlpCount; mlp < MAX_MLPS; mlp++)
		{
			for (const DescriptorHeapConstants rangeStart : mlpDescriptorRanges)
			{
				mDevice->CopyDescriptorsSimple(1,
					getDescriptorHandle(UINT(rangeStart) + mlp),
					getDescriptorHandle(UINT(rangeStart)),
					D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			}
		}

		// Update MLP size
		mExactMlpSizeBytes = getExactMLPSizeBytes();
	}

	void loadTargetImage(std::wstring filePath, int textureIndex) {
		
		if (filePath == L"") return;

		// Load image from file using STB
		int height, width, colorChannelsPerTexel;
		int requiredNumberOfChannels = 4;
		char* texData = (char*) stbi_load(utils::wstringToString(filePath).c_str(), &width, &height, &colorChannelsPerTexel, requiredNumberOfChannels);

		if (texData == nullptr || width == 0 || height == 0) {
			mTargetImageLoaded = false;
			return;
		}

		mTargetImageLoaded = true;
		mTargetWidth = width;
		mTargetHeight = height;

		// Allocate target texture
		{
			createTexture(mDevice, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &mTargetTextureBuffer[textureIndex]);
		}

		const UINT64 uploadBufferSize = GetRequiredIntermediateSize(mTargetTextureBuffer[textureIndex], 0, 1);

		// Create the upload heap
		SAFE_RELEASE(mTextureUploadBuffer[textureIndex]);
		createBuffer(mDevice, D3D12_HEAP_TYPE_UPLOAD, 0, uploadBufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &mTextureUploadBuffer[textureIndex]);

		D3D12_SUBRESOURCE_DATA textureData = {};
		textureData.pData = texData;
		textureData.RowPitch = width * 4;
		textureData.SlicePitch = textureData.RowPitch * height;

		// Schedule a copy from the upload heap to the Texture2D resource
		UpdateSubresources(mCmdList, mTargetTextureBuffer[textureIndex], mTextureUploadBuffer[textureIndex], 0, 0, 1, &textureData);

		transitionBarrier(mTargetTextureBuffer[textureIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		// Create SRV for target texture
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
			srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = 1;
			srvDesc.Texture2D.MostDetailedMip = 0;
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

			mDevice->CreateShaderResourceView(mTargetTextureBuffer[textureIndex], &srvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::TargetTexture) + textureIndex));
		}
	}

	void createComputePasses() {

		const std::wstring shaderName = L"MLPZen.hlsl";
		std::wstring shaderFolder = utils::getExePath() + L"shaders\\";
		std::wstring nnShaderFile = shaderFolder + shaderName;

		FILE* file = std::fopen(utils::wstringToString(nnShaderFile).c_str(), "r");
		if (!file)
		{
			nnShaderFile = utils::stringToWstring(SHADERS_DIR) + shaderName;
		}
		else
		{
			std::fclose(file);
		}

		std::vector<std::wstring> compilerFlags;

		if (mOptimizerType == OptimizerType::SGD) {
			compilerFlags.push_back(L"/D USE_SGD_OPTIMIZER");
		} else {
			compilerFlags.push_back(L"/D USE_ADAM_OPTIMIZER");
		}

		if (mInputEncodingType == InputEncodingType::Identity) {
			compilerFlags.push_back(L"/D USE_IDENTITY_ENCODING");
		} else if (mInputEncodingType == InputEncodingType::Frequency) {
			compilerFlags.push_back(L"/D USE_FREQUENCY_ENCODING");
		} else if (mInputEncodingType == InputEncodingType::OneBlob) {
			compilerFlags.push_back(L"/D USE_ONEBLOB_ENCODING");
		} else if (mInputEncodingType == InputEncodingType::HashGrid) {
			compilerFlags.push_back(L"/D USE_HASHGRID_ENCODING");
		}

		if (mExampleType == ExampleType::LearnCosine) {
			compilerFlags.push_back(L"/D LEARN_COSINE");
		} else if (mExampleType == ExampleType::LearnImage) {
			compilerFlags.push_back(L"/D LEARN_IMAGE");
		} else if (mExampleType == ExampleType::LearnFourImages) {
			compilerFlags.push_back(L"/D LEARN_FOUR_IMAGES");
		} else if (mExampleType == ExampleType::Bubbles3D) {
			compilerFlags.push_back(L"/D LEARN_BUBBLES");
		}

		if (mExampleType == ExampleType::Bubbles3D) {
			compilerFlags.push_back(L"/D INPUT_LENGTH=3");
		} else {
			compilerFlags.push_back(L"/D INPUT_LENGTH=2");
		}

		// Clamp encoding settings before using them in shader macros.
		getInputLayerNeuronCount(mInputDimensions);

		compilerFlags.push_back((std::wstring(L"/D NUM_FREQUENCIES=") + std::to_wstring(mFrequencies)).c_str());
		compilerFlags.push_back((std::wstring(L"/D NUM_ONEBLOB_BINS=") + std::to_wstring(mOneBlobBins)).c_str());
		compilerFlags.push_back((std::wstring(L"/D PI=") + std::to_wstring(glm::pi<float>())).c_str());

		const int oneBlobBinsLog2Neg = -(int)log2((float)mOneBlobBins);
		const float oneBlobZeta = pow(2, oneBlobBinsLog2Neg);
		compilerFlags.push_back((std::wstring(L"/D ONEBLOB_ZETA=") + std::to_wstring(oneBlobZeta)).c_str());

		if (mEnableGradientClipping)
		{
			compilerFlags.push_back(L"/D CLIP_GRADIENT=1");
		}

		if (mEnableHighPrecisionGradient)
		{
			compilerFlags.push_back(L"/D HIGH_PRECISION_GRADIENT=1");
		}

		if (mCalculateGlobalLoss)
		{
			compilerFlags.push_back(L"/D CALCULATE_GLOBAL_LOSS=1");
		}

		if (mUseFP16MLP) {
			compilerFlags.push_back(L"/D MLP_FP16_STORAGE=1");
		}

		if (mUseFP16Hashgrid) {
			compilerFlags.push_back(L"/D HG_FP16_STORAGE=1");
		}

		if (mOutputActivationFunctionType == ActivationFunctionType::LeakyRelu) {
			compilerFlags.push_back(L"/D OUTPUT_LEAKY_RELU=1");
		} else if (mOutputActivationFunctionType == ActivationFunctionType::Sigmoid) {
			compilerFlags.push_back(L"/D OUTPUT_SIGMOID=1");
		} else if (mOutputActivationFunctionType == ActivationFunctionType::None) {
			compilerFlags.push_back(L"/D OUTPUT_NONE=1");
		}

		compilerFlags.push_back(mExampleType == ExampleType::LearnCosine
			? L"/D OUTPUT_MASK=float4(1.0f,0.0f,0.0f,0.0f)"
			: L"/D OUTPUT_MASK=float4(1.0f,1.0f,1.0f,0.0f)");
		
		// Encode hashgrid setup
		{
			buildHashgridParameters(mInputDimensions);
		}

		// Encode network architecture in compiler flags macros
		{		
			compilerFlags.push_back((std::wstring(L"/D LAYER_COUNT=") + std::to_wstring(mLayerCount)).c_str());

			// Figure out number of neurons per layer
			int neuronsPerLayer[MAX_LAYERS];
			getNeuronsPerLayer(mInputDimensions, neuronsPerLayer);

			mMaxNeuronsPerLayer = 0;
			for (int i = 0; i < mLayerCount; i++)
			{
				mMaxNeuronsPerLayer = glm::max(mMaxNeuronsPerLayer, neuronsPerLayer[i]);
			}
			compilerFlags.push_back((std::wstring(L"/D MAX_NEURONS_PER_LAYER=") + std::to_wstring(mMaxNeuronsPerLayer)).c_str());
			compilerFlags.push_back((std::wstring(L"/D MAX_NEURON_QUARTETS_PER_LAYER=") + std::to_wstring(mMaxNeuronsPerLayer / 4)).c_str());
			compilerFlags.push_back((std::wstring(L"/D ACTIVATION_QUARTETS_PER_NETWORK=") + std::to_wstring(mLayerCount * mMaxNeuronsPerLayer / 4)).c_str());
			compilerFlags.push_back((std::wstring(L"/D MAX_PARAM_QUARTETS_PER_LAYER=") + std::to_wstring((mMaxNeuronsPerLayer * (mMaxNeuronsPerLayer + 1)) / 4)).c_str());

			// Encode layer types
			{
				for (int i = 0; i < MAX_LAYERS; i++)
				{
					int layerType = INPUT_LAYER;

					if (i > 0 && i < (mLayerCount - 1)) {
						layerType = HIDDEN_LAYER;
					}
					if (i >= (mLayerCount - 1)) {
						layerType = OUTPUT_LAYER;
					}
					compilerFlags.push_back((std::wstring(L"/D LAYER_TYPE_") + std::to_wstring(i) + L"=" + std::to_wstring(layerType)).c_str());
				}
			}

			// Encode number of neurons per layer
			{
				for (int i = 0; i < mLayerCount; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D NEURONS_PER_LAYER_") + std::to_wstring(i) + L"=" + std::to_wstring(neuronsPerLayer[i])).c_str());
					compilerFlags.push_back((std::wstring(L"/D NEURON_QUARTETS_PER_LAYER_") + std::to_wstring(i) + L"=" + std::to_wstring(neuronsPerLayer[i] / 4)).c_str());
				}
				for (int i = mLayerCount; i < MAX_LAYERS; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D NEURONS_PER_LAYER_") + std::to_wstring(i) + L"=0").c_str());
					compilerFlags.push_back((std::wstring(L"/D NEURON_QUARTETS_PER_LAYER_") + std::to_wstring(i) + L"=0").c_str());
				}
			}
		}

		// Encode hash-grid setup in compiler flags macros
		{
			int paramsQuartetsTotal = ((mLayerCount - 1) * (mMaxNeuronsPerLayer * (mMaxNeuronsPerLayer + 1))) / 4;

			// We emulate hashgrid with feature vector length 8 by adding more levels with identical setting. Get an actual number of levels used in the shader here.
			int actualLevels = mHashgridLevels * (getHgFeatureVectorLength() / 4);
			
			compilerFlags.push_back((std::wstring(L"/D HG_LEVELS=") + std::to_wstring(actualLevels)).c_str());
			compilerFlags.push_back((std::wstring(L"/D HG_MAP_SIZE=") + std::to_wstring(mHashgridMapSize)).c_str());
			compilerFlags.push_back((std::wstring(L"/D HG_OFFSET=") + std::to_wstring(paramsQuartetsTotal)).c_str());
			compilerFlags.push_back((std::wstring(L"/D HG_TOTAL_QUARTETS=") + std::to_wstring(mHashgridTotalParameters / 4)).c_str());
			
			for (int level = 0; level < (HG_MAX_LEVELS * 2); level++)
			{
				compilerFlags.push_back((std::wstring(L"/D HG_LEVEL_RES_") + std::to_wstring(level) + L"=" + std::to_wstring(mHashgridLevelResolution[level])).c_str());
				compilerFlags.push_back((std::wstring(L"/D HG_LEVEL_OFFSET_") + std::to_wstring(level) + L"=" + std::to_wstring(mHashgridLevelOffset[level])).c_str());
				compilerFlags.push_back((std::wstring(L"/D HG_LEVEL_HASHED_") + std::to_wstring(level) + L"=" + (mHashgridLevelHashed[level] ? L"true" : L"false")));
			}
		}

		// Process compiler flags to an array of string pointers
		std::vector<LPCWSTR> flagsPointers;
		flagsPointers.reserve(compilerFlags.size());
		for (const std::wstring& flag : compilerFlags) {
			flagsPointers.push_back(flag.c_str());
		}

		// Create inference pass
		{
			IDxcBlob* shaderBlob = mShaderCompiler.CompileShader(nnShaderFile.c_str(), L"Inference", L"cs_6_2", flagsPointers);

			SAFE_RELEASE(mInferencePSO);
			mInferencePSO = createComputePSO(*shaderBlob, mGlobalRootSignature);
		}		

		// Create backpropagation pass
		{
			IDxcBlob* shaderBlob = mShaderCompiler.CompileShader(nnShaderFile.c_str(), L"Backpropagation", L"cs_6_2", flagsPointers);

			SAFE_RELEASE(mBackpropagationPSO);
			mBackpropagationPSO = createComputePSO(*shaderBlob, mGlobalRootSignature);
		}

		// Create optimization pass
		{
			IDxcBlob* shaderBlob = mShaderCompiler.CompileShader(nnShaderFile.c_str(), L"Optimization", L"cs_6_2", flagsPointers);

			SAFE_RELEASE(mOptimizationPSO);
			mOptimizationPSO = createComputePSO(*shaderBlob, mGlobalRootSignature);
		}

		// Create initialization pass
		{
			IDxcBlob* shaderBlob = mShaderCompiler.CompileShader(nnShaderFile.c_str(), L"Initialize", L"cs_6_2", flagsPointers);

			SAFE_RELEASE(mInitializationPSO);
			mInitializationPSO = createComputePSO(*shaderBlob, mGlobalRootSignature);
		}

		// Create reference output pass
		{
			IDxcBlob* shaderBlob = mShaderCompiler.CompileShader(nnShaderFile.c_str(), L"OutputReference", L"cs_6_2", flagsPointers);

			SAFE_RELEASE(mReferenceOutputPSO);
			mReferenceOutputPSO = createComputePSO(*shaderBlob, mGlobalRootSignature);
		}
	}

	void transitionBarrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier.Transition.pResource = resource;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = from;
		barrier.Transition.StateAfter = to;

		mCmdList->ResourceBarrier(1, &barrier);
	}

	void transitionBarrier(ID3D12Resource** resources, int resourceCount, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
		assert(resourceCount <= MAX_MLPS);
		D3D12_RESOURCE_BARRIER barriers[MAX_MLPS];

		for (int i = 0; i < resourceCount; i++)
		{
			barriers[i] = {};
			barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[i].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			barriers[i].Transition.pResource = resources[i];
			barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			barriers[i].Transition.StateBefore = from;
			barriers[i].Transition.StateAfter = to;
		}

		mCmdList->ResourceBarrier(resourceCount, barriers);
	}

	void uavBarrier(ID3D12Resource* resource) {
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier.UAV.pResource = resource;

		mCmdList->ResourceBarrier(1, &barrier);
	}

	void uavBarrier(ID3D12Resource** resources, int resourceCount) {
		assert(resourceCount <= 8);
		D3D12_RESOURCE_BARRIER barriers[8];

		for (int i = 0; i < resourceCount; i++)
		{
			barriers[i] = {};
			barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			barriers[i].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			barriers[i].UAV.pResource = resources[i];
		}

		mCmdList->ResourceBarrier(resourceCount, barriers);
	}

	void dispatchCompute2D(ID3D12PipelineState* pso, uint32_t dispatchWidth, uint32_t dispatchHeight)
	{
		if (dispatchWidth > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION || dispatchHeight > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
		{
			utils::validate(E_FAIL, L"Error: dispatch size too large!");
		}

		mCmdList->SetPipelineState(pso);
		mCmdList->Dispatch(dispatchWidth, dispatchHeight, 1);
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getDescriptorHandle(UINT index) {
		D3D12_CPU_DESCRIPTOR_HANDLE handle = mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
		handle.ptr += (mCbvSrvUavDescSize * index);
		return handle;
	}

	/**
	* This enum specifies a layout of resources in NN shaders
	*/
	enum class DescriptorHeapConstants {

		// List of resources declared in the shader, as they appear in the descriptors heap
		NNDataCB = 0,
		Output,
		Reference,
		LossData,
		NNParametersOutputBuffer,
		NNAdamDataBuffer = NNParametersOutputBuffer + MAX_MLPS,
		NNGradientBuffer = NNAdamDataBuffer + MAX_MLPS,
		NNParametersBackpropOutputBuffer = NNGradientBuffer + MAX_MLPS,
		TargetTexture = NNParametersBackpropOutputBuffer + MAX_MLPS,
		NNParametersInputBuffer = TargetTexture + MAX_TEXTURES,
		NNParametersBackpropInputBuffer = NNParametersInputBuffer + MAX_MLPS,
		NNHashgridInputBuffer = NNParametersBackpropInputBuffer + MAX_MLPS,
		NNHashgridOutputBuffer = NNHashgridInputBuffer + MAX_MLPS,
		NNMemoryMap = NNHashgridOutputBuffer + MAX_MLPS,
		Total = NNMemoryMap + 1,

		// Constant buffer range
		CBStart = NNDataCB,
		CBEnd = NNDataCB,
		CBTotal = CBEnd - CBStart + 1,

		// UAV space 0 range
		UAV0Start = Output,
		UAV0End = LossData,
		UAV0Total = UAV0End - UAV0Start + 1,

		// UAV space 1 range
		UAV1Start = NNParametersOutputBuffer,
		UAV1End = NNParametersOutputBuffer + MAX_MLPS - 1,
		UAV1Total = MAX_MLPS,

		// UAV space 2 range
		UAV2Start = NNAdamDataBuffer,
		UAV2End = NNAdamDataBuffer + MAX_MLPS - 1,
		UAV2Total = MAX_MLPS,

		// UAV space 3 range
		UAV3Start = NNGradientBuffer,
		UAV3End = NNGradientBuffer + MAX_MLPS - 1,
		UAV3Total = MAX_MLPS,

		// UAV space 4 range
		UAV4Start = NNParametersBackpropOutputBuffer,
		UAV4End = NNParametersBackpropOutputBuffer + MAX_MLPS - 1,
		UAV4Total = MAX_MLPS,

		// UAV space 5 range
		UAV5Start = NNHashgridOutputBuffer,
		UAV5End = NNHashgridOutputBuffer + MAX_MLPS - 1,
		UAV5Total = MAX_MLPS,

		// SRV space 0 range
		SRV0Start = NNParametersInputBuffer,
		SRV0End = NNParametersInputBuffer + MAX_MLPS - 1,
		SRV0Total = MAX_MLPS,

		// SRV space 1 range
		SRV1Start = TargetTexture,
		SRV1End = TargetTexture + MAX_TEXTURES - 1,
		SRV1Total = SRV1End - SRV1Start + 1,

		// SRV space 2 range
		SRV2Start = NNParametersBackpropInputBuffer,
		SRV2End = NNParametersBackpropInputBuffer + MAX_MLPS - 1,
		SRV2Total = MAX_MLPS,

		// SRV space 3 range
		SRV3Start = NNHashgridInputBuffer,
		SRV3End = NNHashgridInputBuffer + MAX_MLPS - 1,
		SRV3Total = MAX_MLPS,

		// SRV space 4 range
		SRV4Start = NNMemoryMap,
		SRV4End = NNMemoryMap,
		SRV4Total = SRV4End - SRV4Start + 1,

	};

	enum class RootParameterIndex {
		CbvSrvUavs,
		RootConstants,
		Count
	};

	ID3D12RootSignature* createGlobalRootSignature() {

		// CBVs
		D3D12_DESCRIPTOR_RANGE cbvRange;
		cbvRange.BaseShaderRegister = 0;
		cbvRange.NumDescriptors = UINT(DescriptorHeapConstants::CBTotal);
		cbvRange.RegisterSpace = 0;
		cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
		cbvRange.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::CBStart);

		// UAVs
		D3D12_DESCRIPTOR_RANGE uavRange0;
		uavRange0.BaseShaderRegister = 0;
		uavRange0.NumDescriptors = UINT(DescriptorHeapConstants::UAV0Total);
		uavRange0.RegisterSpace = 0;
		uavRange0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange0.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV0Start);

		D3D12_DESCRIPTOR_RANGE uavRange1;
		uavRange1.BaseShaderRegister = 0;
		uavRange1.NumDescriptors = UINT(DescriptorHeapConstants::UAV1Total);
		uavRange1.RegisterSpace = 1;
		uavRange1.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange1.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV1Start);

		D3D12_DESCRIPTOR_RANGE uavRange2;
		uavRange2.BaseShaderRegister = 0;
		uavRange2.NumDescriptors = UINT(DescriptorHeapConstants::UAV2Total);
		uavRange2.RegisterSpace = 2;
		uavRange2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange2.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV2Start);

		D3D12_DESCRIPTOR_RANGE uavRange3;
		uavRange3.BaseShaderRegister = 0;
		uavRange3.NumDescriptors = UINT(DescriptorHeapConstants::UAV3Total);
		uavRange3.RegisterSpace = 3;
		uavRange3.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange3.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV3Start);

		D3D12_DESCRIPTOR_RANGE uavRange4;
		uavRange4.BaseShaderRegister = 0;
		uavRange4.NumDescriptors = UINT(DescriptorHeapConstants::UAV4Total);
		uavRange4.RegisterSpace = 4;
		uavRange4.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange4.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV4Start);

		D3D12_DESCRIPTOR_RANGE uavRange5;
		uavRange5.BaseShaderRegister = 0;
		uavRange5.NumDescriptors = UINT(DescriptorHeapConstants::UAV5Total);
		uavRange5.RegisterSpace = 5;
		uavRange5.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange5.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV5Start);

		// SRVs
		D3D12_DESCRIPTOR_RANGE srvRange0;
		srvRange0.BaseShaderRegister = 0;
		srvRange0.NumDescriptors = UINT(DescriptorHeapConstants::SRV0Total);
		srvRange0.RegisterSpace = 0;
		srvRange0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange0.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV0Start);

		D3D12_DESCRIPTOR_RANGE srvRange1;
		srvRange1.BaseShaderRegister = 0;
		srvRange1.NumDescriptors = UINT(DescriptorHeapConstants::SRV1Total);
		srvRange1.RegisterSpace = 1;
		srvRange1.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange1.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV1Start);

		D3D12_DESCRIPTOR_RANGE srvRange2;
		srvRange2.BaseShaderRegister = 0;
		srvRange2.NumDescriptors = UINT(DescriptorHeapConstants::SRV2Total);
		srvRange2.RegisterSpace = 2;
		srvRange2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange2.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV2Start);

		D3D12_DESCRIPTOR_RANGE srvRange3;
		srvRange3.BaseShaderRegister = 0;
		srvRange3.NumDescriptors = UINT(DescriptorHeapConstants::SRV3Total);
		srvRange3.RegisterSpace = 3;
		srvRange3.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange3.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV3Start);

		D3D12_DESCRIPTOR_RANGE srvRange4;
		srvRange4.BaseShaderRegister = 0;
		srvRange4.NumDescriptors = UINT(DescriptorHeapConstants::SRV4Total);
		srvRange4.RegisterSpace = 4;
		srvRange4.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange4.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV4Start);

		D3D12_DESCRIPTOR_RANGE cbvUavSrvRanges[] = {
			cbvRange,
			uavRange0,
			uavRange1,
			uavRange2,
			uavRange3,
			uavRange4,
			uavRange5,
			srvRange0,
			srvRange1,
			srvRange2,
			srvRange3,
			srvRange4
		};

		// Root parameter - CBV/UAV/SRV
		D3D12_ROOT_PARAMETER paramCbvUavSrv = {};
		paramCbvUavSrv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		paramCbvUavSrv.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		paramCbvUavSrv.DescriptorTable.NumDescriptorRanges = _countof(cbvUavSrvRanges);
		paramCbvUavSrv.DescriptorTable.pDescriptorRanges = cbvUavSrvRanges;

		// Root parameter - Root constants
		D3D12_ROOT_PARAMETER paramRootConstants = {};
		paramRootConstants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		paramRootConstants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		paramRootConstants.Constants.Num32BitValues = 12;
		paramRootConstants.Constants.ShaderRegister = 1;
		paramRootConstants.Constants.RegisterSpace = 0;

		D3D12_ROOT_PARAMETER rootParams[UINT(RootParameterIndex::Count)];
		rootParams[UINT(RootParameterIndex::CbvSrvUavs)] = paramCbvUavSrv;
		rootParams[UINT(RootParameterIndex::RootConstants)] = paramRootConstants;

		D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
		rootDesc.NumParameters = _countof(rootParams);
		rootDesc.pParameters = rootParams;
		rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

		return createRootSignature(rootDesc);
	}

	ID3D12RootSignature* createRootSignature(const D3D12_ROOT_SIGNATURE_DESC& desc) {
		HRESULT hr;
		ID3DBlob* sig;
		ID3DBlob* error;

		hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &error);
		utils::validate(hr, L"Error: failed to serialize root signature!");

		ID3D12RootSignature* pRootSig;
		hr = mDevice->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&pRootSig));
		utils::validate(hr, L"Error: failed to create root signature!");

		SAFE_RELEASE(sig);
		SAFE_RELEASE(error);
		return pRootSig;
	}

	void uploadConstantBuffer() {

		transitionBarrier(mNNDataCB, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_RESOURCE_STATE_COPY_DEST);

		BYTE constantBufferData[ALIGN(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, sizeof(NNData))] = {};
		memcpy(constantBufferData, &mNNData, sizeof(mNNData));

		D3D12_SUBRESOURCE_DATA bufferDataDesc = {};
		bufferDataDesc.pData = constantBufferData;
		bufferDataDesc.RowPitch = mNNDataCBSize;
		bufferDataDesc.SlicePitch = bufferDataDesc.RowPitch;

		UINT64 uploadedBytes = UpdateSubresources(mCmdList, mNNDataCB, mNNDataCBUpload, 0, 0, 1, &bufferDataDesc);
		HRESULT hr = (mNNDataCBSize == uploadedBytes ? S_OK : E_FAIL);
		utils::validate(hr, L"Error: failed to update constant buffer!");

		transitionBarrier(mNNDataCB, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
	}

	void createBuffer(ID3D12Device* device, D3D12_HEAP_TYPE heapType, UINT64 alignment, UINT64 size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ID3D12Resource** ppResource)
	{
		HRESULT hr;

		D3D12_HEAP_PROPERTIES heapDesc = {};
		heapDesc.Type = heapType;
		heapDesc.CreationNodeMask = 1;
		heapDesc.VisibleNodeMask = 1;

		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resourceDesc.Alignment = alignment;
		resourceDesc.Height = 1;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.SampleDesc.Quality = 0;
		resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		resourceDesc.Width = size;
		resourceDesc.Flags = flags;

		// Create the GPU resource
		hr = device->CreateCommittedResource(&heapDesc, D3D12_HEAP_FLAG_NONE, &resourceDesc, state, nullptr, IID_PPV_ARGS(ppResource));
		utils::validate(hr, L"Error: failed to create buffer resource!");
	}

	void createTexture(ID3D12Device* device, UINT64 width, UINT64 height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ID3D12Resource** ppResource)
	{
		D3D12_HEAP_PROPERTIES heapDesc = {};
		heapDesc.Type = D3D12_HEAP_TYPE_DEFAULT;
		heapDesc.CreationNodeMask = 1;
		heapDesc.VisibleNodeMask = 1;

		D3D12_RESOURCE_DESC desc = {};
		desc.DepthOrArraySize = 1;
		desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		desc.Format = format;
		desc.Flags = flags;
		desc.Width = width;
		desc.Height = height;
		desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		desc.MipLevels = 1;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;

		// Create the GPU resource
		HRESULT hr = mDevice->CreateCommittedResource(&heapDesc, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(ppResource));
		utils::validate(hr, L"Error: failed to create texture!");
	}

	ID3D12PipelineState* createComputePSO(IDxcBlob& shaderBlob, ID3D12RootSignature* rootSignature)
	{
		D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc = {};
		pipelineDesc.pRootSignature = rootSignature;
		pipelineDesc.CS.pShaderBytecode = shaderBlob.GetBufferPointer();
		pipelineDesc.CS.BytecodeLength = shaderBlob.GetBufferSize();
		pipelineDesc.NodeMask = 0;
		pipelineDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

		ID3D12PipelineState* pso = nullptr;
		HRESULT hr = mDevice->CreateComputePipelineState(&pipelineDesc, IID_PPV_ARGS(&pso));
		utils::validate(hr, L"Error: failed to create compute PSO!");

		return pso;
	}

	void moveToNextFrame()
	{
		// Schedule a Signal command in the queue
		const UINT64 currentFenceValue = mFenceValues[mCurrentFrameIndex];
		HRESULT hr = mCmdQueue->Signal(mFence, currentFenceValue);
		utils::validate(hr, L"Error: failed to signal command queue!");

		// Update the frame index
		mCurrentFrameIndex = mSwapChain->GetCurrentBackBufferIndex();

		// If the next frame is not ready to be rendered yet, wait until it is
		if (mFence->GetCompletedValue() < mFenceValues[mCurrentFrameIndex])
		{
			hr = mFence->SetEventOnCompletion(mFenceValues[mCurrentFrameIndex], mFenceEvent);
			utils::validate(hr, L"Error: failed to set fence value!");

			WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);
		}

		// Set the fence value for the next frame
		mFenceValues[mCurrentFrameIndex] = currentFenceValue + 1;
	}

	void submitCmdList()
	{
		mCmdList->Close();

		ID3D12CommandList* pGraphicsList = { mCmdList };
		mCmdQueue->ExecuteCommandLists(1, &pGraphicsList);
		mFenceValues[mCurrentFrameIndex]++;
		mCmdQueue->Signal(mFence, mFenceValues[mCurrentFrameIndex]);
	}

	void present()
	{
		// When using sync interval 0, it is recommended to always pass the tearing
		// flag when it is supported, even when presenting in windowed mode.
		// However, this flag cannot be used if the app is in fullscreen mode as a
		// result of calling SetFullscreenState.
		UINT presentFlags = (!mEnableVSync && mIsTearingSupport) ? DXGI_PRESENT_ALLOW_TEARING : 0;

		HRESULT hr = mSwapChain->Present(mEnableVSync ? 1 : 0, presentFlags);
		if (FAILED(hr))
		{
			hr = mDevice->GetDeviceRemovedReason();
			utils::validate(hr, L"Error: failed to present!");
		}
	}

	void waitForGPU()
	{
		// Schedule a signal command in the queue
		HRESULT hr = mCmdQueue->Signal(mFence, mFenceValues[mCurrentFrameIndex]);
		utils::validate(hr, L"Error: failed to signal fence!");

		// Wait until the fence has been processed
		hr = mFence->SetEventOnCompletion(mFenceValues[mCurrentFrameIndex], mFenceEvent);
		utils::validate(hr, L"Error: failed to set fence event!");

		WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);

		// Increment the fence value for the current frame
		mFenceValues[mCurrentFrameIndex]++;
	}

	void resetCommandList()
	{
		// Reset the command allocator for the current frame
		HRESULT hr = mCmdAlloc[mCurrentFrameIndex]->Reset();
		utils::validate(hr, L"Error: failed to reset command allocator!");

		// Reset the command list for the current frame
		hr = mCmdList->Reset(mCmdAlloc[mCurrentFrameIndex], nullptr);
		utils::validate(hr, L"Error: failed to reset command list!");
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getBackBufferView(UINT bufferIndex) {

		D3D12_CPU_DESCRIPTOR_HANDLE renderTargetViewHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
		renderTargetViewHandle.ptr += (mRtvDescSize * bufferIndex);

		return renderTargetViewHandle;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getCurrentBackBufferView() {
		return getBackBufferView(mCurrentFrameIndex);
	}

	void initImGui(HWND hwnd) {

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		// Scale default ImGUI font according to DPI scaling
		ImFontConfig fontConfig = {};
		fontConfig.SizePixels = 13.0f * mDpiScale; //< ImGui uses 13px font by default
		mImguiFont = io.Fonts->AddFontDefault(&fontConfig);

		ImGui::StyleColorsDark();
		{
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.NumDescriptors = 1;
			desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			if (mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&imguiSrvDescHeap)) != S_OK)
				return;
		}

		// Setup Platform/Renderer bindings
		ImGui_ImplWin32_Init((void*)hwnd);
		ImGui_ImplDX12_Init(mDevice, kMaxFramesInFlight,
			DXGI_FORMAT_R8G8B8A8_UNORM,
			imguiSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
			imguiSrvDescHeap->GetGPUDescriptorHandleForHeapStart());
	}

	int getHgFeatureVectorLength()
	{
		if (mHashgridFeatureVectorLength == HGFeatureVectorLengthType::Four) {
			return 4;
		} else if (mHashgridFeatureVectorLength == HGFeatureVectorLengthType::Eight) {
			return 8;		
		}

		return 4;
	}

	int getInputLayerNeuronCount(const int nInputs)
	{
		mFrequenciesLimit = mMaxNeuronsPerLayerLimit / (nInputs * 2);
		mOneBlobBinsLimit = (mMaxNeuronsPerLayerLimit / nInputs) / 4 * 4;
		mFrequencies = glm::min(mFrequencies, mFrequenciesLimit);
		mOneBlobBins = glm::min(mOneBlobBins, mOneBlobBinsLimit);

		int inputNeurons = 0;
		if (mInputEncodingType == InputEncodingType::Identity) {
			inputNeurons = nInputs;
		} else if (mInputEncodingType == InputEncodingType::Frequency) {
			inputNeurons = nInputs * mFrequencies * 2;
		} else if (mInputEncodingType == InputEncodingType::OneBlob) {
			inputNeurons = nInputs * mOneBlobBins;
		} else if (mInputEncodingType == InputEncodingType::HashGrid) {
			inputNeurons = mHashgridLevels * getHgFeatureVectorLength();
		}

		inputNeurons = ALIGN(4, inputNeurons);

		if (inputNeurons > mMaxNeuronsPerLayerLimit)
		{
			utils::validate(E_FAIL, L"Too many input neurons, check your hashgrid config!");
		}

		return inputNeurons;
	}

	int getOutputLayerNeuronCount() 
	{

		int outputNeurons = 1;

		if (mExampleType == ExampleType::LearnCosine)
		{
			outputNeurons = 1;
		}
		else
		{
			outputNeurons = 3;
		}

		// Output neuron count must be divisible by 4
		return ALIGN(4, outputNeurons);
	}

	void getNeuronsPerLayer(int inputDimensions, int* neuronsPerLayer)
	{
		neuronsPerLayer[0] = getInputLayerNeuronCount(inputDimensions);
		neuronsPerLayer[mLayerCount - 1] = getOutputLayerNeuronCount();
		for (int i = 1; i < mLayerCount - 1; i++)
		{
			neuronsPerLayer[i] = mNeuronsPerLayer;
		}
	}

	// Note: calling this advances the state of cosine annealing! Make sure to only call once per training step
	float getLearningRate()
	{
		if (mLearningRateSchedule == LearningRateScheduleType::Fixed)
		{
			return mGlobalLearningRate;
		}
		else if (mLearningRateSchedule == LearningRateScheduleType::Linear)
		{
			float alpha = glm::min(1.0f, glm::max(0.0f, float(mTrainingSteps) / float(mDecaySteps)));
			return mGlobalLearningRate + (mMaxLearningRate - mGlobalLearningRate) * (1.0f - alpha);
		}
		else if (mLearningRateSchedule == LearningRateScheduleType::CosineAnnealing)
		{
			// Source: SGDR: Stochastic Gradient Descent with Warm Restarts
			const int tMult = 2;
			const size_t Ti = mBaseTi * (mRestarts + 1) * tMult;
			const float t = float(mTCur) / float(Ti);
			mTCur++;
			if (mTCur == (Ti)) {
				mRestarts++;
				mTCur = 0;
			}

			const float lrMin = mLRMin;
			const float lrMax = glm::max(lrMin, mMaxLearningRate * glm::pow(mLRMaxScaleFactor, float(mRestarts)));
			return lrMin + 0.5f * (lrMax - lrMin) * (1.0f + glm::cos(t * glm::pi<float>()));
		}

		return mGlobalLearningRate;
	}

	void buildHashgridParameters(int inputDimensions)
	{
		assert(inputDimensions == 2 || inputDimensions == 3);

		mHashgridTotalParameters = 0;

		size_t levelResolution = mHashgridBaseResolution;

		// We simulate feature vector length 8 by adding two actual levels with identical settings
		int actualLevelsPerStep = getHgFeatureVectorLength() / 4;
		int actualLevelIndex = 0;

		for (int level = 0; level < mHashgridLevels; level++)
		{
			for (int i = 0; i < actualLevelsPerStep; i++)
			{
				// Figure out number of feature vectors for this level of a grid
				size_t levelFeatureVectors = glm::pow(levelResolution, inputDimensions);

				const bool isLevelHashed = (levelFeatureVectors > mHashgridMapSize);

				// Limit number of feature vectors to hashtable size
				if (isLevelHashed)
				{
					levelFeatureVectors = mHashgridMapSize;
				}

				const size_t featureVectorsLengthPerLevel = 4;
				const size_t levelParameters = levelFeatureVectors * featureVectorsLengthPerLevel;

				mHashgridLevelOffset[actualLevelIndex] = mHashgridTotalParameters / 4;
				mHashgridLevelHashed[actualLevelIndex] = isLevelHashed;
				mHashgridLevelResolution[actualLevelIndex] = levelResolution;

				mHashgridTotalParameters += levelParameters;
				actualLevelIndex++;
			}

			levelResolution *= 2;
		}

		mHashgridActualLevels = actualLevelIndex;
		updateHashgridGradientScaler();

    }

    void updateHashgridGradientScaler()
    {
        if (mHashgridActualLevels == 0)
            return;

        mHashgridGradientScaler = 0.0f;
        const size_t totalFeatureVectors = mHashgridTotalParameters / 4;

        for (int level = 0; level < mHashgridActualLevels; level++)
        {
            const size_t nextLevelOffset = (level + 1 < mHashgridActualLevels)
                ? mHashgridLevelOffset[level + 1]
                : totalFeatureVectors;
            const size_t levelFeatureVectors = nextLevelOffset - mHashgridLevelOffset[level];
            mHashgridGradientScaler += glm::max(1.0f, float(mBatchSize) / float(levelFeatureVectors));
        }

        mHashgridGradientScaler = float(mHashgridActualLevels) / mHashgridGradientScaler;
    }

	// Dx12 Boilerplate things
	ID3D12DescriptorHeap* imguiSrvDescHeap = nullptr;
	std::wstring mAdapterName;

	ID3D12Device5* mDevice = nullptr;
	IDXGIAdapter1* mAdapter = nullptr;
	
	IDXGISwapChain3* mSwapChain = nullptr;
	ID3D12Resource* mOutputBuffer = nullptr;
	ID3D12Resource* mReferenceBuffer = nullptr;

	DXGI_FORMAT mDepthBufferFormat = DXGI_FORMAT::DXGI_FORMAT_D32_FLOAT;
	DXGI_FORMAT mBackBufferFormat = DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
	ID3D12DescriptorHeap* mDsvHeap = nullptr;

	static const unsigned int kMaxFramesInFlight = 2;
	ID3D12Fence* mFence = nullptr;
	UINT64 mFenceValues[kMaxFramesInFlight];
	HANDLE mFenceEvent;
	ID3D12Resource* mBackBuffer[kMaxFramesInFlight];

	bool mIsTearingSupport = false;
	bool mEnableVSync = false;

	IDXGIFactory6* mDxgiFactory = nullptr;
	float mDpiScale = 1.0f;
	ImFont* mImguiFont;
	unsigned int mCurrentFrameIndex = 0;
	ID3D12Resource* mDepthStencilBuffer = nullptr;

	ID3D12DescriptorHeap* mRtvHeap = nullptr;
	UINT mRtvDescSize = 0;

	ID3D12GraphicsCommandList4* mCmdList = nullptr;

	ID3D12CommandQueue* mCmdQueue = nullptr;
	ID3D12CommandAllocator* mCmdAlloc[kMaxFramesInFlight];

	// PSOs
	ID3D12PipelineState* mInferencePSO = nullptr;
	ID3D12PipelineState* mInitializationPSO = nullptr;
	ID3D12PipelineState* mBackpropagationPSO = nullptr;
	ID3D12PipelineState* mOptimizationPSO = nullptr;
	ID3D12PipelineState* mReferenceOutputPSO = nullptr;

	DxcShaderCompiler	mShaderCompiler;

	ID3D12DescriptorHeap* mDescriptorHeap = nullptr;
	ID3D12RootSignature* mGlobalRootSignature = nullptr;
	UINT mCbvSrvUavDescSize = 0;
	ID3D12Debug* mDebugController = nullptr;

	// Loss data
	ID3D12Resource* mLossDataBuffer = nullptr;
	ID3D12Resource* mLossDataReadbackBuffer = nullptr;
	size_t			mLossBufferSize = 2 * MAX_MLPS;

	ID3D12Resource* mNNDataCB = nullptr;
	ID3D12Resource* mNNDataCBUpload = nullptr;
	NNData mNNData;
	UINT mNNDataCBSize = 0;

	ID3D12Resource* mNNParametersBuffer[MAX_MLPS] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	ID3D12Resource* mNNAdamDataBuffer[MAX_MLPS] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	ID3D12Resource* mNNGradientBuffer[MAX_MLPS] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	ID3D12Resource* mNNHashgridBuffer[MAX_MLPS] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	ID3D12Resource* mNNParametersBackpropBuffer[MAX_MLPS] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	ID3D12Resource* mNNMemoryMapBuffer = nullptr;
	ID3D12Resource* mNNMemoryMapUploadBuffer = nullptr;
	size_t			mNNParametersQuartetCount = 0;

    ID3D12Resource* mTargetTextureBuffer[MAX_TEXTURES] = { 0, 0, 0, 0 };
    ID3D12Resource* mTextureUploadBuffer[MAX_TEXTURES] = { 0, 0, 0, 0 };

	// For this sample, all textures are 512x512
	unsigned int mTargetWidth = 0;
	unsigned int mTargetHeight = 0;

	bool mReloadShaders = false;
	bool mNNNeedsInitialization = true;
	bool mNNArchitectureDirty = false;
	bool mTargetImageLoaded = false;

	// NN Settings
	uint32_t mTrainingSteps = 0;
	bool mLimitTrainingSteps = false;
	int mMaxTrainingSteps = 10000;
	bool mCalculateGlobalLoss = false;
	bool mUseFP16MLP = false;
	bool mEnableTraining = true;
	bool mTrainingStep = true;
	bool mEnableGradientClipping = false;
	bool mEnableHighPrecisionGradient = false;
	float mGlobalLearningRate = 0.001f;
	int mBatchSize = 32 * 1024;
	int mLayerCount = 4;
	int mNeuronsPerLayer = 32;
	int mFrequencies = 8;
	int mFrequenciesLimit = 16;
	int mOneBlobBins = 16;
	int mOneBlobBinsLimit = 32;
	int mInputDimensions = 2;
	int mTrainingStepsPerFrame = 1;
	bool mFreezeMLP = false;
	bool mFreezeHashgrid = false;
	float mExactMlpSizeBytes = 0.0f;

	// Hashgrid parameters
	size_t mHashgridLevelOffset[HG_MAX_LEVELS * 2] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	bool mHashgridLevelHashed[HG_MAX_LEVELS * 2] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	size_t mHashgridLevelResolution[HG_MAX_LEVELS * 2] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	size_t mHashgridTotalParameters = 0;
	size_t mHashgridTotalAllocatedParameters = 0;
	bool mUseFP16Hashgrid = true;
	bool mHashgridTotalAllocatedFP16 = mUseFP16Hashgrid;
	int mHashgridLevels = 4;
	int mHashgridActualLevels = 0;
	int mHashgridBaseResolution = 16;
	int mHashgridMapSize = 4096;
	float mHashgridGradientScaler = 1.0f;
	bool mHashgridSettingsChanged = false;

	enum class HGFeatureVectorLengthType : uint {
		Four,
		Eight
	};

	HGFeatureVectorLengthType mHashgridFeatureVectorLength = HGFeatureVectorLengthType::Four;

	int mTextureToLearn = 2;
	float mBubblesZPlane = 0.5f;

	int mNewLayerCount = 4;
	int mNewNeuronsPerLayer = 32;
	int mMaxNeuronsPerLayer = 0;
	int mMaxNeuronsPerLayerLimit = 64;

	Profiler mProfiler;

	enum class InputEncodingType : uint {
		Identity,
		Frequency,
		OneBlob,
		HashGrid
	};

	InputEncodingType mInputEncodingType = InputEncodingType::HashGrid;

	enum class OptimizerType : uint {
		SGD,
		Adam
	};

	OptimizerType mOptimizerType = OptimizerType::Adam;

	enum class ActivationFunctionType : uint {
		LeakyRelu,
		Sigmoid,
		None
	};

	ActivationFunctionType mOutputActivationFunctionType = ActivationFunctionType::Sigmoid;

	enum class InitializationType : uint {
		HeGaussian,
		HeUniform,
		XavierGaussian,
		XavierUniform,
		LeCunGaussian,
		LeCunUniform
	};

	InitializationType mInitializationType = InitializationType::HeGaussian;

	enum class LearningRateScheduleType : uint {
		Fixed,
		Linear,
		CosineAnnealing
	};

	// Learning rate schedule settings
	LearningRateScheduleType mLearningRateSchedule = LearningRateScheduleType::CosineAnnealing;
	float			mMaxLearningRate = 0.05f;
	int				mDecaySteps = 500;

	// Cosine annealing LR settings
	float			mLRMin = 0.000001f;
	float			mLRMaxScaleFactor = 0.8f;
	int				mBaseTi = 50;
	int				mRestarts = 0;
	int				mTCur = 0;

	enum class ExampleType : uint {
		LearnCosine,
		LearnImage,
		LearnFourImages,
		Bubbles3D
	};

	// Learning rate schedule settings
	ExampleType mExampleType = ExampleType::LearnImage;

	int mMlpCount = 1;
	int mAllocatedMlpCount = 0;
	int mAllocatedInputNeuronsCount = 0;
	int mAllocatedParamSize = 0;

};

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	PAINTSTRUCT ps;
	HDC hdc;
	RECT clientRect;
	LPCREATESTRUCT pCreateStruct;

	ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam);
	
	MLPZen* mlpZen = reinterpret_cast<MLPZen*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (message) {
	case WM_CREATE:
		// Save the pointer passed in to CreateWindow as lParam.
		pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		break;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_KEYUP:
		switch (wParam)
		{
		case VK_F5:
			if (mlpZen != nullptr) {
				mlpZen->ReloadShaders();
			}
			break;
		case VK_F6:
			if (mlpZen != nullptr) {
				mlpZen->TrainingStep();
			}
			break;
		}
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

HRESULT Create(LONG width, LONG height, HINSTANCE& instance, HWND& window, LPCWSTR title, MLPZen* mlpZen) {

	// Register the window class
	WNDCLASSEX wcex = { 0 };
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = instance;
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = L"MLPZen";
	wcex.hIcon = nullptr;
	wcex.hIconSm = nullptr;

	if (!RegisterClassEx(&wcex)) {
		utils::validate(E_FAIL, L"Error: failed to register window!");
	}

	// Get the desktop resolution
	RECT desktop;
	const HWND hDesktop = GetDesktopWindow();
	GetWindowRect(hDesktop, &desktop);

	int x = (desktop.right - width) / 2;
	int y = (desktop.bottom - height) / 3;

	// Create the window
	RECT rc = { 0, 0, width, height };
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

	window = CreateWindow(wcex.lpszClassName, title, WS_OVERLAPPEDWINDOW, x, y, (rc.right - rc.left), (rc.bottom - rc.top), NULL, NULL, instance, mlpZen);
	if (!window) return E_FAIL;

	// Show the window
	ShowWindow(window, SW_SHOWDEFAULT);
	UpdateWindow(window);

	return S_OK;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	HRESULT hr = EXIT_SUCCESS;

	{
		MSG msg = { 0 };
		HWND hWnd = { 0 };

		// Tell Windows that we're DPI aware (we handle scaling ourselves, e.g. the scaling of GUI)
		SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

		MLPZen mlpZen;

		// Initialize window
		HRESULT hr = Create(frameWidth, frameHeight, hInstance, hWnd, L"MLP Zen", &mlpZen);
		utils::validate(hr, L"Error: failed to create window!");

		mlpZen.Initialize(hWnd);

		std::chrono::steady_clock::time_point lastFrameTime = {};

		// Main loop
		while (WM_QUIT != msg.message)
		{
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			// Calculate frame time
			const float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - lastFrameTime).count() * 0.001f;
			lastFrameTime = std::chrono::steady_clock::now();

			// Break the loop here when the game is over
			if (!mlpZen.Update(hWnd, elapsedTime)) break;
		}

		mlpZen.Cleanup();
	}

	return hr;
}
