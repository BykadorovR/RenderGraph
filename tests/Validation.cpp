#define VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>
#include <volk.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

import Allocator;
import Buffer;
import DescriptorBuffer;
import Device;
import Graph;
import Instance;
import Pipeline;
import Shader;
import Surface;
import Swapchain;
import Texture;
import Window;
import glm;

namespace {
class ValidationErrorCollector final {
 private:
  std::mutex _mutex;
  std::vector<std::string> _errors;

 public:
  void add(std::string message) {
    std::scoped_lock lock(_mutex);
    _errors.push_back(std::move(message));
  }

  std::vector<std::string> getErrors() {
    std::scoped_lock lock(_mutex);
    return _errors;
  }
};

VkBool32 validationErrorCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                 VkDebugUtilsMessageTypeFlagsEXT messageType,
                                 const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                 void* userData) {
  if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) == 0 || userData == nullptr ||
      callbackData == nullptr || callbackData->pMessage == nullptr) {
    return false;
  }

  auto* collector = static_cast<ValidationErrorCollector*>(userData);
  std::string message;
  if (callbackData->pMessageIdName != nullptr) {
    message += callbackData->pMessageIdName;
    message += ": ";
  }
  message += callbackData->pMessage;
  collector->add(std::move(message));
  return false;
}

class ValidationMessenger final {
 private:
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _messenger = nullptr;

 public:
  ValidationMessenger(VkInstance instance, ValidationErrorCollector& collector) : _instance(instance) {
    const VkDebugUtilsMessengerCreateInfoEXT createInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = validationErrorCallback,
        .pUserData = &collector,
    };

    const VkResult status = vkCreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_messenger);
    if (status != VK_SUCCESS) {
      throw std::runtime_error("Failed to create test validation messenger: " + std::to_string(status));
    }
  }

  ValidationMessenger(const ValidationMessenger&) = delete;
  ValidationMessenger& operator=(const ValidationMessenger&) = delete;

  ~ValidationMessenger() {
    if (_messenger != nullptr) {
      vkDestroyDebugUtilsMessengerEXT(_instance, _messenger, nullptr);
    }
  }
};

class ValidationGraphElement final : public RenderGraph::GraphElement {
 private:
  std::atomic<int> _drawCount = 0;
  std::atomic<int> _resetCount = 0;

 public:
  void draw(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override { ++_drawCount; }

  void update(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {}

  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) override { ++_resetCount; }

  int getDrawCount() const noexcept { return _drawCount; }
  int getResetCount() const noexcept { return _resetCount; }
};

std::vector<char> readTestShader(std::string_view filename) {
  const std::filesystem::path path = std::filesystem::path(renderGraphTestShaderDir) / filename;
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open test shader " + path.string());
  }

  const auto fileSize = static_cast<std::size_t>(file.tellg());
  std::vector<char> code(fileSize);
  file.seekg(0);
  file.read(code.data(), static_cast<std::streamsize>(fileSize));
  return code;
}

class GpuDrivenComputeElement final : public RenderGraph::GraphElement {
 private:
  RenderGraph::Shader _shader;
  RenderGraph::DescriptorSetLayout _descriptorSetLayout;
  RenderGraph::DescriptorPool _descriptorPool;
  RenderGraph::Pipeline _pipeline;
  std::unique_ptr<RenderGraph::DescriptorSet> _descriptorSet;

 public:
  GpuDrivenComputeElement(const RenderGraph::Device& device,
                          const RenderGraph::CommandBuffer& initializationCommandBuffer,
                          const std::vector<RenderGraph::Buffer*>& positions,
                          const std::vector<RenderGraph::Buffer*>& indices,
                          const std::vector<RenderGraph::Buffer*>& drawCommands,
                          const std::vector<RenderGraph::Buffer*>& drawCounts)
      : _shader(device),
        _descriptorSetLayout(device),
        _descriptorPool(RenderGraph::DescriptorPoolSize{}, device),
        _pipeline(device) {
    if (positions.size() != indices.size() || positions.size() != drawCommands.size() ||
        positions.size() != drawCounts.size()) {
      throw std::logic_error("GPU-driven buffers must have the same frame count");
    }

    _shader.add(readTestShader("gpuDriven.comp.spv"));
    const auto& reflectedLayouts = _shader.getDescriptorSetLayoutBindings();
    if (reflectedLayouts.size() != 1) {
      throw std::logic_error("GPU-driven compute shader must have exactly one descriptor set");
    }
    _descriptorSetLayout.createCustom(reflectedLayouts.front());

    std::vector<RenderGraph::DescriptorSetLayout*> descriptorSetLayouts{&_descriptorSetLayout};
    const auto shaderStages = _shader.getShaderStageInfo();
    if (shaderStages.size() != 1 || shaderStages.front().stage != VK_SHADER_STAGE_COMPUTE_BIT) {
      throw std::logic_error("GPU-driven compute shader has an unexpected stage");
    }
    _pipeline.createCompute(shaderStages.front(), descriptorSetLayouts, {});

    _descriptorSet =
        std::make_unique<RenderGraph::DescriptorSet>(descriptorSetLayouts, _descriptorPool, device);
    for (std::size_t frameIndex = 0; frameIndex < positions.size(); ++frameIndex) {
      _descriptorSet->add({positions[frameIndex]});
      _descriptorSet->add({indices[frameIndex]});
      _descriptorSet->add({drawCommands[frameIndex]});
      _descriptorSet->add({drawCounts[frameIndex]});
    }
    _descriptorSet->initialize(initializationCommandBuffer);
  }

  void draw(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {
    vkCmdBindPipeline(commandBuffer.getCommandBuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline.getPipeline());
    _descriptorSet->bind(VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline.getPipelineLayout(), commandBuffer);
    vkCmdDispatch(commandBuffer.getCommandBuffer(), 1, 1, 1);
  }

  void update(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {}

  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) override {}
};

class GpuDrivenGraphicElement final : public RenderGraph::GraphElement {
 private:
  const RenderGraph::Device* _device;
  RenderGraph::Shader _shader;
  RenderGraph::DescriptorSetLayout _descriptorSetLayout;
  RenderGraph::DescriptorPool _descriptorPool;
  RenderGraph::Pipeline _pipeline;
  std::unique_ptr<RenderGraph::DescriptorSet> _descriptorSet;
  std::vector<RenderGraph::Buffer*> _indices;
  std::vector<RenderGraph::Buffer*> _drawCommands;
  std::vector<RenderGraph::Buffer*> _drawCounts;
  VkExtent2D _resolution;
  VkQueryPool _queryPool = nullptr;

 public:
  GpuDrivenGraphicElement(const RenderGraph::Device& device,
                          const RenderGraph::CommandBuffer& initializationCommandBuffer,
                          RenderGraph::PipelineGraphic& pipelineGraphic,
                          VkExtent2D resolution,
                          const std::vector<RenderGraph::Buffer*>& positions,
                          std::vector<RenderGraph::Buffer*> indices,
                          std::vector<RenderGraph::Buffer*> drawCommands,
                          std::vector<RenderGraph::Buffer*> drawCounts)
      : _device(&device),
        _shader(device),
        _descriptorSetLayout(device),
        _descriptorPool(RenderGraph::DescriptorPoolSize{}, device),
        _pipeline(device),
        _indices(std::move(indices)),
        _drawCommands(std::move(drawCommands)),
        _drawCounts(std::move(drawCounts)),
        _resolution(resolution) {
    if (positions.size() != _indices.size() || positions.size() != _drawCommands.size() ||
        positions.size() != _drawCounts.size()) {
      throw std::logic_error("GPU-driven buffers must have the same frame count");
    }

    _shader.add(readTestShader("gpuDriven.vert.spv"));
    _shader.add(readTestShader("gpuDriven.frag.spv"));
    const auto& reflectedLayouts = _shader.getDescriptorSetLayoutBindings();
    if (reflectedLayouts.size() != 1) {
      throw std::logic_error("GPU-driven graphic shader must have exactly one descriptor set");
    }
    _descriptorSetLayout.createCustom(reflectedLayouts.front());

    std::vector<RenderGraph::DescriptorSetLayout*> descriptorSetLayouts{&_descriptorSetLayout};
    pipelineGraphic.setDepthTest(false);
    pipelineGraphic.setDepthWrite(false);
    pipelineGraphic.setAlphaBlending(false);
    _pipeline.createGraphic(pipelineGraphic, _shader.getShaderStageInfo(), descriptorSetLayouts, {},
                            *_shader.getVertexInputInfo());

    _descriptorSet =
        std::make_unique<RenderGraph::DescriptorSet>(descriptorSetLayouts, _descriptorPool, device);
    for (auto* positionBuffer : positions) {
      _descriptorSet->add({positionBuffer});
    }
    _descriptorSet->initialize(initializationCommandBuffer);

    const VkQueryPoolCreateInfo queryPoolInfo{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_OCCLUSION,
        .queryCount = static_cast<uint32_t>(positions.size()),
    };
    if (vkCreateQueryPool(device.getLogicalDevice(), &queryPoolInfo, nullptr, &_queryPool) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create GPU-driven occlusion query pool");
    }
  }

  void draw(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {
    const VkCommandBuffer vkCommandBuffer = commandBuffer.getCommandBuffer();
    vkCmdBindPipeline(vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline.getPipeline());
    _descriptorSet->bind(VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline.getPipelineLayout(), commandBuffer);

    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(_resolution.width),
        .height = static_cast<float>(_resolution.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = _resolution};
    vkCmdSetViewport(vkCommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(vkCommandBuffer, 0, 1, &scissor);
    vkCmdSetDepthBias(vkCommandBuffer, 0.0f, 0.0f, 0.0f);
    vkCmdBindIndexBuffer(vkCommandBuffer, _indices.at(currentFrame)->getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    vkCmdBeginQuery(vkCommandBuffer, _queryPool, static_cast<uint32_t>(currentFrame), 0);
    vkCmdDrawIndexedIndirectCount(vkCommandBuffer, _drawCommands.at(currentFrame)->getBuffer(), 0,
                                  _drawCounts.at(currentFrame)->getBuffer(), 0, 1,
                                  sizeof(VkDrawIndexedIndirectCommand));
    vkCmdEndQuery(vkCommandBuffer, _queryPool, static_cast<uint32_t>(currentFrame));
  }

  void update(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {
    vkCmdResetQueryPool(commandBuffer.getCommandBuffer(), _queryPool, static_cast<uint32_t>(currentFrame), 1);
  }

  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) override {}

  std::uint64_t getPassedSamples(int frameIndex) const {
    std::uint64_t passedSamples = 0;
    const VkResult result = vkGetQueryPoolResults(
        _device->getLogicalDevice(), _queryPool, static_cast<uint32_t>(frameIndex), 1, sizeof(passedSamples),
        &passedSamples, sizeof(passedSamples), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to read GPU-driven occlusion query: " + std::to_string(result));
    }
    return passedSamples;
  }

  ~GpuDrivenGraphicElement() override {
    if (_queryPool != nullptr) {
      vkDestroyQueryPool(_device->getLogicalDevice(), _queryPool, nullptr);
    }
  }
};

std::string joinErrors(const std::vector<std::string>& errors) {
  std::string result;
  for (const std::string& error : errors) {
    if (!result.empty()) {
      result += '\n';
    }
    result += error;
  }
  return result;
}

class ValidationScenarioTest : public testing::TestWithParam<bool> {};
}  // namespace

TEST_P(ValidationScenarioTest, FullGraphPipelineHasNoValidationErrorsAcrossResets) {
  const bool separateComputeQueue = GetParam();
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("ValidationGraphTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 3;
    constexpr int passCount = 7;
    constexpr int resetCycleCount = 3;
    const glm::ivec2 resolution(1280, 720);

    RenderGraph::Window window(resolution);
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(surface, instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);
    graph.initialize();

    if (!device.isFormatFeatureSupported(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                         VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
      GTEST_SKIP() << "VK_FORMAT_R16G16B16A16_SFLOAT storage images are unavailable";
    }

    graph.getGraphStorage().add(
        "Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(
                         swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); }));

    std::vector<std::shared_ptr<RenderGraph::ImageView>> colorImageViews;
    std::vector<std::shared_ptr<RenderGraph::ImageView>> depthImageViews;
    colorImageViews.reserve(framesInFlight);
    depthImageViews.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      auto colorImage = std::make_unique<RenderGraph::Image>(allocator);
      colorImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_STORAGE_BIT);
      auto colorImageView = std::make_shared<RenderGraph::ImageView>(std::move(colorImage), device);
      colorImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      colorImageViews.push_back(std::move(colorImageView));

      auto depthImage = std::make_unique<RenderGraph::Image>(allocator);
      depthImage->createImage(VK_FORMAT_D32_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_DEPTH_BIT,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      auto depthImageView = std::make_shared<RenderGraph::ImageView>(std::move(depthImage), device);
      depthImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      depthImageViews.push_back(std::move(depthImageView));
    }

    graph.getGraphStorage().add(
        "Color", std::make_unique<RenderGraph::ImageViewHolder>(
                     colorImageViews, [&graph]() { return graph.getFrameInFlight(); }));
    graph.getGraphStorage().add(
        "Depth", std::make_unique<RenderGraph::ImageViewHolder>(
                     depthImageViews, [&graph]() { return graph.getFrameInFlight(); }));

    std::vector<std::unique_ptr<RenderGraph::Buffer>> storageBuffers;
    storageBuffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      storageBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
          4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          allocator));
    }
    graph.getGraphStorage().add("Data", storageBuffers);

    std::vector<std::unique_ptr<RenderGraph::Buffer>> indirectBuffers;
    indirectBuffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      indirectBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
          4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
    }
    graph.getGraphStorage().add("IndirectCommands", indirectBuffers);

    auto graphElement = std::make_shared<ValidationGraphElement>();

    auto& unsynchronizedPass = graph.createPassCompute("Unsynchronized", false);
    unsynchronizedPass.addStorageBufferOutput("Data");
    unsynchronizedPass.registerGraphElement(graphElement);

    auto& acquirePass = graph.createPassGraphic("Acquire");
    acquirePass.addColorTarget("Swapchain");
    acquirePass.clearTarget("Swapchain");
    acquirePass.registerGraphElement(graphElement);

    auto& preparePass = graph.createPassCompute("Prepare", false);
    preparePass.addStorageBufferInput("Data");
    preparePass.addStorageBufferOutput("Data");
    preparePass.addStorageTextureInput("Swapchain");
    preparePass.addStorageTextureOutput("Swapchain");
    preparePass.registerGraphElement(graphElement);

    auto& geometryPass = graph.createPassGraphic("Geometry");
    geometryPass.addTextureInput("Swapchain");
    geometryPass.addColorTarget("Color");
    geometryPass.setDepthTarget("Depth");
    geometryPass.clearTarget("Color");
    geometryPass.clearTarget("Depth");
    geometryPass.registerGraphElement(graphElement);

    auto& asyncPass = graph.createPassCompute("Async", separateComputeQueue);
    asyncPass.addStorageBufferInput("Data");
    asyncPass.addStorageBufferOutput("Data");
    asyncPass.addStorageBufferOutput("IndirectCommands");
    asyncPass.addStorageTextureInput("Color");
    asyncPass.addStorageTextureOutput("Color");
    asyncPass.registerGraphElement(graphElement);

    auto& compositePass = graph.createPassGraphic("Composite");
    compositePass.addTextureInput("Color");
    compositePass.addStorageBufferInput("Data");
    compositePass.addIndirectBufferInput("IndirectCommands");
    compositePass.addColorTarget("Swapchain");
    compositePass.clearTarget("Swapchain");
    compositePass.registerGraphElement(graphElement);

    auto& depthOnlyPass = graph.createPassGraphic("DepthOnly");
    depthOnlyPass.addTextureInput("Swapchain");
    depthOnlyPass.setDepthTarget("Depth");
    depthOnlyPass.clearTarget("Depth");
    depthOnlyPass.registerGraphElement(graphElement);

    graph.calculate();
    graph.calculate();
    ASSERT_NO_THROW(graph.reset());
    int totalResetCount = 1;

    constexpr int initialFrameCount = framesInFlight + 2;
    for (int frameIndex = 0; frameIndex < initialFrameCount; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }

    constexpr int framesPerReset = framesInFlight + 1;
    for (int resetIndex = 0; resetIndex < resetCycleCount; ++resetIndex) {
      ASSERT_NO_THROW(graph.reset());
      ++totalResetCount;
      if (resetIndex == 1) {
        ASSERT_NO_THROW(graph.reset());
        ++totalResetCount;
      }
      for (int frameIndex = 0; frameIndex < framesPerReset; ++frameIndex) {
        ASSERT_FALSE(graph.render());
      }
    }

    EXPECT_EQ(graphElement->getDrawCount(), passCount * (initialFrameCount + resetCycleCount * framesPerReset));
    EXPECT_EQ(graphElement->getResetCount(), passCount * totalResetCount);
    EXPECT_EQ(graph.getTimestamps().size(), passCount);
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST_P(ValidationScenarioTest, GpuDrivenIndexedIndirectCountDrawsTriangle) {
  const bool separateComputeQueue = GetParam();
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("GpuDrivenValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 2;
    const glm::ivec2 resolution(256, 256);

    RenderGraph::Window window(resolution);
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(surface, instance);
    // The test intentionally exercises ordinary descriptor sets. Descriptor-buffer
    // behavior is covered independently by DescriptorBufferTest.
    device.setOptionalExtensions({});
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    RenderGraph::Graph graph(2, framesInFlight, swapchain, window, device);
    graph.initialize();

    graph.getGraphStorage().add(
        "Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(
                         swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); }));

    auto addBuffers = [&](std::string_view name, VkDeviceSize size, VkBufferUsageFlags usage) {
      std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
      buffers.reserve(framesInFlight);
      for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
        buffers.push_back(std::make_unique<RenderGraph::Buffer>(
            size, usage, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
      }
      graph.getGraphStorage().add(name, buffers);
    };

    addBuffers("Positions", sizeof(float) * 4 * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    addBuffers("Indices", sizeof(uint32_t) * 3,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    addBuffers("DrawCommands", sizeof(VkDrawIndexedIndirectCommand),
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    addBuffers("DrawCounts", sizeof(uint32_t),
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    const auto positions = graph.getGraphStorage().getBuffer("Positions");
    const auto indices = graph.getGraphStorage().getBuffer("Indices");
    const auto drawCommands = graph.getGraphStorage().getBuffer("DrawCommands");
    const auto drawCounts = graph.getGraphStorage().getBuffer("DrawCounts");

    auto& generatePass = graph.createPassCompute("GenerateDraw", separateComputeQueue);
    generatePass.addStorageBufferOutput("Positions");
    generatePass.addStorageBufferOutput("Indices");
    generatePass.addStorageBufferOutput("DrawCommands");
    generatePass.addStorageBufferOutput("DrawCounts");

    auto& drawPass = graph.createPassGraphic("IndirectDraw");
    drawPass.addStorageBufferInput("Positions");
    drawPass.addIndexBufferInput("Indices");
    drawPass.addIndirectBufferInput("DrawCommands");
    drawPass.addIndirectBufferInput("DrawCounts");
    drawPass.addColorTarget("Swapchain");
    drawPass.clearTarget("Swapchain");

    auto computeElement = std::make_shared<GpuDrivenComputeElement>(
        device, *generatePass.getCommandBuffers().front(), positions, indices, drawCommands, drawCounts);
    auto graphicElement = std::make_shared<GpuDrivenGraphicElement>(
        device, *drawPass.getCommandBuffers().front(), drawPass.getPipelineGraphic(graph.getGraphStorage()),
        VkExtent2D{static_cast<uint32_t>(resolution.x), static_cast<uint32_t>(resolution.y)}, positions, indices,
        drawCommands, drawCounts);
    generatePass.registerGraphElement(computeElement);
    drawPass.registerGraphElement(graphicElement);

    graph.calculate();
    std::vector<int> renderedFrames;
    renderedFrames.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      renderedFrames.push_back(graph.getFrameInFlight());
      ASSERT_FALSE(graph.render());
    }
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);

    for (const int renderedFrame : renderedFrames) {
      EXPECT_GT(graphicElement->getPassedSamples(renderedFrame), 0u);
    }
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, OffscreenComputeGraphPresentsWithoutWritingSwapchain) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("ValidationOffscreenGraphTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 3;
    constexpr int framesBeforeReset = framesInFlight + 2;
    constexpr int framesAfterReset = framesInFlight + 1;
    const glm::ivec2 resolution(1280, 720);

    RenderGraph::Window window(resolution);
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(surface, instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    RenderGraph::Graph graph(2, framesInFlight, swapchain, window, device);
    graph.initialize();

    std::vector<std::unique_ptr<RenderGraph::Buffer>> storageBuffers;
    storageBuffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      storageBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
          4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          allocator));
    }
    graph.getGraphStorage().add("Data", storageBuffers);

    auto graphElement = std::make_shared<ValidationGraphElement>();
    auto& computePass = graph.createPassCompute("OffscreenCompute", false);
    computePass.addStorageBufferOutput("Data");
    computePass.registerGraphElement(graphElement);
    graph.calculate();

    for (int frameIndex = 0; frameIndex < framesBeforeReset; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }

    ASSERT_NO_THROW(graph.reset());

    for (int frameIndex = 0; frameIndex < framesAfterReset; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }

    EXPECT_EQ(graphElement->getDrawCount(), framesBeforeReset + framesAfterReset);
    EXPECT_EQ(graphElement->getResetCount(), 1);
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

INSTANTIATE_TEST_SUITE_P(
    QueueModes, ValidationScenarioTest, testing::Values(false, true),
    [](const testing::TestParamInfo<ValidationScenarioTest::ParamType>& info) {
      return info.param ? "SeparateComputeQueue" : "SingleQueue";
    });
