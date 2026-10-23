#define VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>
#include <volk.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

import Allocator;
import Buffer;
import Command;
import CommandPool;
import DescriptorBuffer;
import Device;
import Graph;
import Instance;
import Instrumentation;
import Pipeline;
import ResourceUploader;
import Shader;
import Surface;
import Swapchain;
import Sync;
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
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
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

// Keep the driver and validation layers active while inspecting what the graph records/submits.
class GraphCommandRecorder final {
 private:
  inline static GraphCommandRecorder* _active = nullptr;
  PFN_vkCmdPipelineBarrier2 _pipelineBarrier = vkCmdPipelineBarrier2;
  PFN_vkQueueSubmit2 _queueSubmit = vkQueueSubmit2;
  std::mutex _mutex;

  static VKAPI_ATTR void VKAPI_CALL recordBarrier(VkCommandBuffer commandBuffer, const VkDependencyInfo* dependency) {
    auto& recorder = *_active;
    {
      std::scoped_lock lock(recorder._mutex);
      for (uint32_t index = 0; index < dependency->imageMemoryBarrierCount; ++index) {
        recorder.imageBarriers.emplace_back(commandBuffer, dependency->pImageMemoryBarriers[index]);
      }
      for (uint32_t index = 0; index < dependency->bufferMemoryBarrierCount; ++index) {
        recorder.bufferBarriers.push_back(dependency->pBufferMemoryBarriers[index]);
      }
    }
    recorder._pipelineBarrier(commandBuffer, dependency);
  }

  static VKAPI_ATTR VkResult VKAPI_CALL recordSubmit(VkQueue queue,
                                                     uint32_t submitCount,
                                                     const VkSubmitInfo2* submits,
                                                     VkFence fence) {
    auto& recorder = *_active;
    {
      std::scoped_lock lock(recorder._mutex);
      for (uint32_t index = 0; index < submitCount; ++index) {
        const auto& submit = submits[index];
        recorder.submissionQueues.push_back(queue);
        recorder.submissionBuffers.emplace_back();
        for (uint32_t buffer = 0; buffer < submit.commandBufferInfoCount; ++buffer) {
          recorder.submissionBuffers.back().push_back(submit.pCommandBufferInfos[buffer].commandBuffer);
        }
      }
    }
    return recorder._queueSubmit(queue, submitCount, submits, fence);
  }

 public:
  std::vector<std::pair<VkCommandBuffer, VkImageMemoryBarrier2>> imageBarriers;
  std::vector<VkBufferMemoryBarrier2> bufferBarriers;
  std::vector<VkQueue> submissionQueues;
  std::vector<std::vector<VkCommandBuffer>> submissionBuffers;

  GraphCommandRecorder() {
    _active = this;
    vkCmdPipelineBarrier2 = recordBarrier;
    vkQueueSubmit2 = recordSubmit;
  }
  GraphCommandRecorder(const GraphCommandRecorder&) = delete;
  GraphCommandRecorder& operator=(const GraphCommandRecorder&) = delete;
  ~GraphCommandRecorder() {
    vkCmdPipelineBarrier2 = _pipelineBarrier;
    vkQueueSubmit2 = _queueSubmit;
    _active = nullptr;
  }
};

class ResourceSharingRecorder final {
 private:
  inline static ResourceSharingRecorder* _active = nullptr;
  PFN_vkCreateBuffer _createBuffer = vkCreateBuffer;
  PFN_vkCreateImage _createImage = vkCreateImage;
  const RenderGraph::Device& _device;

  void check(VkSharingMode mode, uint32_t count, const uint32_t* families) {
    EXPECT_EQ(mode, _device.getResourceSharingMode());
    ASSERT_NE(families, nullptr);
    EXPECT_EQ(std::vector<uint32_t>(families, families + count), _device.getResourceQueueFamilies());
  }

 public:
  uint32_t buffers = 0, images = 0;
  explicit ResourceSharingRecorder(const RenderGraph::Device& device) : _device(device) {
    _active = this;
    vkCreateBuffer = [](VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* callbacks,
                        VkBuffer* buffer) {
      auto& recorder = *_active;
      ++recorder.buffers;
      recorder.check(info->sharingMode, info->queueFamilyIndexCount, info->pQueueFamilyIndices);
      return recorder._createBuffer(device, info, callbacks, buffer);
    };
    vkCreateImage = [](VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* callbacks,
                       VkImage* image) {
      auto& recorder = *_active;
      ++recorder.images;
      recorder.check(info->sharingMode, info->queueFamilyIndexCount, info->pQueueFamilyIndices);
      return recorder._createImage(device, info, callbacks, image);
    };
  }
  ~ResourceSharingRecorder() {
    vkCreateBuffer = _createBuffer;
    vkCreateImage = _createImage;
    _active = nullptr;
  }
};

bool supportsStorageSwapchain(const RenderGraph::Device& device, const RenderGraph::Surface& surface) {
  VkSurfaceCapabilitiesKHR capabilities{};
  if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.getPhysicalDevice(), surface.getSurface(), &capabilities) !=
          VK_SUCCESS ||
      (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) == 0) {
    return false;
  }

  uint32_t formatCount = 0;
  if (vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(), surface.getSurface(), &formatCount, nullptr) !=
          VK_SUCCESS ||
      formatCount == 0) {
    return false;
  }

  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  if (vkGetPhysicalDeviceSurfaceFormatsKHR(device.getPhysicalDevice(), surface.getSurface(), &formatCount,
                                           formats.data()) != VK_SUCCESS) {
    return false;
  }

  constexpr VkSurfaceFormatKHR desiredFormat{
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
  };
  VkSurfaceFormatKHR selectedFormat = formats.front();
  for (const VkSurfaceFormatKHR format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == desiredFormat.colorSpace) {
      selectedFormat = format;
      break;
    }
  }
  for (const VkSurfaceFormatKHR format : formats) {
    if (format.format == desiredFormat.format && format.colorSpace == desiredFormat.colorSpace) {
      selectedFormat = format;
      break;
    }
  }

  VkImageFormatProperties properties{};
  return vkGetPhysicalDeviceImageFormatProperties(
             device.getPhysicalDevice(), selectedFormat.format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 0, &properties) == VK_SUCCESS;
}

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

    _descriptorSet = std::make_unique<RenderGraph::DescriptorSet>(descriptorSetLayouts, _descriptorPool, device);
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

    _descriptorSet = std::make_unique<RenderGraph::DescriptorSet>(descriptorSetLayouts, _descriptorPool, device);
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

TEST(ValidationTest, ReuploadsAllImageMipLevels) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("MipmapReuploadValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  {
    RenderGraph::Device device(instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Image image(allocator, device);
    image.createImage(VK_FORMAT_R8G8B8A8_UNORM, {4, 4}, 3, 2, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    RenderGraph::ResourceUploader uploader(allocator, device);
    const std::array<std::byte, 128> pixels{};
    const std::array<VkDeviceSize, 2> offsets{0, 64};
    for (int batch = 0; batch < 2; ++batch) {
      // Exercise both repeated writes in one command buffer and writes across submissions.
      uploader.upload(image, pixels, offsets);
      uploader.upload(image, pixels, offsets);
      uploader.submit();
    }
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
  }
  const auto errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, EmptyGraphRenderIsNoOp) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("EmptyGraphValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  {
    RenderGraph::Window window({64, 64});
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain({64, 64}, allocator, device);
    swapchain.initialize();
    RenderGraph::Tracer tracer(device);
    for (bool withSwapchain : {false, true}) {
      SCOPED_TRACE(withSwapchain);
      RenderGraph::Graph graph(1, 2, device, tracer);
      if (withSwapchain) graph.setSwapchain(swapchain);
      graph.initialize();
      graph.calculate();
      for (int frame = 0; frame < swapchain.getImageCount() + 2; ++frame) {
        ASSERT_FALSE(graph.render());
        EXPECT_EQ(graph.getFrameInFlight(), 0);
      }
      // Empty renders must leave synchronization usable when passes are subsequently added.
      if (withSwapchain) {
        graph.getGraphStorage().add(
            "Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(
                             swapchain.getImageViews(), [&swapchain] { return swapchain.getSwapchainIndex(); }));
        graph.createPassGraphic("Draw").addColorTarget("Swapchain");
      } else {
        graph.createPassCompute("Compute", false);
      }
      graph.calculate();
      for (int frame = 0; frame < 3; ++frame) ASSERT_FALSE(graph.render());
      ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
    }
  }
  const auto errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, DescriptorSetBindsAtNonZeroSet) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("DescriptorSetOffsetValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  RenderGraph::Window window({64, 64});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  RenderGraph::DeviceRequirements deviceRequirements;
  deviceRequirements.optionalExtensions = {"VK_KHR_dynamic_rendering"};
  device.initialize(deviceRequirements);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                             allocator, device);
  RenderGraph::DescriptorPool descriptorPool({}, device);

  std::vector<RenderGraph::DescriptorSetLayout> layouts;
  layouts.reserve(3);
  for (int i = 0; i < 3; ++i) {
    layouts.emplace_back(device);
    layouts.back().createCustom({{.binding = 0,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  .descriptorCount = 1,
                                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                                  .pImmutableSamplers = nullptr}});
  }

  const std::vector<VkDescriptorSetLayout> nativeLayouts{
      layouts[0].getDescriptorSetLayout(), layouts[1].getDescriptorSetLayout(), layouts[2].getDescriptorSetLayout()};
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = static_cast<std::uint32_t>(nativeLayouts.size()),
      .pSetLayouts = nativeLayouts.data(),
  };
  VkPipelineLayout pipelineLayout = nullptr;
  ASSERT_EQ(vkCreatePipelineLayout(device.getLogicalDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
            VK_SUCCESS);

  {
    RenderGraph::CommandPool commandPool(RenderGraph::QueueType::GRAPHICS, device);
    RenderGraph::CommandBuffer commandBuffer(commandPool, device);
    RenderGraph::DescriptorSet descriptorSet({&layouts[2]}, descriptorPool, device, 2);
    descriptorSet.add({&buffer});
    commandBuffer.beginCommands();
    descriptorSet.initialize(commandBuffer);
    descriptorSet.bind(VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, commandBuffer);
    commandBuffer.endCommands();
  }

  vkDestroyPipelineLayout(device.getLogicalDevice(), pipelineLayout, nullptr);
  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, DescriptorBufferBindsAtNonZeroSet) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("DescriptorBufferOffsetValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  RenderGraph::Window window({64, 64});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  RenderGraph::DeviceRequirements deviceRequirements;
  deviceRequirements.optionalExtensions = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME};
  device.initialize(deviceRequirements);
  if (!device.isExtensionEnabled(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
    GTEST_SKIP() << "VK_EXT_descriptor_buffer or its required features are unavailable";
  }

  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                             allocator, device);
  std::vector<RenderGraph::DescriptorSetLayout> layouts;
  layouts.reserve(3);
  for (int i = 0; i < 3; ++i) {
    layouts.emplace_back(device);
    layouts.back().createCustom({{.binding = 0,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  .descriptorCount = 1,
                                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                                  .pImmutableSamplers = nullptr}});
  }

  const std::vector<VkDescriptorSetLayout> nativeLayouts{
      layouts[0].getDescriptorSetLayout(), layouts[1].getDescriptorSetLayout(), layouts[2].getDescriptorSetLayout()};
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = static_cast<std::uint32_t>(nativeLayouts.size()),
      .pSetLayouts = nativeLayouts.data(),
  };
  VkPipelineLayout pipelineLayout = nullptr;
  ASSERT_EQ(vkCreatePipelineLayout(device.getLogicalDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
            VK_SUCCESS);

  {
    RenderGraph::CommandPool commandPool(RenderGraph::QueueType::GRAPHICS, device);
    RenderGraph::CommandBuffer commandBuffer(commandPool, device);
    RenderGraph::DescriptorBuffer descriptorBuffer({&layouts[2]}, allocator, device, 2);
    descriptorBuffer.add({&buffer});
    commandBuffer.beginCommands();
    descriptorBuffer.initialize(commandBuffer);
    descriptorBuffer.bind(VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, commandBuffer);
    commandBuffer.endCommands();
  }

  vkDestroyPipelineLayout(device.getLogicalDevice(), pipelineLayout, nullptr);
  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

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
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    RenderGraph::DeviceRequirements deviceRequirements;
    deviceRequirements.features.geometryShader = true;
    deviceRequirements.features.tessellationShader = true;
    device.initialize(deviceRequirements);
    if (!supportsStorageSwapchain(device, surface)) {
      GTEST_SKIP() << "Storage swapchain images are unavailable";
    }
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    swapchain.initialize();
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(4, framesInFlight, device, tracer);
    graph.setSwapchain(swapchain);
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
      auto colorImage = std::make_unique<RenderGraph::Image>(allocator, device);
      colorImage->createImage(
          VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
      auto colorImageView = std::make_shared<RenderGraph::ImageView>(std::move(colorImage), device);
      colorImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      colorImageViews.push_back(std::move(colorImageView));

      auto depthImage = std::make_unique<RenderGraph::Image>(allocator, device);
      depthImage->createImage(VK_FORMAT_D32_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_DEPTH_BIT,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      auto depthImageView = std::make_shared<RenderGraph::ImageView>(std::move(depthImage), device);
      depthImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      depthImageViews.push_back(std::move(depthImageView));
    }

    graph.getGraphStorage().add("Color",
                                std::make_unique<RenderGraph::ImageViewHolder>(
                                    colorImageViews, [&graph]() { return graph.getFrameInFlight(); }),
                                RenderGraph::ImageResizePolicy::SWAPCHAIN);
    graph.getGraphStorage().add("Depth",
                                std::make_unique<RenderGraph::ImageViewHolder>(
                                    depthImageViews, [&graph]() { return graph.getFrameInFlight(); }),
                                RenderGraph::ImageResizePolicy::SWAPCHAIN);

    std::vector<std::unique_ptr<RenderGraph::Buffer>> storageBuffers;
    storageBuffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      storageBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
          4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator,
          device));
    }
    graph.getGraphStorage().add("Data", storageBuffers);

    std::vector<std::unique_ptr<RenderGraph::Buffer>> indirectBuffers;
    indirectBuffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      indirectBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
          4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator, device));
    }
    graph.getGraphStorage().add("IndirectCommands", indirectBuffers);

    auto graphElement = std::make_shared<ValidationGraphElement>();

    auto& unsynchronizedPass = graph.createPassCompute("Unsynchronized", false);
    unsynchronizedPass.addStorageBufferOutput("Data");
    unsynchronizedPass.registerGraphElement(graphElement);

    auto& acquirePass = graph.createPassGraphic("Acquire");
    acquirePass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::KEEP);
    acquirePass.registerGraphElement(graphElement);

    auto& preparePass = graph.createPassCompute("Prepare", false);
    preparePass.addStorageBufferInput("Data");
    preparePass.addStorageBufferOutput("Data");
    preparePass.addStorageTextureInput("Swapchain");
    preparePass.addStorageTextureOutput("Swapchain");
    preparePass.registerGraphElement(graphElement);

    auto& geometryPass = graph.createPassGraphic("Geometry");
    geometryPass.addTextureInput("Swapchain");
    geometryPass.addColorTarget("Color", RenderGraph::RenderTargetLoadAction::CLEAR);
    geometryPass.setDepthTarget("Depth", RenderGraph::RenderTargetLoadAction::CLEAR);
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
    compositePass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::CLEAR);
    compositePass.registerGraphElement(graphElement);

    auto& depthOnlyPass = graph.createPassGraphic("DepthOnly");
    depthOnlyPass.addTextureInput("Swapchain");
    depthOnlyPass.setDepthTarget("Depth", RenderGraph::RenderTargetLoadAction::CLEAR);
    depthOnlyPass.registerGraphElement(graphElement);

    graph.calculate();
    graph.calculate();
    ASSERT_NO_THROW(graph.reset(resolution));
    int totalResetCount = 1;
    for (const auto& view : swapchain.getImageViews()) {
      EXPECT_EQ(view->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
    }

    constexpr int initialFrameCount = framesInFlight + 2;
    for (int frameIndex = 0; frameIndex < initialFrameCount; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }

    constexpr int framesPerReset = framesInFlight + 1;
    for (int resetIndex = 0; resetIndex < resetCycleCount; ++resetIndex) {
      ASSERT_NO_THROW(graph.reset(resolution));
      ++totalResetCount;
      if (resetIndex == 1) {
        ASSERT_NO_THROW(graph.reset(resolution));
        ++totalResetCount;
      }
      for (int frameIndex = 0; frameIndex < framesPerReset; ++frameIndex) {
        ASSERT_FALSE(graph.render());
      }
    }

    EXPECT_EQ(graphElement->getDrawCount(), passCount * (initialFrameCount + resetCycleCount * framesPerReset));
    EXPECT_EQ(graphElement->getResetCount(), passCount * totalResetCount);
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
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    // The test intentionally exercises ordinary descriptor sets. Descriptor-buffer
    // behavior is covered independently by DescriptorBufferTest.
    RenderGraph::DeviceRequirements deviceRequirements;
    deviceRequirements.features.geometryShader = true;
    deviceRequirements.features.tessellationShader = true;
    deviceRequirements.features12.drawIndirectCount = true;
    device.initialize(deviceRequirements);
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.setSwapchain(swapchain);
    graph.initialize();

    graph.getGraphStorage().add(
        "Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(
                         swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); }));

    auto addBuffers = [&](std::string_view name, VkDeviceSize size, VkBufferUsageFlags usage) {
      std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
      buffers.reserve(framesInFlight);
      for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
        buffers.push_back(std::make_unique<RenderGraph::Buffer>(
            size, usage, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator, device));
      }
      graph.getGraphStorage().add(name, buffers);
    };

    addBuffers("Positions", sizeof(float) * 4 * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    addBuffers("Indices", sizeof(uint32_t) * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
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
    drawPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::CLEAR);

    auto computeElement = std::make_shared<GpuDrivenComputeElement>(device, *generatePass.getCommandBuffers().front(),
                                                                    positions, indices, drawCommands, drawCounts);
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

TEST_P(ValidationScenarioTest, GraphicsStorageBufferBarriersUseDefaultDeviceFeatures) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("GraphicsStorageBufferValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 2;
    RenderGraph::Device device(instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.initialize();

    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    std::vector<std::shared_ptr<RenderGraph::ImageView>> colorImageViews;
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(
          std::make_unique<RenderGraph::Buffer>(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, allocator, device));
      auto image = std::make_unique<RenderGraph::Image>(allocator, device);
      image->createImage(VK_FORMAT_R8G8B8A8_UNORM, {16, 16}, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
      auto imageView = std::make_shared<RenderGraph::ImageView>(std::move(image), device);
      imageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      colorImageViews.push_back(std::move(imageView));
    }
    graph.getGraphStorage().add("Data", buffers);
    graph.getGraphStorage().add("Color", std::make_unique<RenderGraph::ImageViewHolder>(
                                             colorImageViews, [&graph]() { return graph.getFrameInFlight(); }));

    auto& producer = graph.createPassCompute("Produce", GetParam());
    producer.addStorageBufferOutput("Data");
    auto& consumer = graph.createPassGraphic("Draw");
    consumer.addStorageBufferInput("Data");
    consumer.addColorTarget("Color");
    // Exercise the graphics stage mask in both destination and source scopes.
    auto& rewrite = graph.createPassCompute("Rewrite", GetParam());
    rewrite.addStorageBufferOutput("Data");
    graph.calculate();

    for (int frameIndex = 0; frameIndex < framesInFlight + 1; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST_P(ValidationScenarioTest, ComputePassesWithoutResourcesRunAcrossFrames) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("ResourceFreePassValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 2;
    constexpr int frameCount = framesInFlight + 1;
    RenderGraph::Device device(instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.initialize();

    const std::array passes{&graph.createPassCompute("Before", false), &graph.createPassCompute("Middle", GetParam()),
                            &graph.createPassCompute("After", false)};
    auto graphElement = std::make_shared<ValidationGraphElement>();
    for (auto* pass : passes) {
      pass->registerGraphElement(graphElement);
    }
    ASSERT_NO_THROW(graph.calculate());
    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
    EXPECT_EQ(graphElement->getDrawCount(), frameCount * 3);

    // Recalculate with an empty pass between a producer and its consumer.
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(
          std::make_unique<RenderGraph::Buffer>(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, allocator, device));
    }
    graph.getGraphStorage().add("Data", buffers);
    passes.front()->addStorageBufferOutput("Data");
    passes.back()->addStorageBufferInput("Data");
    ASSERT_NO_THROW(graph.calculate());
    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
    EXPECT_EQ(graphElement->getDrawCount(), frameCount * 6);
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, ComputeGraphRunsWithoutSwapchain) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("ValidationComputeGraphTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }

  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 3;
    constexpr int frameCount = framesInFlight + 2;
    const glm::ivec2 resolution(64, 64);

    RenderGraph::Device device(instance);
    device.initialize();
    if (!device.isFormatFeatureSupported(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                                         VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
      GTEST_SKIP() << "VK_FORMAT_R8G8B8A8_UNORM storage images are unavailable";
    }
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.initialize();

    auto addMappedBuffers = [&](std::string_view name, VkDeviceSize size) {
      std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
      buffers.reserve(framesInFlight);
      for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
        buffers.push_back(std::make_unique<RenderGraph::Buffer>(
            size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, allocator,
            device));
      }
      graph.getGraphStorage().add(name, buffers);
    };

    addMappedBuffers("Positions", sizeof(float) * 4 * 3);
    addMappedBuffers("Indices", sizeof(uint32_t) * 3);
    addMappedBuffers("DrawCommands", sizeof(VkDrawIndexedIndirectCommand));
    addMappedBuffers("DrawCounts", sizeof(uint32_t));

    const auto positions = graph.getGraphStorage().getBuffer("Positions");
    const auto indices = graph.getGraphStorage().getBuffer("Indices");
    const auto drawCommands = graph.getGraphStorage().getBuffer("DrawCommands");
    const auto drawCounts = graph.getGraphStorage().getBuffer("DrawCounts");

    std::vector<std::shared_ptr<RenderGraph::ImageView>> storageImageViews;
    storageImageViews.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      auto storageImage = std::make_unique<RenderGraph::Image>(allocator, device);
      storageImage->createImage(VK_FORMAT_R8G8B8A8_UNORM, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_USAGE_STORAGE_BIT);
      auto storageImageView = std::make_shared<RenderGraph::ImageView>(std::move(storageImage), device);
      storageImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      storageImageViews.push_back(std::move(storageImageView));
    }
    graph.getGraphStorage().add("Image", std::make_unique<RenderGraph::ImageViewHolder>(
                                             storageImageViews, [&graph]() { return graph.getFrameInFlight(); }));

    auto& computePass = graph.createPassCompute("Compute", false);
    computePass.addStorageBufferOutput("Positions");
    computePass.addStorageBufferOutput("Indices");
    computePass.addStorageBufferOutput("DrawCommands");
    computePass.addStorageBufferOutput("DrawCounts");
    computePass.addStorageTextureOutput("Image");
    computePass.registerGraphElement(std::make_shared<GpuDrivenComputeElement>(
        device, *computePass.getCommandBuffers().front(), positions, indices, drawCommands, drawCounts));
    graph.calculate();

    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
      ASSERT_FALSE(graph.render());
    }

    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      ASSERT_EQ(
          vmaInvalidateAllocation(allocator.getAllocator(), drawCounts[frameIndex]->getAllocation(), 0, VK_WHOLE_SIZE),
          VK_SUCCESS);
      const auto* drawCount = static_cast<const uint32_t*>(drawCounts[frameIndex]->getAllocationInfo().pMappedData);
      ASSERT_NE(drawCount, nullptr);
      EXPECT_EQ(*drawCount, 1u);
    }
    for (const auto& imageView : storageImageViews) {
      EXPECT_EQ(imageView->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
    }
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST_P(ValidationScenarioTest, UploadTicketsSynchronizeFirstPassOnEachQueueOnly) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("ValidationUploadTicketsTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    RenderGraph::Device device(instance);
    device.initialize();
    GraphCommandRecorder recorder;
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Tracer tracer(device);
    RenderGraph::ResourceUploader uploader(allocator, device);
    RenderGraph::Graph graph(1, 1, device, tracer);
    graph.initialize();
    const auto destinationQueueType = GetParam() ? RenderGraph::QueueType::COMPUTE : RenderGraph::QueueType::GRAPHICS;

    const std::array<uint32_t, 2> firstData{11, 22};
    const std::array<uint32_t, 2> secondData{33, 44};
    std::vector<RenderGraph::UploadTicket> tickets;
    for (const auto& [name, data] : {std::pair{"First", firstData}, std::pair{"Second", secondData}}) {
      std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(
          sizeof(data),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0,
          allocator, device));
      uploader.upload(*buffers.front(), std::as_bytes(std::span(data)));
      const auto queueType = std::string_view(name) == "First" ? destinationQueueType
                                                               : RenderGraph::QueueType::GRAPHICS;
      tickets.push_back(uploader.submit(queueType));
      EXPECT_EQ(tickets.back().destinationQueueType, queueType);
      graph.getGraphStorage().add(name, buffers);
    }

    RenderGraph::Image image(allocator, device);
    image.createImage(VK_FORMAT_R8G8B8A8_UNORM, {2, 2}, 2, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    const std::array<std::byte, 16> pixels{
        std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42},
        std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42},
        std::byte{42}, std::byte{42}, std::byte{42}, std::byte{42},
    };
    const std::array<VkDeviceSize, 1> offsets{0};
    uploader.upload(image, pixels, offsets);
    tickets.push_back(uploader.submit(destinationQueueType));

    RenderGraph::Buffer firstReadback(12, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      allocator, device);
    RenderGraph::Buffer secondReadback(8, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                       allocator, device);
    class ReadbackElement final : public RenderGraph::GraphElement {
     private:
      VkBuffer _source, _readback;
      VkImage _image;

     public:
      ReadbackElement(VkBuffer source, VkBuffer readback, VkImage image = nullptr)
          : _source(source),
            _readback(readback),
            _image(image) {}

      void draw(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {
        const VkCommandBuffer commands = commandBuffer.getCommandBuffer();
        const VkBufferCopy bufferCopy{0, 0, 8};
        vkCmdCopyBuffer(commands, _source, _readback, 1, &bufferCopy);
        if (_image != nullptr) {
          const VkBufferImageCopy imageCopy{
              .bufferOffset = 8,
              .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1},
              .imageExtent = {1, 1, 1},
          };
          vkCmdCopyImageToBuffer(commands, _image, VK_IMAGE_LAYOUT_GENERAL, _readback, 1, &imageCopy);
        }
        const VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
            .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        };
        const VkDependencyInfo dependencyInfo{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(commands, &dependencyInfo);
      }

      void update(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override {}
      void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) override {}
    };

    auto& firstPass = graph.createPassCompute("First readback", GetParam());
    firstPass.addStorageBufferInput("First");
    firstPass.registerGraphElement(std::make_shared<ReadbackElement>(
        graph.getGraphStorage().getBuffer("First").front()->getBuffer(), firstReadback.getBuffer(), image.getImage()));
    auto& secondPass = graph.createPassCompute("Second readback", false);
    secondPass.addStorageBufferInput("Second");
    secondPass.registerGraphElement(std::make_shared<ReadbackElement>(
        graph.getGraphStorage().getBuffer("Second").front()->getBuffer(), secondReadback.getBuffer()));
    graph.calculate(tickets);
    // Recalculation before submission must retain pending uploads; later recalculations must not repeat waits.
    for (int frame = 0; frame < 3; ++frame) {
      graph.calculate();
      ASSERT_FALSE(graph.render());
      ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
      ASSERT_EQ(vmaInvalidateAllocation(allocator.getAllocator(), firstReadback.getAllocation(), 0, VK_WHOLE_SIZE),
                VK_SUCCESS);
      ASSERT_EQ(vmaInvalidateAllocation(allocator.getAllocator(), secondReadback.getAllocation(), 0, VK_WHOLE_SIZE),
                VK_SUCCESS);
      const auto* firstResult = static_cast<const uint32_t*>(firstReadback.getAllocationInfo().pMappedData);
      const auto* secondResult = static_cast<const uint32_t*>(secondReadback.getAllocationInfo().pMappedData);
      ASSERT_NE(firstResult, nullptr);
      ASSERT_NE(secondResult, nullptr);
      EXPECT_EQ(firstResult[0], 11u);
      EXPECT_EQ(firstResult[1], 22u);
      EXPECT_EQ(firstResult[2], 0x2a2a2a2au);
      EXPECT_EQ(secondResult[0], 33u);
      EXPECT_EQ(secondResult[1], 44u);
      uploader.reclaim();
    }
    ASSERT_FALSE(recorder.bufferBarriers.empty());
    ASSERT_FALSE(recorder.imageBarriers.empty());
    for (const auto& barrier : recorder.bufferBarriers) {
      EXPECT_EQ(barrier.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
      EXPECT_EQ(barrier.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
    }
    for (const auto& [command, barrier] : recorder.imageBarriers) {
      EXPECT_EQ(barrier.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
      EXPECT_EQ(barrier.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
    }
  }

  EXPECT_TRUE(validationErrors.getErrors().empty()) << joinErrors(validationErrors.getErrors());
}

TEST(ValidationTest, GraphStorageResetPreservesFixedImages) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("FixedImageResetValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    RenderGraph::Device device(instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::GraphStorage storage;
    auto createImageView = [&](glm::ivec2 resolution) {
      auto image = std::make_unique<RenderGraph::Image>(allocator, device);
      image->createImage(VK_FORMAT_R8G8B8A8_UNORM, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
      auto imageView = std::make_shared<RenderGraph::ImageView>(std::move(image), device);
      imageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
      return imageView;
    };

    const glm::ivec2 fixedResolution(4, 2);
    const glm::ivec2 initialSwapchainResolution(64, 48);
    auto fixedView = createImageView(fixedResolution);
    auto sameSizeView = createImageView(initialSwapchainResolution);
    auto resizedView = createImageView(initialSwapchainResolution);
    storage.add("Fixed", std::make_unique<RenderGraph::ImageViewHolder>(std::vector{fixedView}, []() { return 0; }));
    storage.add("FixedSameSize",
                std::make_unique<RenderGraph::ImageViewHolder>(std::vector{sameSizeView}, []() { return 0; }));
    storage.add("Resizable",
                std::make_unique<RenderGraph::ImageViewHolder>(std::vector{resizedView}, []() { return 0; }),
                RenderGraph::ImageResizePolicy::SWAPCHAIN);
    std::vector swapchainViews{createImageView(initialSwapchainResolution)};
    storage.add("Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(swapchainViews, []() { return 0; }));

    RenderGraph::ResourceUploader uploader(allocator, device);
    const std::vector<std::byte> pixels(fixedResolution.x * fixedResolution.y * 4, std::byte{0x5a});
    const std::array<VkDeviceSize, 1> offsets{0};
    uploader.upload(fixedView->getImage(), pixels, offsets);
    uploader.submit();
    ASSERT_EQ(vkQueueWaitIdle(device.getQueue(RenderGraph::QueueType::GRAPHICS)), VK_SUCCESS);

    const VkImage fixedImage = fixedView->getImage().getImage();
    const VkImageView fixedImageView = fixedView->getImageView();
    const VkImage sameSizeImage = sameSizeView->getImage().getImage();
    const VkImageView sameSizeImageView = sameSizeView->getImageView();
    for (const glm::ivec2 resolution : {glm::ivec2(96, 64), initialSwapchainResolution, initialSwapchainResolution}) {
      std::vector newSwapchainViews{createImageView(resolution)};
      storage.reset(swapchainViews, newSwapchainViews);
      EXPECT_EQ(storage.getImageViewHolder("Swapchain").getImageView().getImageView(),
                newSwapchainViews.front()->getImageView());
      ASSERT_EQ(fixedView->getImage().getResolution(), fixedResolution);
      EXPECT_EQ(fixedView->getImage().getImage(), fixedImage);
      EXPECT_EQ(fixedView->getImageView(), fixedImageView);
      EXPECT_EQ(fixedView->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
      EXPECT_EQ(sameSizeView->getImage().getResolution(), initialSwapchainResolution);
      EXPECT_EQ(sameSizeView->getImage().getImage(), sameSizeImage);
      EXPECT_EQ(sameSizeView->getImageView(), sameSizeImageView);
      EXPECT_EQ(resizedView->getImage().getResolution(), resolution);
      swapchainViews = std::move(newSwapchainViews);
    }

    // Replacing a resizable entry with a fixed image must also replace its policy.
    auto replacementView = createImageView(fixedResolution);
    const VkImage replacementImage = replacementView->getImage().getImage();
    storage.add("Resizable",
                std::make_unique<RenderGraph::ImageViewHolder>(std::vector{replacementView}, []() { return 0; }));
    std::vector replacementSwapchainViews{createImageView({80, 56})};
    storage.reset(swapchainViews, replacementSwapchainViews);
    EXPECT_EQ(replacementView->getImage().getResolution(), fixedResolution);
    EXPECT_EQ(replacementView->getImage().getImage(), replacementImage);

    RenderGraph::Buffer readback(pixels.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                 allocator, device);
    RenderGraph::CommandPool commandPool(RenderGraph::QueueType::GRAPHICS, device);
    RenderGraph::CommandBuffer commandBuffer(commandPool, device);
    commandBuffer.beginCommands();
    RenderGraph::Barrier beforeReadback;
    beforeReadback.addImage(fixedImage, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    beforeReadback.execute(commandBuffer.getCommandBuffer());
    const VkBufferImageCopy imageCopy{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {static_cast<uint32_t>(fixedResolution.x), static_cast<uint32_t>(fixedResolution.y), 1},
    };
    vkCmdCopyImageToBuffer(commandBuffer.getCommandBuffer(), fixedImage, VK_IMAGE_LAYOUT_GENERAL, readback.getBuffer(),
                           1, &imageCopy);
    const VkMemoryBarrier2 readbackBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
    };
    const VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &readbackBarrier,
    };
    vkCmdPipelineBarrier2(commandBuffer.getCommandBuffer(), &dependencyInfo);
    commandBuffer.endCommands();
    const VkCommandBufferSubmitInfo commandBufferInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = commandBuffer.getCommandBuffer(),
    };
    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferInfo,
    };
    ASSERT_EQ(vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GRAPHICS), 1, &submitInfo, nullptr), VK_SUCCESS);
    ASSERT_EQ(vkQueueWaitIdle(device.getQueue(RenderGraph::QueueType::GRAPHICS)), VK_SUCCESS);
    ASSERT_EQ(vmaInvalidateAllocation(allocator.getAllocator(), readback.getAllocation(), 0, VK_WHOLE_SIZE),
              VK_SUCCESS);
    const auto* result = static_cast<const std::byte*>(readback.getAllocationInfo().pMappedData);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(std::vector<std::byte>(result, result + pixels.size()), pixels);
  }

  const std::vector<std::string> errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, SampledImageLayoutsPreserveUploadedContentsAcrossFrames) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("SampledImageLayoutValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  {
    constexpr int framesInFlight = 2;
    const glm::ivec2 resolution(4, 4);
    RenderGraph::Device device(instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.initialize();
    auto addImages = [&](std::string_view name, VkImageUsageFlags usage) {
      std::vector<std::shared_ptr<RenderGraph::ImageView>> views;
      for (int frame = 0; frame < framesInFlight; ++frame) {
        auto image = std::make_unique<RenderGraph::Image>(allocator, device);
        image->createImage(VK_FORMAT_R8G8B8A8_UNORM, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT, usage);
        auto view = std::make_shared<RenderGraph::ImageView>(std::move(image), device);
        view->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
        views.push_back(std::move(view));
      }
      graph.getGraphStorage().add(
          name, std::make_unique<RenderGraph::ImageViewHolder>(views, [&graph]() { return graph.getFrameInFlight(); }));
      return views;
    };
    const auto inputs = addImages("Input", VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    addImages("Color", VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    RenderGraph::ResourceUploader uploader(allocator, device);
    const std::vector<std::byte> pixels(4 * 4 * 4, std::byte{42});
    const std::array<VkDeviceSize, 1> offsets{0};
    uploader.upload(inputs[0]->getImage(), pixels, offsets);
    const auto ticket = uploader.submit();
    auto& sample = graph.createPassGraphic("Sample");
    sample.addTextureInput("Input");
    sample.addColorTarget("Color");
    auto& keep = graph.createPassGraphic("Keep");
    keep.addColorTarget("Input", RenderGraph::RenderTargetLoadAction::KEEP);
    graph.calculate(std::span(&ticket, 1));
    graph.calculate();
    EXPECT_EQ(inputs[0]->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_EQ(inputs[1]->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int frame = 0; frame < framesInFlight * 2; ++frame) ASSERT_FALSE(graph.render());
      ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
      graph.calculate();
    }
    for (const auto& input : inputs) EXPECT_EQ(input->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);

    RenderGraph::Buffer readback(pixels.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                 allocator, device);
    RenderGraph::CommandPool pool(RenderGraph::QueueType::GRAPHICS, device);
    RenderGraph::CommandBuffer commandBuffer(pool, device);
    commandBuffer.beginCommands();
    RenderGraph::Barrier beforeCopy;
    beforeCopy.addImage(inputs[0]->getImage().getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    beforeCopy.execute(commandBuffer.getCommandBuffer());
    const VkBufferImageCopy copy{.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {4, 4, 1}};
    vkCmdCopyImageToBuffer(commandBuffer.getCommandBuffer(), inputs[0]->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           readback.getBuffer(), 1, &copy);
    RenderGraph::Barrier afterCopy;
    afterCopy.addBuffer(&readback, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    afterCopy.execute(commandBuffer.getCommandBuffer());
    commandBuffer.endCommands();
    const VkCommandBufferSubmitInfo commandBufferInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                      .commandBuffer = commandBuffer.getCommandBuffer()};
    const VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                   .commandBufferInfoCount = 1,
                                   .pCommandBufferInfos = &commandBufferInfo};
    ASSERT_EQ(vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GRAPHICS), 1, &submitInfo, nullptr), VK_SUCCESS);
    ASSERT_EQ(vkQueueWaitIdle(device.getQueue(RenderGraph::QueueType::GRAPHICS)), VK_SUCCESS);
    ASSERT_EQ(vmaInvalidateAllocation(allocator.getAllocator(), readback.getAllocation(), 0, VK_WHOLE_SIZE),
              VK_SUCCESS);
    const auto* result = static_cast<const std::byte*>(readback.getAllocationInfo().pMappedData);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(std::vector<std::byte>(result, result + pixels.size()), pixels);
  }
  const auto errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, ImageUploadAfterCalculationPreservesContents) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("LateImageUploadValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  {
    RenderGraph::Device device(instance);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(1, 1, device, tracer);
    graph.initialize();
    auto image = std::make_unique<RenderGraph::Image>(allocator, device);
    image->createImage(
        VK_FORMAT_R8G8B8A8_UNORM, {4, 4}, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    auto view = std::make_shared<RenderGraph::ImageView>(std::move(image), device);
    view->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    graph.getGraphStorage().add("Input",
                                std::make_unique<RenderGraph::ImageViewHolder>(std::vector{view}, [] { return 0; }));
    graph.createPassGraphic("KeepUploadedImage").addColorTarget("Input", RenderGraph::RenderTargetLoadAction::KEEP);
    graph.calculate();
    EXPECT_EQ(view->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);

    RenderGraph::ResourceUploader uploader(allocator, device);
    const std::vector<std::byte> pixels(4 * 4 * 4, std::byte{42});
    const std::array<VkDeviceSize, 1> offsets{0};
    uploader.upload(view->getImage(), pixels, offsets);
    const auto ticket = uploader.submit();
    EXPECT_EQ(view->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);

    RenderGraph::CommandPool pool(RenderGraph::QueueType::GRAPHICS, device);
    RenderGraph::CommandBuffer commands(pool, device);
    const auto submitCommands = [&](std::span<const VkSemaphoreSubmitInfo> waits = {}) {
      commands.endCommands();
      const VkCommandBufferSubmitInfo commandInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                  .commandBuffer = commands.getCommandBuffer()};
      const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                 .waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size()),
                                 .pWaitSemaphoreInfos = waits.data(),
                                 .commandBufferInfoCount = 1,
                                 .pCommandBufferInfos = &commandInfo};
      return vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GRAPHICS), 1, &submit, nullptr);
    };
    // Synchronize the upload externally so the graph's original plan is used without recalculation.
    commands.beginCommands();
    const VkSemaphoreSubmitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                     .semaphore = ticket.semaphore,
                                     .value = ticket.value,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    ASSERT_EQ(submitCommands(std::span(&wait, 1)), VK_SUCCESS);
    ASSERT_EQ(vkQueueWaitIdle(device.getQueue(RenderGraph::QueueType::GRAPHICS)), VK_SUCCESS);

    {
      GraphCommandRecorder recorder;
      ASSERT_FALSE(graph.render());
      EXPECT_TRUE(recorder.imageBarriers.empty());
    }
    ASSERT_EQ(vkQueueWaitIdle(device.getQueue(RenderGraph::QueueType::GRAPHICS)), VK_SUCCESS);

    RenderGraph::Buffer readback(pixels.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                 allocator, device);
    commands.beginCommands();
    RenderGraph::Barrier beforeCopy;
    beforeCopy.addImage(view->getImage().getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                        VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    beforeCopy.execute(commands.getCommandBuffer());
    const VkBufferImageCopy copy{.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {4, 4, 1}};
    vkCmdCopyImageToBuffer(commands.getCommandBuffer(), view->getImage().getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           readback.getBuffer(), 1, &copy);
    RenderGraph::Barrier afterCopy;
    afterCopy.addBuffer(&readback, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    afterCopy.execute(commands.getCommandBuffer());
    ASSERT_EQ(submitCommands(), VK_SUCCESS);
    ASSERT_EQ(vkQueueWaitIdle(device.getQueue(RenderGraph::QueueType::GRAPHICS)), VK_SUCCESS);
    ASSERT_EQ(vmaInvalidateAllocation(allocator.getAllocator(), readback.getAllocation(), 0, VK_WHOLE_SIZE),
              VK_SUCCESS);
    const auto* result = static_cast<const std::byte*>(readback.getAllocationInfo().pMappedData);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(std::vector<std::byte>(result, result + pixels.size()), pixels);
    EXPECT_EQ(view->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
  }
  EXPECT_TRUE(validationErrors.getErrors().empty()) << joinErrors(validationErrors.getErrors());
}

TEST(ValidationTest, SharedImagesSurviveResizeAcrossFrames) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("SharedImageResizeValidationTest", true);
  if (!instance.isDebug()) {
    GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  }
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);

  {
    constexpr int framesInFlight = 2;
    const glm::ivec2 resolution(64, 64);
    RenderGraph::Window window(resolution);
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    const auto swapchainResolution = swapchain.getImage(0).getResolution();
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.setSwapchain(swapchain);
    graph.initialize();
    graph.getGraphStorage().add(
        "Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(
                         swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); }));

    auto addImages = [&](std::string_view name, glm::ivec2 size, RenderGraph::ImageResizePolicy policy) {
      std::vector<std::shared_ptr<RenderGraph::ImageView>> views;
      for (int frame = 0; frame < framesInFlight; ++frame) {
        auto image = std::make_unique<RenderGraph::Image>(allocator, device);
        image->createImage(VK_FORMAT_R8G8B8A8_UNORM, size, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_STORAGE_BIT);
        auto view = std::make_shared<RenderGraph::ImageView>(std::move(image), device);
        view->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
        views.push_back(std::move(view));
      }
      graph.getGraphStorage().add(
          name, std::make_unique<RenderGraph::ImageViewHolder>(views, [&graph]() { return graph.getFrameInFlight(); }),
          policy);
      return views;
    };
    const auto resized = addImages("Resized", {4, 4}, RenderGraph::ImageResizePolicy::SWAPCHAIN);
    const auto fixed = addImages("Fixed", {8, 8}, RenderGraph::ImageResizePolicy::FIXED);
    const auto sameSize = addImages("SameSize", swapchainResolution, RenderGraph::ImageResizePolicy::SWAPCHAIN);
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    for (int frame = 0; frame < framesInFlight; ++frame) {
      buffers.push_back(
          std::make_unique<RenderGraph::Buffer>(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, allocator, device));
    }
    graph.getGraphStorage().add("State", buffers);
    auto& compute = graph.createPassCompute("Compute", true);
    auto& graphics = graph.createPassCompute("ReadOnGraphics", false);
    compute.addStorageBufferInput("State");
    compute.addStorageBufferOutput("State");
    graphics.addStorageBufferInput("State");
    for (const auto* name : {"Resized", "Fixed", "SameSize"}) {
      compute.addStorageTextureInput(name);
      compute.addStorageTextureOutput(name);
      graphics.addStorageTextureInput(name);
    }
    graph.calculate();
    for (int frame = 0; frame < framesInFlight + 1; ++frame) ASSERT_FALSE(graph.render());
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);

    const VkImage fixedHandle = fixed.front()->getImage().getImage();
    const VkImage sameSizeHandle = sameSize.front()->getImage().getImage();
    const VkImage oldResizedHandle = resized.front()->getImage().getImage();
    for (int cycle = 0; cycle < 2; ++cycle) {
      ASSERT_NO_THROW(graph.reset(swapchainResolution));
      if (cycle == 0) RecordProperty("reusedImageHandle", oldResizedHandle == resized.front()->getImage().getImage());
      EXPECT_EQ(fixed.front()->getImage().getImage(), fixedHandle);
      EXPECT_EQ(sameSize.front()->getImage().getImage(), sameSizeHandle);
      EXPECT_EQ(resized.front()->getImage().getResolution(), swapchainResolution);
      if (cycle == 0) EXPECT_EQ(resized.front()->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
      for (int frame = 0; frame < framesInFlight * 2; ++frame) ASSERT_FALSE(graph.render());
      EXPECT_EQ(resized.front()->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
      ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
    }
  }
  const auto errors = validationErrors.getErrors();
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
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, framesInFlight, device, tracer);
    graph.setSwapchain(swapchain);
    graph.initialize();

    std::vector<std::unique_ptr<RenderGraph::Buffer>> storageBuffers;
    storageBuffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      storageBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
          4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator,
          device));
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

    ASSERT_NO_THROW(graph.reset(resolution));

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

TEST(ValidationTest, ResourceSharingIncludesAllDeviceQueueFamilies) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("ResourceSharingValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  for (bool present : {false, true}) {
    SCOPED_TRACE(present);
    RenderGraph::Window window({64, 64});
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(instance);
    if (present) device.setSurface(surface);
    device.initialize();
    std::vector<uint32_t> expected;
    for (auto type :
         {RenderGraph::QueueType::GRAPHICS, RenderGraph::QueueType::COMPUTE, RenderGraph::QueueType::TRANSFER}) {
      expected.push_back(static_cast<uint32_t>(device.getQueueIndex(type)));
    }
    if (present) expected.push_back(static_cast<uint32_t>(device.getQueueIndex(RenderGraph::QueueType::PRESENT)));
    std::ranges::sort(expected);
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(device.getResourceQueueFamilies(), expected);
    EXPECT_EQ(device.getResourceSharingMode(),
              expected.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE);
    ResourceSharingRecorder recorder(device);
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Buffer buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, allocator, device);
    RenderGraph::Image image(allocator, device);
    image.createImage(VK_FORMAT_R8G8B8A8_UNORM, {4, 4}, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_STORAGE_BIT);
    EXPECT_GT(recorder.buffers, 0u);
    EXPECT_GT(recorder.images, 0u);
    if (present) {
      RenderGraph::Swapchain swapchain({64, 64}, allocator, device);
      swapchain.initialize();
      EXPECT_GE(swapchain.getImageCount(), 2);
      swapchain.reset({96, 64});
      EXPECT_EQ(device.getQueueIndex(RenderGraph::QueueType::PRESENT),
                device.getQueueIndex(RenderGraph::QueueType::GRAPHICS));
    } else {
      EXPECT_THROW((RenderGraph::Swapchain({64, 64}, allocator, device)), std::runtime_error);
    }
  }
  EXPECT_TRUE(validationErrors.getErrors().empty()) << joinErrors(validationErrors.getErrors());
}

TEST(ValidationTest, SwapchainLayoutsFollowAcquiredImagesAcrossRecalculationAndReset) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("SwapchainLayoutValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  for (bool useSwapchain : {false, true}) {
    SCOPED_TRACE(useSwapchain);
    const glm::ivec2 resolution(64, 64);
    RenderGraph::Window window(resolution);
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    device.initialize();
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device);
    swapchain.initialize();
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, 2, device, tracer);
    graph.setSwapchain(swapchain);
    graph.initialize();
    graph.getGraphStorage().add("Swapchain",
                                std::make_unique<RenderGraph::ImageViewHolder>(
                                    swapchain.getImageViews(), [&swapchain] { return swapchain.getSwapchainIndex(); }));
    graph.createPassCompute("UnrelatedFirstPass", false);
    RenderGraph::GraphPass* firstUse = nullptr;
    if (useSwapchain) {
      auto& draw = graph.createPassGraphic("Draw");
      draw.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::KEEP);
      firstUse = &draw;
    }
    auto& tail = graph.createPassCompute("UnrelatedFinalPass", false);
    if (firstUse == nullptr) firstUse = &tail;

    for (int cycle = 0; cycle < 2; ++cycle) {
      if (cycle != 0) ASSERT_NO_THROW(graph.reset({96, 64}));
      graph.calculate();
      graph.calculate();
      std::vector<VkImageLayout> layouts(swapchain.getImageCount(), VK_IMAGE_LAYOUT_UNDEFINED);
      for (int frame = 0; frame < swapchain.getImageCount() + 4; ++frame) {
        SCOPED_TRACE(frame);
        const auto firstCommand = firstUse->getCommandBuffers()[graph.getFrameInFlight()]->getCommandBuffer();
        const auto finalCommand = tail.getCommandBuffers()[graph.getFrameInFlight()]->getCommandBuffer();
        GraphCommandRecorder recorder;
        ASSERT_FALSE(graph.render());
        const auto imageIndex = swapchain.getSwapchainIndex();
        const auto& image = swapchain.getImage(imageIndex);
        ASSERT_EQ(recorder.imageBarriers.size(), 2);
        const auto& [acquireCommand, acquire] = recorder.imageBarriers.front();
        const auto& [presentCommand, present] = recorder.imageBarriers.back();
        EXPECT_EQ(acquireCommand, firstCommand);
        EXPECT_EQ(acquire.image, image.getImage());
        EXPECT_EQ(acquire.oldLayout, layouts[imageIndex]);
        EXPECT_EQ(acquire.newLayout, VK_IMAGE_LAYOUT_GENERAL);
        EXPECT_EQ(presentCommand, finalCommand);
        EXPECT_EQ(present.image, image.getImage());
        EXPECT_EQ(present.oldLayout, VK_IMAGE_LAYOUT_GENERAL);
        EXPECT_EQ(present.newLayout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        EXPECT_EQ(present.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
        EXPECT_EQ(present.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
        EXPECT_EQ(image.getImageLayout(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        ASSERT_EQ(recorder.submissionQueues.size(), 1);
        EXPECT_EQ(recorder.submissionQueues.front(), device.getQueue(RenderGraph::QueueType::GRAPHICS));
        ASSERT_EQ(recorder.submissionBuffers.front().size(), useSwapchain ? 3 : 2);
        EXPECT_EQ(recorder.submissionBuffers.front().back(), finalCommand);
        layouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        // Rebuilding the plan must not change the acquired-image history.
        ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
        graph.calculate();
        EXPECT_EQ(image.getImageLayout(), layouts[imageIndex]);
      }
    }
  }
  const auto errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

TEST(ValidationTest, SwapchainComputeUseHasNoOwnershipBarriers) {
  ValidationErrorCollector validationErrors;
  RenderGraph::Instance instance("SwapchainComputeSharingValidationTest", true);
  if (!instance.isDebug()) GTEST_SKIP() << "Vulkan validation layers or VK_EXT_debug_utils are unavailable";
  ValidationMessenger validationMessenger(instance.getInstance().instance, validationErrors);
  for (bool graphicsFirst : {false, true}) {
    SCOPED_TRACE(graphicsFirst);
    const glm::ivec2 resolution(64, 64);
    RenderGraph::Window window(resolution);
    window.initialize();
    RenderGraph::Surface surface(window, instance);
    RenderGraph::Device device(instance);
    device.setSurface(surface);
    device.initialize();
    if (!supportsStorageSwapchain(device, surface)) GTEST_SKIP() << "Storage swapchain images are unavailable";
    RenderGraph::MemoryAllocator allocator(device, instance);
    RenderGraph::Swapchain swapchain(resolution, allocator, device,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    swapchain.initialize();
    RenderGraph::Tracer tracer(device);
    RenderGraph::Graph graph(2, 2, device, tracer);
    graph.setSwapchain(swapchain);
    graph.initialize();
    graph.getGraphStorage().add("Swapchain",
                                std::make_unique<RenderGraph::ImageViewHolder>(
                                    swapchain.getImageViews(), [&swapchain] { return swapchain.getSwapchainIndex(); }));
    RenderGraph::GraphPass* firstUse = nullptr;
    if (graphicsFirst) {
      auto& draw = graph.createPassGraphic("Draw");
      draw.addColorTarget("Swapchain");
      firstUse = &draw;
    }
    auto& compute = graph.createPassCompute("Compute", true);
    if (graphicsFirst) {
      compute.addStorageTextureInput("Swapchain");
    } else {
      compute.addStorageTextureInput("Swapchain");
      compute.addStorageTextureOutput("Swapchain");
      firstUse = &compute;
    }
    // End on compute after graphics, and on graphics after compute, without extra submissions.
    auto& tail = graph.createPassCompute("UnrelatedFinalPass", graphicsFirst);
    graph.calculate();
    std::vector<VkImageLayout> layouts(swapchain.getImageCount(), VK_IMAGE_LAYOUT_UNDEFINED);
    for (int frame = 0; frame < swapchain.getImageCount() + 4; ++frame) {
      SCOPED_TRACE(frame);
      const auto firstCommand = firstUse->getCommandBuffers()[graph.getFrameInFlight()]->getCommandBuffer();
      const auto finalCommand = tail.getCommandBuffers()[graph.getFrameInFlight()]->getCommandBuffer();
      GraphCommandRecorder recorder;
      ASSERT_FALSE(graph.render());
      const auto imageIndex = swapchain.getSwapchainIndex();
      const auto& image = swapchain.getImage(imageIndex);
      ASSERT_EQ(recorder.imageBarriers.size(), 2);
      const auto& [acquireCommand, acquire] = recorder.imageBarriers.front();
      const auto& [presentCommand, present] = recorder.imageBarriers.back();
      EXPECT_EQ(acquireCommand, firstCommand);
      EXPECT_EQ(acquire.oldLayout, layouts[imageIndex]);
      EXPECT_EQ(acquire.newLayout, VK_IMAGE_LAYOUT_GENERAL);
      EXPECT_EQ(presentCommand, finalCommand);
      EXPECT_EQ(present.oldLayout, VK_IMAGE_LAYOUT_GENERAL);
      EXPECT_EQ(present.newLayout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
      for (const auto& [command, barrier] : recorder.imageBarriers) {
        EXPECT_EQ(barrier.image, image.getImage());
        EXPECT_EQ(barrier.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
        EXPECT_EQ(barrier.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
      }
      // QueueType changes retain two submissions even if the actual queues alias.
      ASSERT_EQ(recorder.submissionQueues.size(), 2);
      EXPECT_EQ(recorder.submissionQueues.back(),
                device.getQueue(graphicsFirst ? RenderGraph::QueueType::COMPUTE : RenderGraph::QueueType::GRAPHICS));
      EXPECT_EQ(recorder.submissionBuffers.back().back(), finalCommand);
      EXPECT_EQ(image.getImageLayout(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
      layouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }
    ASSERT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
  }
  const auto errors = validationErrors.getErrors();
  EXPECT_TRUE(errors.empty()) << joinErrors(errors);
}

INSTANTIATE_TEST_SUITE_P(QueueModes,
                         ValidationScenarioTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<ValidationScenarioTest::ParamType>& info) {
                           return info.param ? "SeparateComputeQueue" : "SingleQueue";
                         });
