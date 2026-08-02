#define VK_NO_PROTOTYPES
#include <volk.h>

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

import Allocator;
import Buffer;
import Device;
import Graph;
import Instance;
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
    asyncPass.addStorageTextureInput("Color");
    asyncPass.addStorageTextureOutput("Color");
    asyncPass.registerGraphElement(graphElement);

    auto& compositePass = graph.createPassGraphic("Composite");
    compositePass.addTextureInput("Color");
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

INSTANTIATE_TEST_SUITE_P(
    QueueModes, ValidationScenarioTest, testing::Values(false, true),
    [](const testing::TestParamInfo<ValidationScenarioTest::ParamType>& info) {
      return info.param ? "SeparateComputeQueue" : "SingleQueue";
    });
