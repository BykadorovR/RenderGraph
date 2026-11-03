#define VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>
#include <volk.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

import Instance;
import Window;
import Surface;
import Allocator;
import Buffer;
import Swapchain;
import Device;
import Graph;
import Instrumentation;
import Texture;
import CommandPool;
import Command;
import Sync;
import glm;

class GraphElementMock : public RenderGraph::GraphElement {
 private:
  std::atomic<int> _drawCount = 0;
  std::atomic<int> _updateCount = 0;
  std::atomic<int> _resetCount = 0;

 public:
  void draw(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override { _drawCount++; }
  void update(int currentFrame, const RenderGraph::CommandBuffer& commandBuffer) override { _updateCount++; }
  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) override { _resetCount++; }

  int getDrawCount() const noexcept { return _drawCount; }
  int getUpdateCount() const noexcept { return _updateCount; }
  int getResetCount() const noexcept { return _resetCount; }
};

TEST(ScenarioTest, ComputePassQueueTypes) {
  RenderGraph::Instance instance("ComputeQueueTest", false);
  RenderGraph::Device device(instance);
  device.initialize();
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(1, 1, device, tracer);

  for (const auto invalidQueue : {RenderGraph::QueueType::TRANSFER, static_cast<RenderGraph::QueueType>(-1)}) {
    SCOPED_TRACE(static_cast<int>(invalidQueue));
    EXPECT_THROW(graph.createPassCompute("Compute", invalidQueue), std::invalid_argument);
    EXPECT_EQ(graph.getPassCompute("Compute"), nullptr);
    EXPECT_THROW(RenderGraph::GraphPassCompute("Compute", 1, invalidQueue, graph.getGraphStorage(), device),
                 std::invalid_argument);
  }

  const RenderGraph::GraphPass& computePass = graph.createPassCompute("Compute", RenderGraph::QueueType::COMPUTE);
  const RenderGraph::GraphPass& generalComputePass = graph.createPassCompute("ComputeOnGeneral",
                                                                             RenderGraph::QueueType::GENERAL);
  const RenderGraph::GraphPass& graphicsPass = graph.createPassGraphic("Draw");
  EXPECT_EQ(computePass.getQueueType(), RenderGraph::QueueType::COMPUTE);
  EXPECT_EQ(generalComputePass.getQueueType(), RenderGraph::QueueType::GENERAL);
  EXPECT_EQ(graphicsPass.getQueueType(), RenderGraph::QueueType::GENERAL);
}

TEST(ScenarioTest, GraphOneQueue) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(RenderGraph::QueueType::GENERAL, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.setSwapchain(swapchain);
  graph.initialize();

  EXPECT_EQ(graph.getFrameInFlight(), 0);

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  std::vector<std::shared_ptr<RenderGraph::ImageView>> positionImageViews;
  for (int i = 0; i < framesInFlight; i++) {
    auto positionImage = std::make_unique<RenderGraph::Image>(allocator, device);
    positionImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto positionImageView = std::make_shared<RenderGraph::ImageView>(std::move(positionImage), device);
    positionImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    positionImageViews.push_back(positionImageView);
  }

  std::unique_ptr<RenderGraph::ImageViewHolder> positionHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      positionImageViews, [&]() { return graph.getFrameInFlight(); });
  graph.getGraphStorage().add("Target", std::move(positionHolder), RenderGraph::ImageResizePolicy::SWAPCHAIN);

  EXPECT_GE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews().size(), 2);

  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::CLEAR);
  renderPass.addColorTarget("Target", RenderGraph::RenderTargetLoadAction::CLEAR);
  auto elementMock = std::make_shared<GraphElementMock>();
  renderPass.registerGraphElement(elementMock);

  EXPECT_EQ(graph.getPassGraphic("Render"), &renderPass);
  EXPECT_EQ(graph.getPassCompute("Render"), nullptr);

  EXPECT_THROW(graph.createPassGraphic("Render"), std::logic_error);
  EXPECT_THROW(graph.createPassCompute("Render", RenderGraph::QueueType::GENERAL), std::logic_error);

  EXPECT_EQ(renderPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(renderPass.getCommandBuffers()[i], renderPass.getCommandBuffers()[i - 1]);
  }

  auto& pipelineGraphic = renderPass.getPipelineGraphic(graph.getGraphStorage());
  // it should be modifiable
  pipelineGraphic.setDepthTest(true);
  pipelineGraphic.setDepthWrite(true);
  pipelineGraphic.setTesselation(4);
  pipelineGraphic.setTopology(VK_PRIMITIVE_TOPOLOGY_PATCH_LIST);
  pipelineGraphic.setCullMode(VK_CULL_MODE_BACK_BIT);

  EXPECT_EQ(renderPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 2);
  EXPECT_EQ(renderPass.getDepthTarget(), std::nullopt);
  EXPECT_TRUE(renderPass.getStorageBufferInputs().empty());
  EXPECT_TRUE(renderPass.getIndirectBufferInputs().empty());

  auto& postprocessingPass = graph.createPassCompute("Postprocessing", RenderGraph::QueueType::GENERAL);
  postprocessingPass.registerGraphElement(elementMock);
  postprocessingPass.addStorageTextureInput("Swapchain");
  postprocessingPass.addStorageTextureOutput("Swapchain");

  EXPECT_EQ(graph.getPassCompute("Postprocessing"), &postprocessingPass);
  EXPECT_EQ(graph.getPassGraphic("Postprocessing"), nullptr);

  EXPECT_EQ(postprocessingPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(postprocessingPass.getCommandBuffers()[i], postprocessingPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(postprocessingPass.getStorageTextureInputs().size(), 1);
  EXPECT_EQ(postprocessingPass.getStorageTextureOutputs().size(), 1);
  EXPECT_EQ(postprocessingPass.getStorageBufferInputs().size(), 0);
  EXPECT_EQ(postprocessingPass.getStorageBufferOutputs().size(), 0);

  auto& guiPass = graph.createPassGraphic("GUI");
  guiPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::KEEP);
  guiPass.registerGraphElement(elementMock);

  EXPECT_EQ(guiPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(guiPass.getCommandBuffers()[i], guiPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(guiPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 1);

  graph.calculate();

  EXPECT_TRUE(graph._resources.at(&renderPass)
                  .contains("Swapchain", RenderGraph::Graph::Resource::Type::IMAGE,
                            RenderGraph::Graph::Resource::Operation::WRITE));
  EXPECT_FALSE(graph._resources.at(&renderPass)
                   .contains("Swapchain", RenderGraph::Graph::Resource::Type::IMAGE,
                             RenderGraph::Graph::Resource::Operation::READ));
  EXPECT_TRUE(graph._resources.at(&guiPass).contains("Swapchain", RenderGraph::Graph::Resource::Type::IMAGE,
                                                     RenderGraph::Graph::Resource::Operation::READ));
  EXPECT_TRUE(graph._resources.at(&guiPass).contains("Swapchain", RenderGraph::Graph::Resource::Type::IMAGE,
                                                     RenderGraph::Graph::Resource::Operation::WRITE));

  RenderGraph::Graph offscreenGraph(1, framesInFlight, device, tracer);
  offscreenGraph.setSwapchain(swapchain);
  offscreenGraph.initialize();
  offscreenGraph.getGraphStorage().add(
      "Offscreen", std::make_unique<RenderGraph::ImageViewHolder>(positionImageViews, []() { return 0; }));
  auto& offscreenPass = offscreenGraph.createPassGraphic("Offscreen");
  offscreenPass.addColorTarget("Offscreen", RenderGraph::RenderTargetLoadAction::CLEAR);
  offscreenGraph.calculate();

  // calculate() prepares usage descriptions; physical barriers are built for the selected frame.
  EXPECT_TRUE(offscreenGraph._physicalBarriers.at(&offscreenPass).getImageBarriers().empty());
  EXPECT_TRUE(graph._physicalBarriers.at(&renderPass).getImageBarriers().empty());
  EXPECT_TRUE(graph._queueTimelines.empty());
  commandBuffer[graph.getFrameInFlight()].endCommands();

  auto loadSemaphore = RenderGraph::Semaphore(VK_SEMAPHORE_TYPE_TIMELINE, device);
  uint64_t loadCounter = 1;
  auto semaphore = loadSemaphore.getSemaphore();
  VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = commandBuffer[graph.getFrameInFlight()].getCommandBuffer()};
  VkSemaphoreSubmitInfo signalSemaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                            .semaphore = semaphore,
                                            .value = loadCounter,
                                            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
  VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                           .commandBufferInfoCount = 1,
                           .pCommandBufferInfos = &commandBufferInfo,
                           .signalSemaphoreInfoCount = 1,
                           .pSignalSemaphoreInfos = &signalSemaphoreInfo};
  vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GENERAL), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

  EXPECT_EQ(positionImageViews.front()->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
  graph.render();
  EXPECT_EQ(positionImageViews.front()->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
  EXPECT_EQ(positionImageViews.back()->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_EQ(graph._swapchainFirstPass, &renderPass);
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 3);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), (2 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 6);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 9);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), (4 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 12);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  for (int i = 0; i < 100; i++) {
    graph.render();
    EXPECT_EQ(graph.getFrameInFlight(), ((4 + i + 1) % framesInFlight));
    EXPECT_EQ(elementMock->getDrawCount(), 3 * (i + 5));
    EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());
  }

  // wait device idle before destroying resources
  vkDeviceWaitIdle(device.getLogicalDevice());
}

TEST(ScenarioTest, GraphSeparateQueues) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(RenderGraph::QueueType::GENERAL, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.setSwapchain(swapchain);
  graph.initialize();

  EXPECT_EQ(graph.getFrameInFlight(), 0);

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  std::vector<std::shared_ptr<RenderGraph::ImageView>> positionImageViews;
  for (int i = 0; i < framesInFlight; i++) {
    auto positionImage = std::make_unique<RenderGraph::Image>(allocator, device);
    positionImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto positionImageView = std::make_shared<RenderGraph::ImageView>(std::move(positionImage), device);
    positionImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    positionImageViews.push_back(positionImageView);
  }

  std::unique_ptr<RenderGraph::ImageViewHolder> positionHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      positionImageViews, [&]() { return graph.getFrameInFlight(); });
  graph.getGraphStorage().add("Target", std::move(positionHolder), RenderGraph::ImageResizePolicy::SWAPCHAIN);

  EXPECT_GE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews().size(), 2);

  auto elementMock = std::make_shared<GraphElementMock>();
  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::CLEAR);
  renderPass.addColorTarget("Target", RenderGraph::RenderTargetLoadAction::CLEAR);
  renderPass.registerGraphElement(elementMock);

  EXPECT_EQ(renderPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(renderPass.getCommandBuffers()[i], renderPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(renderPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 2);
  EXPECT_EQ(renderPass.getDepthTarget(), std::nullopt);

  // compute queue
  auto& postprocessingPass = graph.createPassCompute("Postprocessing", RenderGraph::QueueType::COMPUTE);
  postprocessingPass.registerGraphElement(elementMock);
  postprocessingPass.addStorageTextureInput("Swapchain");
  postprocessingPass.addStorageTextureOutput("Swapchain");

  EXPECT_EQ(postprocessingPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(postprocessingPass.getCommandBuffers()[i], postprocessingPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(postprocessingPass.getStorageTextureInputs().size(), 1);
  EXPECT_EQ(postprocessingPass.getStorageTextureOutputs().size(), 1);
  EXPECT_EQ(postprocessingPass.getStorageBufferInputs().size(), 0);
  EXPECT_EQ(postprocessingPass.getStorageBufferOutputs().size(), 0);

  auto& guiPass = graph.createPassGraphic("GUI");
  guiPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::KEEP);
  guiPass.registerGraphElement(elementMock);

  EXPECT_EQ(guiPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(guiPass.getCommandBuffers()[i], guiPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(guiPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 1);

  graph.calculate();
  const bool separateQueues = device.getQueue(RenderGraph::QueueType::GENERAL) !=
                              device.getQueue(RenderGraph::QueueType::COMPUTE);
  EXPECT_EQ(graph._queueTimelines.size(), separateQueues ? 2 : 0);
  commandBuffer[graph.getFrameInFlight()].endCommands();

  auto loadSemaphore = RenderGraph::Semaphore(VK_SEMAPHORE_TYPE_TIMELINE, device);
  uint64_t loadCounter = 1;
  auto semaphore = loadSemaphore.getSemaphore();
  VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = commandBuffer[graph.getFrameInFlight()].getCommandBuffer()};
  VkSemaphoreSubmitInfo signalSemaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                            .semaphore = semaphore,
                                            .value = loadCounter,
                                            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
  VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                           .commandBufferInfoCount = 1,
                           .pCommandBufferInfos = &commandBufferInfo,
                           .signalSemaphoreInfoCount = 1,
                           .pSignalSemaphoreInfos = &signalSemaphoreInfo};
  vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GENERAL), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 3);

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), (2 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 6);

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 9);

  graph.render();
  EXPECT_EQ(graph.getFrameInFlight(), (4 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 12);

  for (int i = 0; i < 100; i++) {
    graph.render();
    EXPECT_EQ(graph.getFrameInFlight(), ((4 + i + 1) % framesInFlight));
    EXPECT_EQ(elementMock->getDrawCount(), 3 * (i + 5));
  }

  // wait device idle before destroying resources
  vkDeviceWaitIdle(device.getLogicalDevice());
}

TEST(ScenarioTest, BufferQueueChangesUseSemaphores) {
  glm::ivec2 resolution(1920, 1080);

  RenderGraph::Instance instance("TestApp", false);

  RenderGraph::Window window(resolution);
  window.initialize();

  RenderGraph::Surface surface(window, instance);

  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);

  constexpr int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);
  graph.initialize();

  auto addStorageBuffers = [&](std::string_view name) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                                              allocator, device));
    }

    graph.getGraphStorage().add(name, std::make_unique<RenderGraph::BufferHandler>(
                                          std::move(buffers), [&graph] { return graph.getFrameInFlight(); }));
  };

  addStorageBuffers("X");
  addStorageBuffers("Y");
  addStorageBuffers("Link");
  addStorageBuffers("Independent");

  /*
   * X, Y, Link are resources we want to pass between First, Middle, Last passes.
   * Passes flow is: First -> Middle -> Last
   *
   * graphics queue: First(writes: X, writes: Link) -> Middle(reads: Link, writes: Y)
   * compute queue: Last(reads: X, Y)
   *
   * X and Y cross queues using the Middle -> Last semaphore.
   * Link stays on the graphics queue and requires a memory barrier.
   * An unrelated compute pass between First and Middle must not make Middle wait.
   */

  auto& firstPass = graph.createPassCompute("First", RenderGraph::QueueType::GENERAL);
  firstPass.addStorageBufferOutput("X");
  firstPass.addStorageBufferOutput("Link");

  auto& independentPass = graph.createPassCompute("Independent", RenderGraph::QueueType::COMPUTE);
  independentPass.addStorageBufferOutput("Independent");

  auto& middlePass = graph.createPassCompute("Middle", RenderGraph::QueueType::GENERAL);
  middlePass.addStorageBufferInput("Link");
  middlePass.addStorageBufferOutput("Y");

  auto& lastPass = graph.createPassCompute("Last", RenderGraph::QueueType::COMPUTE);

  // Preserve the First -> Middle -> Last dependency traversal.
  lastPass.addStorageBufferInput("Y");
  lastPass.addStorageBufferInput("X");

  graph.calculate();

  ASSERT_EQ(graph._submissions.size(), 4);
  graph._preparePhysicalAccesses();
  EXPECT_TRUE(graph._submissions[1].waits.empty());
  EXPECT_TRUE(graph._submissions[2].waits.empty());
  // Link stays on the graphics queue and requires one barrier before Middle.
  const auto& middleBarriers = graph._physicalBarriers.at(&middlePass).getBufferBarriers();
  ASSERT_EQ(middleBarriers.size(), 1);
  EXPECT_TRUE(graph._physicalBarriers.at(&firstPass).getBufferBarriers().empty());
  const auto& barrier = middleBarriers.front();
  EXPECT_EQ(barrier.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
  EXPECT_EQ(barrier.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);

  if (device.getQueue(RenderGraph::QueueType::GENERAL) != device.getQueue(RenderGraph::QueueType::COMPUTE)) {
    EXPECT_TRUE(graph._physicalBarriers.at(&lastPass).getBufferBarriers().empty());
    const auto& waits = graph._submissions.back().waits;
    ASSERT_EQ(waits.size(), 1);
    EXPECT_EQ(waits.front().semaphore,
              graph._queueTimelines.at(device.getQueue(RenderGraph::QueueType::GENERAL)).semaphore->getSemaphore());
    EXPECT_EQ(waits.front().value, graph._submissions[2].value);
    EXPECT_LT(graph._submissions.front().value, graph._submissions[2].value);
  } else {
    EXPECT_EQ(graph._physicalBarriers.at(&lastPass).getBufferBarriers().size(), 2);
    EXPECT_TRUE(graph._submissions.back().waits.empty());
  }
}

TEST(ScenarioTest, BufferQueueChangesSurviveFrameSlotReuse) {
  RenderGraph::Instance instance("FrameSlotQueueTest", false);
  RenderGraph::Device device(instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Tracer tracer(device);
  constexpr int framesInFlight = 2;
  RenderGraph::Graph graph(1, framesInFlight, device, tracer);
  graph.initialize();
  std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
  for (int frame = 0; frame < framesInFlight; ++frame) {
    buffers.push_back(
        std::make_unique<RenderGraph::Buffer>(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, allocator, device));
  }
  graph.getGraphStorage().add("State", std::make_unique<RenderGraph::BufferHandler>(
                                           std::move(buffers), [&graph] { return graph.getFrameInFlight(); }));

  auto& computePass = graph.createPassCompute("UpdateState", RenderGraph::QueueType::COMPUTE);
  computePass.addStorageBufferInput("State");
  computePass.addStorageBufferOutput("State");
  // This compute pass uses the graphics queue.
  auto& graphicsPass = graph.createPassCompute("ReadStateOnGraphics", RenderGraph::QueueType::GENERAL);
  graphicsPass.addStorageBufferInput("State");
  graph.calculate();

  for (int frame = 0; frame < framesInFlight * 3; ++frame) {
    const auto buffer = graph.getGraphStorage().getBufferHandler("State").getBuffer().getBuffer();
    ASSERT_FALSE(graph.render());
    const auto& history = graph._bufferHistory.at(buffer);
    ASSERT_EQ(history.size(), 2);
    EXPECT_TRUE(history.front().writes);
    EXPECT_EQ(history.front().queue, device.getQueue(RenderGraph::QueueType::COMPUTE));
    EXPECT_FALSE(history.back().writes);
    EXPECT_EQ(history.back().queue, device.getQueue(RenderGraph::QueueType::GENERAL));
  }
  EXPECT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
}

TEST(ScenarioTest, SwapchainGraphCanEndOnCompute) {
  const glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(1, framesInFlight, device, tracer);
  graph.setSwapchain(swapchain);
  graph.initialize();

  std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
  buffers.reserve(framesInFlight);
  for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
    buffers.push_back(std::make_unique<RenderGraph::Buffer>(1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                                            allocator, device));
  }
  graph.getGraphStorage().add("Data", std::make_unique<RenderGraph::BufferHandler>(
                                          std::move(buffers), [&graph] { return graph.getFrameInFlight(); }));

  auto& computePass = graph.createPassCompute("Compute", RenderGraph::QueueType::COMPUTE);
  computePass.addStorageBufferOutput("Data");

  graph.calculate();
  for (int frame = 0; frame < framesInFlight * 3; ++frame) ASSERT_FALSE(graph.render());
  EXPECT_EQ(vkDeviceWaitIdle(device.getLogicalDevice()), VK_SUCCESS);
}

TEST(ScenarioTest, GraphicsPassBufferInputs) {
  const glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(2, framesInFlight, device, tracer);
  graph.setSwapchain(swapchain);
  graph.initialize();
  graph.getGraphStorage().add("Swapchain",
                              std::make_unique<RenderGraph::ImageViewHolder>(
                                  swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); }));

  auto addBuffers = [&](std::string_view name, VkBufferUsageFlags usage) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(
          1024, usage, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator, device));
    }
    graph.getGraphStorage().add(name, std::make_unique<RenderGraph::BufferHandler>(
                                          std::move(buffers), [&graph] { return graph.getFrameInFlight(); }));
  };

  addBuffers("VisibleObjects", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  addBuffers("IndirectCommands", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  addBuffers("Vertices", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  addBuffers("Indices", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

  auto& cullPass = graph.createPassCompute("Cull", RenderGraph::QueueType::GENERAL);
  cullPass.addStorageBufferOutput("VisibleObjects");
  cullPass.addStorageBufferOutput("IndirectCommands");
  cullPass.addStorageBufferOutput("Vertices");
  cullPass.addStorageBufferOutput("Indices");

  auto& drawPass = graph.createPassGraphic("Draw");
  drawPass.addStorageBufferInput("VisibleObjects");
  drawPass.addIndirectBufferInput("IndirectCommands");
  drawPass.addVertexBufferInput("Vertices");
  drawPass.addIndexBufferInput("Indices");
  drawPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::DONT_CARE);

  graph.calculate();

  ASSERT_EQ(drawPass.getStorageBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getStorageBufferInputs().front(), "VisibleObjects");
  ASSERT_EQ(drawPass.getIndirectBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getIndirectBufferInputs().front(), "IndirectCommands");
  ASSERT_EQ(drawPass.getVertexBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getVertexBufferInputs().front(), "Vertices");
  ASSERT_EQ(drawPass.getIndexBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getIndexBufferInputs().front(), "Indices");
  EXPECT_FALSE(graph._resources.at(&drawPass).contains("Swapchain", RenderGraph::Graph::Resource::Type::IMAGE,
                                                       RenderGraph::Graph::Resource::Operation::READ));

  ASSERT_EQ(graph._passesOrdered.size(), 2);
  EXPECT_EQ(graph._passesOrdered.front(), &cullPass);
  EXPECT_EQ(graph._passesOrdered.back(), &drawPass);

  graph._preparePhysicalAccesses();
  const auto& barriers = graph._physicalBarriers.at(&drawPass).getBufferBarriers();
  ASSERT_EQ(barriers.size(), 4);

  const VkBuffer visibleObjectsBuffer =
      graph.getGraphStorage().getBufferHandler("VisibleObjects").getBuffer().getBuffer();
  const VkBuffer indirectCommandsBuffer =
      graph.getGraphStorage().getBufferHandler("IndirectCommands").getBuffer().getBuffer();
  const VkBuffer vertexBuffer = graph.getGraphStorage().getBufferHandler("Vertices").getBuffer().getBuffer();
  const VkBuffer indexBuffer = graph.getGraphStorage().getBufferHandler("Indices").getBuffer().getBuffer();

  bool foundStorageBarrier = false;
  bool foundIndirectBarrier = false;
  bool foundVertexBarrier = false;
  bool foundIndexBarrier = false;
  for (const auto& bufferBarrier : barriers) {
    EXPECT_EQ(bufferBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(bufferBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);

    if (bufferBarrier.buffer == visibleObjectsBuffer) {
      foundStorageBarrier = true;
      EXPECT_EQ(bufferBarrier.dstStageMask, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
      EXPECT_EQ(bufferBarrier.dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);
    } else if (bufferBarrier.buffer == indirectCommandsBuffer) {
      foundIndirectBarrier = true;
      EXPECT_EQ(bufferBarrier.dstStageMask, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
      EXPECT_EQ(bufferBarrier.dstAccessMask, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    } else if (bufferBarrier.buffer == vertexBuffer) {
      foundVertexBarrier = true;
      EXPECT_EQ(bufferBarrier.dstStageMask, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT);
      EXPECT_EQ(bufferBarrier.dstAccessMask, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
    } else if (bufferBarrier.buffer == indexBuffer) {
      foundIndexBarrier = true;
      EXPECT_EQ(bufferBarrier.dstStageMask, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT);
      EXPECT_EQ(bufferBarrier.dstAccessMask, VK_ACCESS_2_INDEX_READ_BIT);
    } else {
      FAIL() << "Barrier references an unexpected buffer";
    }
  }

  EXPECT_TRUE(foundStorageBarrier);
  EXPECT_TRUE(foundIndirectBarrier);
  EXPECT_TRUE(foundVertexBarrier);
  EXPECT_TRUE(foundIndexBarrier);
}

TEST(ScenarioTest, ComputePassIndirectBufferInput) {
  const glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 3;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(2, framesInFlight, device, tracer);
  graph.setSwapchain(swapchain);
  graph.initialize();

  std::vector<std::unique_ptr<RenderGraph::Buffer>> indirectBuffers;
  for (int bufferIndex = 0; bufferIndex < 2; ++bufferIndex) {
    indirectBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
        sizeof(VkDispatchIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator, device));
  }
  int selectedBuffer = 1;
  graph.getGraphStorage().add(
      "DispatchCommands", std::make_unique<RenderGraph::BufferHandler>(std::move(indirectBuffers),
                                                                       [&selectedBuffer] { return selectedBuffer; }));

  auto& generatePass = graph.createPassCompute("GenerateDispatch", RenderGraph::QueueType::GENERAL);
  generatePass.addStorageBufferOutput("DispatchCommands");

  auto& dispatchPass = graph.createPassCompute("DispatchIndirect", RenderGraph::QueueType::GENERAL);
  dispatchPass.addIndirectBufferInput("DispatchCommands");

  auto& inspectPass = graph.createPassCompute("InspectDispatch", RenderGraph::QueueType::GENERAL);
  inspectPass.addStorageBufferInput("DispatchCommands");

  auto& rewritePass = graph.createPassCompute("RewriteDispatch", RenderGraph::QueueType::GENERAL);
  rewritePass.addStorageBufferInput("DispatchCommands");
  rewritePass.addStorageBufferOutput("DispatchCommands");

  auto& inspectRewritePass = graph.createPassCompute("InspectRewrittenDispatch", RenderGraph::QueueType::GENERAL);
  inspectRewritePass.addStorageBufferInput("DispatchCommands");

  graph.calculate();

  ASSERT_EQ(dispatchPass.getIndirectBufferInputs().size(), 1);
  EXPECT_EQ(dispatchPass.getIndirectBufferInputs().front(), "DispatchCommands");

  ASSERT_EQ(graph._passesOrdered.size(), 5);
  EXPECT_EQ(graph._passesOrdered.front(), &generatePass);
  EXPECT_EQ(graph._passesOrdered.back(), &inspectRewritePass);

  graph._preparePhysicalAccesses();
  const auto& barriers = graph._physicalBarriers.at(&dispatchPass).getBufferBarriers();
  ASSERT_EQ(barriers.size(), 1);

  const auto& bufferBarrier = barriers.front();
  const auto& bufferHandler = graph.getGraphStorage().getBufferHandler("DispatchCommands");
  EXPECT_EQ(bufferBarrier.buffer, bufferHandler.getBuffer().getBuffer());
  EXPECT_NE(bufferBarrier.buffer, bufferHandler.getBuffers().front()->getBuffer());
  EXPECT_EQ(bufferBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(bufferBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);
  EXPECT_EQ(bufferBarrier.dstStageMask, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
  EXPECT_EQ(bufferBarrier.dstAccessMask, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

  // The shader reader still depends on GenerateDispatch, not on the intervening indirect reader.
  const auto& inspectBarriers = graph._physicalBarriers.at(&inspectPass).getBufferBarriers();
  ASSERT_EQ(inspectBarriers.size(), 1);
  const auto& inspectBarrier = inspectBarriers.front();
  EXPECT_EQ(inspectBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(inspectBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);
  EXPECT_EQ(inspectBarrier.dstStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(inspectBarrier.dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);

  // Rewriting must wait for both readers, even though only the shader reader used the buffer last.
  const auto& rewriteBarriers = graph._physicalBarriers.at(&rewritePass).getBufferBarriers();
  ASSERT_EQ(rewriteBarriers.size(), 1);
  const auto& rewriteBarrier = rewriteBarriers.front();
  EXPECT_EQ(rewriteBarrier.srcStageMask,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
  EXPECT_EQ(rewriteBarrier.srcAccessMask,
            VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
  EXPECT_EQ(rewriteBarrier.dstStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(rewriteBarrier.dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

  const auto& inspectRewriteBarriers = graph._physicalBarriers.at(&inspectRewritePass).getBufferBarriers();
  ASSERT_EQ(inspectRewriteBarriers.size(), 1);
  const auto& inspectRewriteBarrier = inspectRewriteBarriers.front();
  EXPECT_EQ(inspectRewriteBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(inspectRewriteBarrier.srcAccessMask, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  EXPECT_EQ(inspectRewriteBarrier.dstStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(inspectRewriteBarrier.dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);

  for (const int index : {0, 1, 0}) {
    selectedBuffer = index;
    EXPECT_EQ(&bufferHandler.getBuffer(), bufferHandler.getBuffers()[index]);
    graph._preparePhysicalAccesses();
    for (auto* pass : {&dispatchPass, &inspectPass, &rewritePass, &inspectRewritePass}) {
      const auto& barrier = graph._physicalBarriers.at(pass);
      ASSERT_EQ(barrier.getBufferBarriers().size(), 1);
      EXPECT_EQ(barrier.getBufferBarriers().front().buffer, bufferHandler.getBuffer().getBuffer());
    }
  }
}

TEST(ScenarioTest, GraphReset) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  const glm::ivec2 windowResolution = window.getResolution();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(RenderGraph::QueueType::GENERAL, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.setSwapchain(swapchain);
  graph.initialize();

  auto swapchainOldImages = swapchain.getImageViews();

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  std::vector<std::shared_ptr<RenderGraph::ImageView>> positionImageViews;
  for (int i = 0; i < framesInFlight; i++) {
    auto positionImage = std::make_unique<RenderGraph::Image>(allocator, device);
    positionImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto positionImageView = std::make_shared<RenderGraph::ImageView>(std::move(positionImage), device);
    positionImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    positionImageViews.push_back(positionImageView);
  }

  std::unique_ptr<RenderGraph::ImageViewHolder> positionHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      positionImageViews, [&]() { return graph.getFrameInFlight(); });
  graph.getGraphStorage().add("Target", std::move(positionHolder), RenderGraph::ImageResizePolicy::SWAPCHAIN);

  auto elementMock = std::make_shared<GraphElementMock>();
  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::CLEAR);
  renderPass.addColorTarget("Target", RenderGraph::RenderTargetLoadAction::CLEAR);
  renderPass.registerGraphElement(elementMock);

  // compute queue
  auto& postprocessingPass = graph.createPassCompute("Postprocessing", RenderGraph::QueueType::COMPUTE);
  postprocessingPass.registerGraphElement(elementMock);
  postprocessingPass.addStorageTextureInput("Swapchain");
  postprocessingPass.addStorageTextureOutput("Swapchain");

  auto& guiPass = graph.createPassGraphic("GUI");
  guiPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::KEEP);
  guiPass.registerGraphElement(elementMock);

  graph.calculate();
  commandBuffer[graph.getFrameInFlight()].endCommands();

  auto loadSemaphore = RenderGraph::Semaphore(VK_SEMAPHORE_TYPE_TIMELINE, device);
  uint64_t loadCounter = 1;
  auto semaphore = loadSemaphore.getSemaphore();
  VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = commandBuffer[graph.getFrameInFlight()].getCommandBuffer()};
  VkSemaphoreSubmitInfo signalSemaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                            .semaphore = semaphore,
                                            .value = loadCounter,
                                            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
  VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                           .commandBufferInfoCount = 1,
                           .pCommandBufferInfos = &commandBufferInfo,
                           .signalSemaphoreInfoCount = 1,
                           .pSignalSemaphoreInfos = &signalSemaphoreInfo};
  vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GENERAL), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

  graph.render();

  for (int i = 0; i < swapchainOldImages.size(); i++) {
    EXPECT_NE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews()[i]->getImageView(), nullptr);
    EXPECT_NE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews()[i]->getImage().getImage(),
              nullptr);
  }

  EXPECT_EQ(elementMock->getResetCount(), 0);
  EXPECT_EQ(window.getResolution().x, windowResolution.x);
  EXPECT_EQ(window.getResolution().y, windowResolution.y);
  // call reset explicitly, usually it should be called if render() returns true
  graph.reset(resolution);
  EXPECT_EQ(elementMock->getResetCount(), 3);
  // we don't change window resolution here
  EXPECT_EQ(window.getResolution().x, windowResolution.x);
  EXPECT_EQ(window.getResolution().y, windowResolution.y);

  // we can't guarantee that swapchain images are different after reset, but they should be valid
  for (int i = 0; i < swapchainOldImages.size(); i++) {
    EXPECT_NE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews()[i]->getImageView(), nullptr);
    EXPECT_NE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews()[i]->getImage().getImage(),
              nullptr);
  }

  // reset() must rebuild barriers and swapchain-dependent semaphores.
  EXPECT_FALSE(graph.render());

  // the most important here is the resolution
  std::vector<std::shared_ptr<RenderGraph::ImageView>> swapchainNewImages;
  for (int i = 0; i < swapchainOldImages.size(); i++) {
    auto swapchainImage = std::make_unique<RenderGraph::Image>(allocator, device);
    swapchainImage->createImage(VK_FORMAT_R32G32B32A32_UINT, {720, 480}, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto swapchainImageView = std::make_shared<RenderGraph::ImageView>(std::move(swapchainImage), device);
    swapchainImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    swapchainNewImages.push_back(swapchainImageView);
  }

  graph.getGraphStorage().reset(swapchain.getImageViews(), swapchainNewImages);
  // swapchain is resized differently but resized anyway
  for (int i = 0; i < swapchainOldImages.size(); i++) {
    EXPECT_EQ(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews()[i]->getImage().getResolution().x,
              720);
    EXPECT_EQ(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews()[i]->getImage().getResolution().y,
              480);
  }

  for (int i = 0; i < framesInFlight; i++) {
    EXPECT_EQ(graph.getGraphStorage().getImageViewHolder("Target").getImageViews()[i]->getImage().getResolution().x,
              720);
    EXPECT_EQ(graph.getGraphStorage().getImageViewHolder("Target").getImageViews()[i]->getImage().getResolution().y,
              480);
  }

  // wait device idle before destroying resources
  vkDeviceWaitIdle(device.getLogicalDevice());
}

TEST(ScenarioTest, DepthExistance) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  int framesInFlight = 2;
  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(RenderGraph::QueueType::GENERAL, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.setSwapchain(swapchain);
  graph.initialize();

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  auto depthAttachment = std::make_unique<RenderGraph::Image>(allocator, device);
  depthAttachment->createImage(VK_FORMAT_D32_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_DEPTH_BIT,
                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

  auto depthAttachmentImageView = std::make_shared<RenderGraph::ImageView>(std::move(depthAttachment), device);
  depthAttachmentImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);

  graph.getGraphStorage().add("Depth", std::make_unique<RenderGraph::ImageViewHolder>(
                                           std::vector{depthAttachmentImageView}, []() -> int { return 0; }));

  auto elementMock = std::make_shared<GraphElementMock>();
  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain", RenderGraph::RenderTargetLoadAction::CLEAR,
                            RenderGraph::RenderTargetStoreAction::STORE,
                            VkClearValue{.color = {{0.1f, 0.2f, 0.3f, 1.0f}}});
  renderPass.setDepthTarget("Depth", RenderGraph::RenderTargetLoadAction::CLEAR,
                            RenderGraph::RenderTargetStoreAction::DONT_CARE, VkClearValue{.depthStencil = {0.5f, 0}});
  renderPass.registerGraphElement(elementMock);

  graph.calculate();
  commandBuffer[graph.getFrameInFlight()].endCommands();

  auto loadSemaphore = RenderGraph::Semaphore(VK_SEMAPHORE_TYPE_TIMELINE, device);
  uint64_t loadCounter = 1;
  auto semaphore = loadSemaphore.getSemaphore();
  VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = commandBuffer[graph.getFrameInFlight()].getCommandBuffer()};
  VkSemaphoreSubmitInfo signalSemaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                            .semaphore = semaphore,
                                            .value = loadCounter,
                                            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
  VkSubmitInfo2 submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                           .commandBufferInfoCount = 1,
                           .pCommandBufferInfos = &commandBufferInfo,
                           .signalSemaphoreInfoCount = 1,
                           .pSignalSemaphoreInfos = &signalSemaphoreInfo};
  vkQueueSubmit2(device.getQueue(RenderGraph::QueueType::GENERAL), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

  graph.render();

  // wait device idle before destroying resources
  vkDeviceWaitIdle(device.getLogicalDevice());
}

TEST(ScenarioTest, TraversalKeepsTransitiveProducerBeforeConsumer) {
  glm::ivec2 resolution(1920, 1080);

  RenderGraph::Instance instance("TestApp", false);

  RenderGraph::Window window(resolution);
  window.initialize();

  RenderGraph::Surface surface(window, instance);

  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);

  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;

  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);
  graph.setSwapchain(swapchain);

  graph.initialize();

  auto addStorageBuffers = [&](std::string_view name) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                                              allocator, device));
    }

    graph.getGraphStorage().add(name, std::make_unique<RenderGraph::BufferHandler>(
                                          std::move(buffers), [&graph] { return graph.getFrameInFlight(); }));
  };

  addStorageBuffers("X");
  addStorageBuffers("Y");

  // First produces X.
  auto& firstPass = graph.createPassCompute("First", RenderGraph::QueueType::GENERAL);

  firstPass.addStorageBufferOutput("X");

  // Middle depends on First through X and produces Y.
  auto& middlePass = graph.createPassCompute("Middle", RenderGraph::QueueType::GENERAL);

  middlePass.addStorageBufferInput("X");
  middlePass.addStorageBufferOutput("Y");

  // Last depends directly on First through X
  // and transitively on First through Middle -> Y.
  auto& lastPass = graph.createPassCompute("Last", RenderGraph::QueueType::GENERAL);

  /*
   * This input order exposes the current traversal bug:
   *
   * passNext becomes [First, Middle].
   * First is traversed and removed from passesBackup.
   * Middle is then unable to find First as its producer.
   * push_front() produces Middle, First, Last.
   */
  lastPass.addStorageBufferInput("X");
  lastPass.addStorageBufferInput("Y");

  graph.calculate();

  ASSERT_EQ(graph._passesOrdered.size(), 3);

  auto orderedPass = graph._passesOrdered.begin();

  EXPECT_EQ(*orderedPass, &firstPass);
  ++orderedPass;

  EXPECT_EQ(*orderedPass, &middlePass);
  ++orderedPass;

  EXPECT_EQ(*orderedPass, &lastPass);
}

TEST(ScenarioTest, TraversalDiamondGraph) {
  /*
   *                         Source
   *                         writes X
   *                         /      \
   *                       X          X
   *                      /            \
   *                     v              v
   *                   Left           Right
   *                reads X         reads X
   *                writes Y        writes Z
   *                     \              /
   *                      Y            Z
   *                       \          /
   *                        v        v
   *                          Join
   *                      reads Y and Z
   *
   * Required ordering:
   *
   *   Source < Left  < Join
   *   Source < Right < Join
   *
   * The relative order of Left and Right does not matter.
   */

  glm::ivec2 resolution(1920, 1080);

  RenderGraph::Instance instance("TestApp", false);

  RenderGraph::Window window(resolution);
  window.initialize();

  RenderGraph::Surface surface(window, instance);

  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);

  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;

  RenderGraph::Tracer tracer(device);
  RenderGraph::Graph graph(4, framesInFlight, device, tracer);
  graph.setSwapchain(swapchain);

  graph.initialize();

  auto addStorageBuffers = [&](std::string_view name) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                                              allocator, device));
    }

    graph.getGraphStorage().add(name, std::make_unique<RenderGraph::BufferHandler>(
                                          std::move(buffers), [&graph] { return graph.getFrameInFlight(); }));
  };

  addStorageBuffers("X");
  addStorageBuffers("Y");
  addStorageBuffers("Z");

  auto& sourcePass = graph.createPassCompute("Source", RenderGraph::QueueType::GENERAL);

  sourcePass.addStorageBufferOutput("X");

  auto& leftPass = graph.createPassCompute("Left", RenderGraph::QueueType::GENERAL);

  leftPass.addStorageBufferInput("X");
  leftPass.addStorageBufferOutput("Y");

  auto& rightPass = graph.createPassCompute("Right", RenderGraph::QueueType::GENERAL);

  rightPass.addStorageBufferInput("X");
  rightPass.addStorageBufferOutput("Z");

  auto& joinPass = graph.createPassCompute("Join", RenderGraph::QueueType::GENERAL);

  joinPass.addStorageBufferInput("Y");
  joinPass.addStorageBufferInput("Z");

  graph.calculate();

  ASSERT_EQ(graph._passesOrdered.size(), 4);

  const auto sourceIt = std::ranges::find(graph._passesOrdered, &sourcePass);

  const auto leftIt = std::ranges::find(graph._passesOrdered, &leftPass);

  const auto rightIt = std::ranges::find(graph._passesOrdered, &rightPass);

  const auto joinIt = std::ranges::find(graph._passesOrdered, &joinPass);

  ASSERT_NE(sourceIt, graph._passesOrdered.end());
  ASSERT_NE(leftIt, graph._passesOrdered.end());
  ASSERT_NE(rightIt, graph._passesOrdered.end());
  ASSERT_NE(joinIt, graph._passesOrdered.end());

  const auto sourceIndex = std::distance(graph._passesOrdered.begin(), sourceIt);

  const auto leftIndex = std::distance(graph._passesOrdered.begin(), leftIt);

  const auto rightIndex = std::distance(graph._passesOrdered.begin(), rightIt);

  const auto joinIndex = std::distance(graph._passesOrdered.begin(), joinIt);

  EXPECT_LT(sourceIndex, leftIndex);
  EXPECT_LT(sourceIndex, rightIndex);
  EXPECT_LT(leftIndex, joinIndex);
  EXPECT_LT(rightIndex, joinIndex);

  RenderGraph::Graph imageGraph(1, 1, device, tracer);
  imageGraph.initialize();
  auto image = std::make_unique<RenderGraph::Image>(allocator, device);
  image->createImage(VK_FORMAT_R8G8B8A8_UNORM, {8, 8}, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  auto imageView = std::make_shared<RenderGraph::ImageView>(std::move(image), device);
  imageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
  imageGraph.getGraphStorage().add(
      "Image", std::make_unique<RenderGraph::ImageViewHolder>(std::vector{imageView}, [] { return 0; }));

  auto& initialGraphicsReader = imageGraph.createPassGraphic("InitialSample");
  initialGraphicsReader.addTextureInput("Image");
  auto& initialComputeReader = imageGraph.createPassCompute("InitialRead", RenderGraph::QueueType::GENERAL);
  initialComputeReader.addStorageTextureInput("Image");
  auto& imageWriter = imageGraph.createPassCompute("WriteImage", RenderGraph::QueueType::GENERAL);
  imageWriter.addStorageTextureOutput("Image");
  auto& graphicsReader = imageGraph.createPassGraphic("SampleImage");
  graphicsReader.addTextureInput("Image");
  auto& computeReader = imageGraph.createPassCompute("ReadImage", RenderGraph::QueueType::GENERAL);
  computeReader.addStorageTextureInput("Image");
  auto& imageRewrite = imageGraph.createPassCompute("RewriteImage", RenderGraph::QueueType::GENERAL);
  imageRewrite.addStorageTextureOutput("Image");
  auto& sampleRewrite = imageGraph.createPassGraphic("SampleRewrittenImage");
  sampleRewrite.addTextureInput("Image");
  imageGraph.calculate();
  imageGraph.calculate();

  // calculate() leaves layouts untouched; render planning transitions the selected physical image.
  EXPECT_EQ(imageView->getImage().getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_TRUE(imageGraph._physicalBarriers.at(&initialGraphicsReader).getImageBarriers().empty());
  imageGraph._preparePhysicalAccesses();
  ASSERT_EQ(imageGraph._physicalBarriers.at(&initialGraphicsReader).getImageBarriers().size(), 1);
  const auto& layoutBarrier = imageGraph._physicalBarriers.at(&initialGraphicsReader).getImageBarriers().front();
  EXPECT_EQ(layoutBarrier.oldLayout, VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_EQ(layoutBarrier.newLayout, VK_IMAGE_LAYOUT_GENERAL);
  EXPECT_EQ(layoutBarrier.dstStageMask, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  EXPECT_EQ(layoutBarrier.dstAccessMask, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
  EXPECT_EQ(imageView->getImage().getImageLayout(), VK_IMAGE_LAYOUT_GENERAL);
  // The initial layout transition is a write; subsequent readers must also depend on it.
  ASSERT_EQ(imageGraph._physicalBarriers.at(&initialComputeReader).getImageBarriers().size(), 1);
  const auto& initialReadBarrier = imageGraph._physicalBarriers.at(&initialComputeReader).getImageBarriers().front();
  EXPECT_EQ(initialReadBarrier.srcAccessMask, VK_ACCESS_2_MEMORY_WRITE_BIT);
  const auto& initialWriteBarriers = imageGraph._physicalBarriers.at(&imageWriter).getImageBarriers();
  ASSERT_EQ(initialWriteBarriers.size(), 1);
  const auto& initialWriteBarrier = initialWriteBarriers.front();
  EXPECT_EQ(initialWriteBarrier.srcStageMask, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
                                                  VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(initialWriteBarrier.srcAccessMask, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
  EXPECT_EQ(initialWriteBarrier.dstStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(initialWriteBarrier.dstAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);

  // Sampling the image does not remove the write dependency needed by a later compute reader.
  const auto& computeBarriers = imageGraph._physicalBarriers.at(&computeReader).getImageBarriers();
  ASSERT_EQ(computeBarriers.size(), 1);
  const auto& computeBarrier = computeBarriers.front();
  EXPECT_EQ(computeBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(computeBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);
  EXPECT_EQ(computeBarrier.dstStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(computeBarrier.dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);

  const auto& imageRewriteBarriers = imageGraph._physicalBarriers.at(&imageRewrite).getImageBarriers();
  ASSERT_EQ(imageRewriteBarriers.size(), 1);
  const auto& imageRewriteBarrier = imageRewriteBarriers.front();
  EXPECT_EQ(imageRewriteBarrier.srcStageMask,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
  EXPECT_EQ(imageRewriteBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);

  // A new writer starts a fresh history; old graphics readers must not remain in its source mask.
  const auto& sampleRewriteBarriers = imageGraph._physicalBarriers.at(&sampleRewrite).getImageBarriers();
  ASSERT_EQ(sampleRewriteBarriers.size(), 1);
  const auto& sampleRewriteBarrier = sampleRewriteBarriers.front();
  EXPECT_EQ(sampleRewriteBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(sampleRewriteBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);
  EXPECT_EQ(sampleRewriteBarrier.dstStageMask, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
  EXPECT_EQ(sampleRewriteBarrier.dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);

  for (const auto& [pass, barrier] : imageGraph._physicalBarriers) {
    for (const auto& imageBarrier : barrier.getImageBarriers()) {
      EXPECT_EQ(imageBarrier.oldLayout,
                pass == &initialGraphicsReader ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL);
      EXPECT_EQ(imageBarrier.newLayout, VK_IMAGE_LAYOUT_GENERAL);
      EXPECT_EQ(imageBarrier.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
      EXPECT_EQ(imageBarrier.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
    }
  }
}
