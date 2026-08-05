#define VK_NO_PROTOTYPES
#include <volk.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
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
import Texture;
import CommandPool;
import Command;
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

TEST(ScenarioTest, GraphOneQueue) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  int framesInFlight = 2;
  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(vkb::QueueType::graphics, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.initialize();

  EXPECT_EQ(graph.getFrameInFlight(), 0);

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  std::vector<std::shared_ptr<RenderGraph::ImageView>> positionImageViews;
  for (int i = 0; i < framesInFlight; i++) {
    auto positionImage = std::make_unique<RenderGraph::Image>(allocator);
    positionImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    positionImage->changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                commandBuffer[graph.getFrameInFlight()]);
    auto positionImageView = std::make_shared<RenderGraph::ImageView>(std::move(positionImage), device);
    positionImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    positionImageViews.push_back(positionImageView);
  }

  std::unique_ptr<RenderGraph::ImageViewHolder> positionHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      positionImageViews, [&]() { return graph.getFrameInFlight(); });
  graph.getGraphStorage().add("Target", std::move(positionHolder));

  EXPECT_GE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews().size(), 2);

  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain");
  renderPass.addColorTarget("Target");
  renderPass.clearTarget("Swapchain");
  renderPass.clearTarget("Target");
  auto elementMock = std::make_shared<GraphElementMock>();
  renderPass.registerGraphElement(elementMock);

  EXPECT_THROW(graph.createPassGraphic("Render"), std::logic_error);
  EXPECT_THROW(graph.createPassCompute("Render", false), std::logic_error);

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

  auto& postprocessingPass = graph.createPassCompute("Postprocessing", false);
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
  guiPass.addColorTarget("Swapchain");
  guiPass.registerGraphElement(elementMock);

  EXPECT_EQ(guiPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(guiPass.getCommandBuffers()[i], guiPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(guiPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 1);

  graph.calculate();

  RenderGraph::Graph offscreenGraph(1, framesInFlight, swapchain, window, device);
  offscreenGraph.initialize();
  offscreenGraph.getGraphStorage().add(
      "Offscreen", std::make_unique<RenderGraph::ImageViewHolder>(positionImageViews, []() { return 0; }));
  auto& offscreenPass = offscreenGraph.createPassGraphic("Offscreen");
  offscreenPass.addColorTarget("Offscreen");
  offscreenPass.clearTarget("Offscreen");
  offscreenGraph.calculate();

  ASSERT_TRUE(offscreenGraph._sync.contains(&offscreenPass));
  const auto offscreenWaitSemaphores = offscreenGraph._sync.at(&offscreenPass).getWaitSemaphores();
  ASSERT_EQ(offscreenWaitSemaphores.size(), 1);
  EXPECT_EQ(offscreenWaitSemaphores.front(), offscreenGraph._semaphoreImageAvailable.front().get());

  // Render waits for the swapchain image-available semaphore.
  ASSERT_TRUE(graph._sync.contains(&renderPass));
  EXPECT_EQ(graph._sync.at(&renderPass).getWaitSemaphores().size(), 1);
  EXPECT_EQ(graph._sync.at(&renderPass).getSignalSemaphores().size(), 0);
  // Postprocessing does not use any semaphores.
  ASSERT_TRUE(graph._sync.contains(&postprocessingPass));
  EXPECT_EQ(graph._sync.at(&postprocessingPass).getWaitSemaphores().size(), 0);
  EXPECT_EQ(graph._sync.at(&postprocessingPass).getSignalSemaphores().size(), 0);
  // GUI signals the render-finished semaphore.
  ASSERT_TRUE(graph._sync.contains(&guiPass));
  EXPECT_EQ(graph._sync.at(&guiPass).getWaitSemaphores().size(), 0);
  EXPECT_EQ(graph._sync.at(&guiPass).getSignalSemaphores().size(), 1);
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
  vkQueueSubmit2(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

  graph.render();
  EXPECT_TRUE(graph.getTimestamps().empty());
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 3);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  graph.render();
  EXPECT_TRUE(graph.getTimestamps().empty());
  EXPECT_EQ(graph.getFrameInFlight(), (2 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 6);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  graph.render();
  auto timestamps1 = graph.getTimestamps();
  EXPECT_EQ(timestamps1.size(), 3);
  EXPECT_TRUE(timestamps1.find("Render") != timestamps1.end());
  EXPECT_TRUE(timestamps1.find("Postprocessing") != timestamps1.end());
  EXPECT_TRUE(timestamps1.find("GUI") != timestamps1.end());
  EXPECT_GE(timestamps1["Render"].y, timestamps1["Render"].x);
  EXPECT_GE(timestamps1["Postprocessing"].x, timestamps1["Render"].y);
  EXPECT_GE(timestamps1["Postprocessing"].y, timestamps1["Postprocessing"].x);
  EXPECT_GE(timestamps1["GUI"].x, timestamps1["Postprocessing"].y);
  EXPECT_GE(timestamps1["GUI"].y, timestamps1["GUI"].x);
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 9);
  EXPECT_EQ(elementMock->getUpdateCount(), elementMock->getDrawCount());

  graph.render();
  auto timestamps2 = graph.getTimestamps();
  EXPECT_EQ(timestamps2.size(), 3);
  EXPECT_TRUE(timestamps2.find("Render") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("Postprocessing") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("GUI") != timestamps2.end());
  EXPECT_GE(timestamps2["Render"].y, timestamps2["Render"].x);
  EXPECT_GE(timestamps2["Postprocessing"].x, timestamps2["Render"].y);
  EXPECT_GE(timestamps2["Postprocessing"].y, timestamps2["Postprocessing"].x);
  EXPECT_GE(timestamps2["GUI"].x, timestamps2["Postprocessing"].y);
  EXPECT_GE(timestamps2["GUI"].y, timestamps2["GUI"].x);
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
  RenderGraph::Device device(surface, instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  int framesInFlight = 2;
  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(vkb::QueueType::graphics, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.initialize();

  EXPECT_EQ(graph.getFrameInFlight(), 0);

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  std::vector<std::shared_ptr<RenderGraph::ImageView>> positionImageViews;
  for (int i = 0; i < framesInFlight; i++) {
    auto positionImage = std::make_unique<RenderGraph::Image>(allocator);
    positionImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    positionImage->changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                commandBuffer[graph.getFrameInFlight()]);
    auto positionImageView = std::make_shared<RenderGraph::ImageView>(std::move(positionImage), device);
    positionImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    positionImageViews.push_back(positionImageView);
  }

  std::unique_ptr<RenderGraph::ImageViewHolder> positionHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      positionImageViews, [&]() { return graph.getFrameInFlight(); });
  graph.getGraphStorage().add("Target", std::move(positionHolder));

  EXPECT_GE(graph.getGraphStorage().getImageViewHolder("Swapchain").getImageViews().size(), 2);

  auto elementMock = std::make_shared<GraphElementMock>();
  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain");
  renderPass.addColorTarget("Target");
  renderPass.clearTarget("Swapchain");
  renderPass.clearTarget("Target");
  renderPass.registerGraphElement(elementMock);

  EXPECT_EQ(renderPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(renderPass.getCommandBuffers()[i], renderPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(renderPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 2);
  EXPECT_EQ(renderPass.getDepthTarget(), std::nullopt);

  // separate queue
  auto& postprocessingPass = graph.createPassCompute("Postprocessing", true);
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
  guiPass.addColorTarget("Swapchain");
  guiPass.registerGraphElement(elementMock);

  EXPECT_EQ(guiPass.getCommandBuffers().size(), framesInFlight);
  for (int i = 1; i < framesInFlight; i++) {
    EXPECT_NE(guiPass.getCommandBuffers()[i], guiPass.getCommandBuffers()[i - 1]);
  }

  EXPECT_EQ(guiPass.getPipelineGraphic(graph.getGraphStorage()).getColorAttachments().size(), 1);

  graph.calculate();
  const bool separateQueueFamilies = device.getQueueIndex(vkb::QueueType::graphics) !=
                                     device.getQueueIndex(vkb::QueueType::compute);

  // Ownership transfer: Render -> Postprocessing.
  EXPECT_EQ(graph._sync.at(&renderPass).getBarriersAfter().size(), separateQueueFamilies ? 1 : 0);
  EXPECT_EQ(graph._sync.at(&postprocessingPass).getBarriersBefore().size(), separateQueueFamilies ? 1 : 0);
  // Ownership transfer: Postprocessing -> GUI.
  EXPECT_EQ(graph._sync.at(&postprocessingPass).getBarriersAfter().size(), separateQueueFamilies ? 1 : 0);
  EXPECT_EQ(graph._sync.at(&guiPass).getBarriersBefore().size(), separateQueueFamilies ? 1 : 0);
  // Render waits for imageAvailable and signals the queue-transfer semaphore.
  ASSERT_TRUE(graph._sync.contains(&renderPass));
  EXPECT_EQ(graph._sync.at(&renderPass).getWaitSemaphores().size(), 1);
  EXPECT_EQ(graph._sync.at(&renderPass).getSignalSemaphores().size(), 1);
  // Postprocessing waits for Render and signals GUI.
  ASSERT_TRUE(graph._sync.contains(&postprocessingPass));
  EXPECT_EQ(graph._sync.at(&postprocessingPass).getWaitSemaphores().size(), 1);
  EXPECT_EQ(graph._sync.at(&postprocessingPass).getSignalSemaphores().size(), 1);
  // GUI waits for Postprocessing and signals renderFinished.
  ASSERT_TRUE(graph._sync.contains(&guiPass));
  EXPECT_EQ(graph._sync.at(&guiPass).getWaitSemaphores().size(), 1);
  EXPECT_EQ(graph._sync.at(&guiPass).getSignalSemaphores().size(), 1);
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
  vkQueueSubmit2(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

  graph.render();
  EXPECT_TRUE(graph.getTimestamps().empty());
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 3);

  graph.render();
  EXPECT_TRUE(graph.getTimestamps().empty());
  EXPECT_EQ(graph.getFrameInFlight(), (2 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 6);

  graph.render();
  auto timestamps1 = graph.getTimestamps();
  EXPECT_EQ(timestamps1.size(), 3);
  EXPECT_TRUE(timestamps1.find("Render") != timestamps1.end());
  EXPECT_TRUE(timestamps1.find("Postprocessing") != timestamps1.end());
  EXPECT_TRUE(timestamps1.find("GUI") != timestamps1.end());
  EXPECT_GE(timestamps1["Render"].y, timestamps1["Render"].x);
  EXPECT_GE(timestamps1["Postprocessing"].x, timestamps1["Render"].y);
  EXPECT_GE(timestamps1["Postprocessing"].y, timestamps1["Postprocessing"].x);
  EXPECT_GE(timestamps1["GUI"].x, timestamps1["Postprocessing"].y);
  EXPECT_GE(timestamps1["GUI"].y, timestamps1["GUI"].x);
  EXPECT_EQ(graph.getFrameInFlight(), 1);
  EXPECT_EQ(elementMock->getDrawCount(), 9);

  graph.render();
  auto timestamps2 = graph.getTimestamps();
  EXPECT_EQ(timestamps2.size(), 3);
  EXPECT_TRUE(timestamps2.find("Render") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("Postprocessing") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("GUI") != timestamps2.end());
  EXPECT_GE(timestamps2["Render"].y, timestamps2["Render"].x);
  EXPECT_GE(timestamps2["Postprocessing"].x, timestamps2["Render"].y);
  EXPECT_GE(timestamps2["Postprocessing"].y, timestamps2["Postprocessing"].x);
  EXPECT_GE(timestamps2["GUI"].x, timestamps2["Postprocessing"].y);
  EXPECT_GE(timestamps2["GUI"].y, timestamps2["GUI"].x);
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

TEST(ScenarioTest, BufferOwnershipTransferUsesLastResourceOwner) {
  glm::ivec2 resolution(1920, 1080);

  RenderGraph::Instance instance("TestApp", false);

  RenderGraph::Window window(resolution);
  window.initialize();

  RenderGraph::Surface surface(window, instance);

  RenderGraph::Device device(surface, instance);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);

  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;
  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);
  graph.initialize();

  auto addStorageBuffers = [&](std::string_view name) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(
          1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
    }

    graph.getGraphStorage().add(name, buffers);
  };

  addStorageBuffers("X");
  addStorageBuffers("Y");
  addStorageBuffers("Link");

  /*
   * X, Y, Link are resources we want to pass between First, Middle, Last passes.
   * Passes flow is: First -> Middle -> Last
   *
   * graphics queue: First(writes: X, writes: Link) -> Middle(reads: Link, writes: Y)
   * compute queue: Last(reads: X, Y)
   *
   * Ownership transfers: X: First -> Last, Y: Middle -> Last.
   * Link stays on the graphics queue and requires no transfer.
   */

  auto& firstPass = graph.createPassCompute("First", false);
  firstPass.addStorageBufferOutput("X");
  firstPass.addStorageBufferOutput("Link");

  auto& middlePass = graph.createPassCompute("Middle", false);
  middlePass.addStorageBufferInput("Link");
  middlePass.addStorageBufferOutput("Y");

  auto& lastPass = graph.createPassCompute("Last", true);

  // Preserve the First -> Middle -> Last dependency traversal.
  lastPass.addStorageBufferInput("Y");
  lastPass.addStorageBufferInput("X");

  graph.calculate();

  const bool separateQueueFamilies = device.getQueueIndex(vkb::QueueType::graphics) !=
                                     device.getQueueIndex(vkb::QueueType::compute);

  ASSERT_TRUE(graph._sync.contains(&middlePass));
  ASSERT_TRUE(graph._sync.contains(&lastPass));

  // Link stays on the graphics queue and requires one ordinary barrier before Middle.
  EXPECT_EQ(graph._sync.at(&middlePass).getBarriersBefore().size(), 1);

  if (separateQueueFamilies) {
    ASSERT_TRUE(graph._sync.contains(&firstPass));
    // X is released by First, Y is released by Middle, and Last acquires both.
    EXPECT_EQ(graph._sync.at(&firstPass).getBarriersAfter().size(), 1);
    EXPECT_EQ(graph._sync.at(&middlePass).getBarriersAfter().size(), 1);
    EXPECT_EQ(graph._sync.at(&lastPass).getBarriersBefore().size(), 2);
  } else {
    EXPECT_FALSE(graph._sync.contains(&firstPass));
    EXPECT_TRUE(graph._sync.at(&middlePass).getBarriersAfter().empty());
    EXPECT_TRUE(graph._sync.at(&lastPass).getBarriersBefore().empty());
  }

  const auto middleSignalSemaphores = graph._sync.at(&middlePass).getSignalSemaphores();
  const auto lastWaitSemaphores = graph._sync.at(&lastPass).getWaitSemaphores();
  ASSERT_EQ(middleSignalSemaphores.size(), 1);
  ASSERT_EQ(lastWaitSemaphores.size(), 2);
  EXPECT_TRUE(std::ranges::contains(lastWaitSemaphores, middleSignalSemaphores.front()));
  EXPECT_TRUE(std::ranges::contains(lastWaitSemaphores, graph._semaphoreImageAvailable.front().get()));
}

TEST(ScenarioTest, GraphicsPassBufferInputs) {
  const glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;
  RenderGraph::Graph graph(2, framesInFlight, swapchain, window, device);
  graph.initialize();
  graph.getGraphStorage().add(
      "Swapchain", std::make_unique<RenderGraph::ImageViewHolder>(
                       swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); }));

  auto addBuffers = [&](std::string_view name, VkBufferUsageFlags usage) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);
    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(
          1024, usage, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
    }
    graph.getGraphStorage().add(name, buffers);
  };

  addBuffers("VisibleObjects", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  addBuffers("IndirectCommands", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  addBuffers("Vertices", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  addBuffers("Indices", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

  auto& cullPass = graph.createPassCompute("Cull", false);
  cullPass.addStorageBufferOutput("VisibleObjects");
  cullPass.addStorageBufferOutput("IndirectCommands");
  cullPass.addStorageBufferOutput("Vertices");
  cullPass.addStorageBufferOutput("Indices");

  auto& drawPass = graph.createPassGraphic("Draw");
  drawPass.addStorageBufferInput("VisibleObjects");
  drawPass.addIndirectBufferInput("IndirectCommands");
  drawPass.addVertexBufferInput("Vertices");
  drawPass.addIndexBufferInput("Indices");
  drawPass.addColorTarget("Swapchain");

  graph.calculate();

  ASSERT_EQ(drawPass.getStorageBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getStorageBufferInputs().front(), "VisibleObjects");
  ASSERT_EQ(drawPass.getIndirectBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getIndirectBufferInputs().front(), "IndirectCommands");
  ASSERT_EQ(drawPass.getVertexBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getVertexBufferInputs().front(), "Vertices");
  ASSERT_EQ(drawPass.getIndexBufferInputs().size(), 1);
  EXPECT_EQ(drawPass.getIndexBufferInputs().front(), "Indices");

  ASSERT_EQ(graph._passesOrdered.size(), 2);
  EXPECT_EQ(graph._passesOrdered.front(), &cullPass);
  EXPECT_EQ(graph._passesOrdered.back(), &drawPass);

  ASSERT_TRUE(graph._sync.contains(&drawPass));
  const auto barriers = graph._sync.at(&drawPass).getBarriersBefore();
  ASSERT_EQ(barriers.size(), 4);

  const VkBuffer visibleObjectsBuffer = graph.getGraphStorage().getBuffer("VisibleObjects")[0]->getBuffer();
  const VkBuffer indirectCommandsBuffer = graph.getGraphStorage().getBuffer("IndirectCommands")[0]->getBuffer();
  const VkBuffer vertexBuffer = graph.getGraphStorage().getBuffer("Vertices")[0]->getBuffer();
  const VkBuffer indexBuffer = graph.getGraphStorage().getBuffer("Indices")[0]->getBuffer();
  const VkPipelineStageFlags2 graphicsShaderStages =
      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
      VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

  bool foundStorageBarrier = false;
  bool foundIndirectBarrier = false;
  bool foundVertexBarrier = false;
  bool foundIndexBarrier = false;
  for (const auto* barrier : barriers) {
    ASSERT_EQ(barrier->getBufferBarriers().size(), 1);
    const auto& bufferBarrier = barrier->getBufferBarriers().front();
    EXPECT_EQ(bufferBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(bufferBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);

    if (bufferBarrier.buffer == visibleObjectsBuffer) {
      foundStorageBarrier = true;
      EXPECT_EQ(bufferBarrier.dstStageMask, graphicsShaderStages);
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
  RenderGraph::Device device(surface, instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;
  RenderGraph::Graph graph(2, framesInFlight, swapchain, window, device);
  graph.initialize();

  std::vector<std::unique_ptr<RenderGraph::Buffer>> indirectBuffers;
  indirectBuffers.reserve(framesInFlight);
  for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
    indirectBuffers.push_back(std::make_unique<RenderGraph::Buffer>(
        sizeof(VkDispatchIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
  }
  graph.getGraphStorage().add("DispatchCommands", indirectBuffers);

  auto& generatePass = graph.createPassCompute("GenerateDispatch", false);
  generatePass.addStorageBufferOutput("DispatchCommands");

  auto& dispatchPass = graph.createPassCompute("DispatchIndirect", false);
  dispatchPass.addIndirectBufferInput("DispatchCommands");

  graph.calculate();

  ASSERT_EQ(dispatchPass.getIndirectBufferInputs().size(), 1);
  EXPECT_EQ(dispatchPass.getIndirectBufferInputs().front(), "DispatchCommands");

  ASSERT_EQ(graph._passesOrdered.size(), 2);
  EXPECT_EQ(graph._passesOrdered.front(), &generatePass);
  EXPECT_EQ(graph._passesOrdered.back(), &dispatchPass);

  ASSERT_TRUE(graph._sync.contains(&dispatchPass));
  const auto barriers = graph._sync.at(&dispatchPass).getBarriersBefore();
  ASSERT_EQ(barriers.size(), 1);
  ASSERT_EQ(barriers.front()->getBufferBarriers().size(), 1);

  const auto& bufferBarrier = barriers.front()->getBufferBarriers().front();
  EXPECT_EQ(bufferBarrier.srcStageMask, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(bufferBarrier.srcAccessMask, VK_ACCESS_2_SHADER_WRITE_BIT);
  EXPECT_EQ(bufferBarrier.dstStageMask, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
  EXPECT_EQ(bufferBarrier.dstAccessMask, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

TEST(ScenarioTest, GraphReset) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  int framesInFlight = 2;
  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(vkb::QueueType::graphics, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.initialize();

  auto swapchainOldImages = swapchain.getImageViews();

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  std::vector<std::shared_ptr<RenderGraph::ImageView>> positionImageViews;
  for (int i = 0; i < framesInFlight; i++) {
    auto positionImage = std::make_unique<RenderGraph::Image>(allocator);
    positionImage->createImage(VK_FORMAT_R16G16B16A16_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    positionImage->changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                commandBuffer[graph.getFrameInFlight()]);
    auto positionImageView = std::make_shared<RenderGraph::ImageView>(std::move(positionImage), device);
    positionImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);
    positionImageViews.push_back(positionImageView);
  }

  std::unique_ptr<RenderGraph::ImageViewHolder> positionHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      positionImageViews, [&]() { return graph.getFrameInFlight(); });
  graph.getGraphStorage().add("Target", std::move(positionHolder));

  auto elementMock = std::make_shared<GraphElementMock>();
  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain");
  renderPass.addColorTarget("Target");
  renderPass.clearTarget("Swapchain");
  renderPass.clearTarget("Target");
  renderPass.registerGraphElement(elementMock);

  // separate queue
  auto& postprocessingPass = graph.createPassCompute("Postprocessing", true);
  postprocessingPass.registerGraphElement(elementMock);
  postprocessingPass.addStorageTextureInput("Swapchain");
  postprocessingPass.addStorageTextureOutput("Swapchain");

  auto& guiPass = graph.createPassGraphic("GUI");
  guiPass.addColorTarget("Swapchain");
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
  vkQueueSubmit2(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

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
  EXPECT_EQ(window.getResolution().x, 1920);
  EXPECT_EQ(window.getResolution().y, 1080);
  // call reset explicitly, usually it should be called if render() returns true
  graph.reset();
  EXPECT_EQ(elementMock->getResetCount(), 3);
  // we don't change window resolution here
  EXPECT_EQ(window.getResolution().x, 1920);
  EXPECT_EQ(window.getResolution().y, 1080);

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
    auto swapchainImage = std::make_unique<RenderGraph::Image>(allocator);
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
  RenderGraph::Device device(surface, instance);
  device.initialize();
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  int framesInFlight = 2;
  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);

  auto commandPool = std::make_shared<RenderGraph::CommandPool>(vkb::QueueType::graphics, device);
  std::vector<RenderGraph::CommandBuffer> commandBuffer;
  commandBuffer.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    commandBuffer.emplace_back(*commandPool, device);
  }

  commandBuffer[graph.getFrameInFlight()].beginCommands();
  swapchain.initialize();
  graph.initialize();

  std::unique_ptr<RenderGraph::ImageViewHolder> swapchainHolder = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });

  graph.getGraphStorage().add("Swapchain", std::move(swapchainHolder));

  auto depthAttachment = std::make_unique<RenderGraph::Image>(allocator);
  depthAttachment->createImage(VK_FORMAT_D32_SFLOAT, resolution, 1, 1, VK_IMAGE_ASPECT_DEPTH_BIT,
                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  // set layout to depth image
  depthAttachment->changeLayout(
      depthAttachment->getImageLayout(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 0,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, commandBuffer[graph.getFrameInFlight()]);

  auto depthAttachmentImageView = std::make_shared<RenderGraph::ImageView>(std::move(depthAttachment), device);
  depthAttachmentImageView->createImageView(VK_IMAGE_VIEW_TYPE_2D, 0, 0);

  graph.getGraphStorage().add("Depth", std::make_unique<RenderGraph::ImageViewHolder>(
                                           std::vector{depthAttachmentImageView}, []() -> int { return 0; }));

  auto elementMock = std::make_shared<GraphElementMock>();
  auto& renderPass = graph.createPassGraphic("Render");
  renderPass.addColorTarget("Swapchain");
  renderPass.setDepthTarget("Depth");
  renderPass.clearTarget("Swapchain");
  renderPass.clearTarget("Depth");
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
  vkQueueSubmit2(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

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

  RenderGraph::Device device(surface, instance);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);

  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;

  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);

  graph.initialize();

  auto addStorageBuffers = [&](std::string_view name) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(
          1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
    }

    graph.getGraphStorage().add(name, buffers);
  };

  addStorageBuffers("X");
  addStorageBuffers("Y");

  // First produces X.
  auto& firstPass = graph.createPassCompute("First", false);

  firstPass.addStorageBufferOutput("X");

  // Middle depends on First through X and produces Y.
  auto& middlePass = graph.createPassCompute("Middle", false);

  middlePass.addStorageBufferInput("X");
  middlePass.addStorageBufferOutput("Y");

  // Last depends directly on First through X
  // and transitively on First through Middle -> Y.
  auto& lastPass = graph.createPassCompute("Last", false);

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

  RenderGraph::Device device(surface, instance);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);

  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  constexpr int framesInFlight = 2;

  RenderGraph::Graph graph(4, framesInFlight, swapchain, window, device);

  graph.initialize();

  auto addStorageBuffers = [&](std::string_view name) {
    std::vector<std::unique_ptr<RenderGraph::Buffer>> buffers;
    buffers.reserve(framesInFlight);

    for (int frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
      buffers.push_back(std::make_unique<RenderGraph::Buffer>(
          1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator));
    }

    graph.getGraphStorage().add(name, buffers);
  };

  addStorageBuffers("X");
  addStorageBuffers("Y");
  addStorageBuffers("Z");

  auto& sourcePass = graph.createPassCompute("Source", false);

  sourcePass.addStorageBufferOutput("X");

  auto& leftPass = graph.createPassCompute("Left", false);

  leftPass.addStorageBufferInput("X");
  leftPass.addStorageBufferOutput("Y");

  auto& rightPass = graph.createPassCompute("Right", false);

  rightPass.addStorageBufferInput("X");
  rightPass.addStorageBufferOutput("Z");

  auto& joinPass = graph.createPassCompute("Join", false);

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
}
