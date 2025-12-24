#include <gtest/gtest.h>
import Instance;
import Window;
import Surface;
import Allocator;
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
  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain,
             const RenderGraph::CommandBuffer& commandBuffer) override {
    _resetCount++;
  }

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
    positionImage->changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_NONE, VK_ACCESS_NONE,
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

  // wait swapchain semaphore
  EXPECT_EQ(renderPass.getWaitSemaphores().size(), 1);
  EXPECT_EQ(renderPass.getSignalSemaphores().size(), 0);
  EXPECT_EQ(postprocessingPass.getWaitSemaphores().size(), 0);
  EXPECT_EQ(postprocessingPass.getSignalSemaphores().size(), 0);
  EXPECT_EQ(guiPass.getWaitSemaphores().size(), 0);
  EXPECT_EQ(guiPass.getSignalSemaphores().size(), 1);
  commandBuffer[graph.getFrameInFlight()].endCommands();

  auto loadSemaphore = RenderGraph::Semaphore(VK_SEMAPHORE_TYPE_TIMELINE, device);
  uint64_t loadCounter = 1;
  VkTimelineSemaphoreSubmitInfo timelineInfo = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &loadCounter,
  };

  auto semaphore = loadSemaphore.getSemaphore();
  VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .pNext = &timelineInfo,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &commandBuffer[graph.getFrameInFlight()].getCommandBuffer(),
                             .signalSemaphoreCount = 1,
                             .pSignalSemaphores = &semaphore};
  vkQueueSubmit(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

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
  EXPECT_EQ(elementMock->getDrawCount(), 3);
  graph.render();

  auto timestamps2 = graph.getTimestamps();
  EXPECT_EQ(graph.getFrameInFlight(), (2 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 6);
  EXPECT_EQ(timestamps2.size(), 3);
  EXPECT_TRUE(timestamps2.find("Render") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("Postprocessing") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("GUI") != timestamps2.end());
  EXPECT_GE(timestamps2["Render"].y, timestamps2["Render"].x);
  EXPECT_GE(timestamps2["Postprocessing"].x, timestamps2["Render"].y);
  EXPECT_GE(timestamps2["Postprocessing"].y, timestamps2["Postprocessing"].x);
  EXPECT_GE(timestamps2["GUI"].x, timestamps2["Postprocessing"].y);
  EXPECT_GE(timestamps1["GUI"].y, timestamps1["GUI"].x);

  for (int i = 0; i < 100; i++) {
    graph.render();
    EXPECT_EQ(graph.getFrameInFlight(), ((2 + i + 1) % framesInFlight));
    EXPECT_EQ(elementMock->getDrawCount(), 3 * (i + 3));
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
    positionImage->changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_NONE, VK_ACCESS_NONE,
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
  // wait swapchain semaphore
  EXPECT_EQ(renderPass.getWaitSemaphores().size(), 1);
  EXPECT_EQ(renderPass.getSignalSemaphores().size(), 1);
  EXPECT_EQ(postprocessingPass.getWaitSemaphores().size(), 1);
  EXPECT_EQ(postprocessingPass.getSignalSemaphores().size(), 1);
  EXPECT_EQ(guiPass.getWaitSemaphores().size(), 1);
  EXPECT_EQ(guiPass.getSignalSemaphores().size(), 1);
  commandBuffer[graph.getFrameInFlight()].endCommands();

  auto loadSemaphore = RenderGraph::Semaphore(VK_SEMAPHORE_TYPE_TIMELINE, device);
  uint64_t loadCounter = 1;
  VkTimelineSemaphoreSubmitInfo timelineInfo = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &loadCounter,
  };

  auto semaphore = loadSemaphore.getSemaphore();
  VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .pNext = &timelineInfo,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &commandBuffer[graph.getFrameInFlight()].getCommandBuffer(),
                             .signalSemaphoreCount = 1,
                             .pSignalSemaphores = &semaphore};
  vkQueueSubmit(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

  VkSemaphoreWaitInfo waitInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = 0,
      .semaphoreCount = 1,
      .pSemaphores = &semaphore,
      .pValues = &loadCounter,
  };

  vkWaitSemaphores(device.getLogicalDevice(), &waitInfo, UINT64_MAX);

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
  EXPECT_EQ(elementMock->getDrawCount(), 3);
  graph.render();

  auto timestamps2 = graph.getTimestamps();
  EXPECT_EQ(graph.getFrameInFlight(), (2 % framesInFlight));
  EXPECT_EQ(elementMock->getDrawCount(), 6);
  EXPECT_EQ(timestamps2.size(), 3);
  EXPECT_TRUE(timestamps2.find("Render") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("Postprocessing") != timestamps2.end());
  EXPECT_TRUE(timestamps2.find("GUI") != timestamps2.end());
  EXPECT_GE(timestamps2["Render"].y, timestamps2["Render"].x);
  EXPECT_GE(timestamps2["Postprocessing"].x, timestamps2["Render"].y);
  EXPECT_GE(timestamps2["Postprocessing"].y, timestamps2["Postprocessing"].x);
  EXPECT_GE(timestamps2["GUI"].x, timestamps2["Postprocessing"].y);
  EXPECT_GE(timestamps1["GUI"].y, timestamps1["GUI"].x);

  for (int i = 0; i < 100; i++) {
    graph.render();
    EXPECT_EQ(graph.getFrameInFlight(), ((2 + i + 1) % framesInFlight));
    EXPECT_EQ(elementMock->getDrawCount(), 3 * (i + 3));
  }

  // wait device idle before destroying resources
  vkDeviceWaitIdle(device.getLogicalDevice());
}

TEST(ScenarioTest, GraphReset) {
  glm::ivec2 resolution(1920, 1080);
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
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
    positionImage->changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_NONE, VK_ACCESS_NONE,
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
  VkTimelineSemaphoreSubmitInfo timelineInfo = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &loadCounter,
  };

  auto semaphore = loadSemaphore.getSemaphore();
  VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .pNext = &timelineInfo,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &commandBuffer[graph.getFrameInFlight()].getCommandBuffer(),
                             .signalSemaphoreCount = 1,
                             .pSignalSemaphores = &semaphore};
  vkQueueSubmit(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

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
  commandBuffer[graph.getFrameInFlight()].beginCommands();
  graph.getGraphStorage().reset(swapchain.getImageViews(), swapchainNewImages, commandBuffer[graph.getFrameInFlight()]);
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

  commandBuffer[graph.getFrameInFlight()].endCommands();
  VkSubmitInfo submitInfoReset = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers = &commandBuffer[graph.getFrameInFlight()].getCommandBuffer()};
  vkQueueSubmit(device.getQueue(vkb::QueueType::graphics), 1, &submitInfoReset, nullptr);

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
  depthAttachment->changeLayout(depthAttachment->getImageLayout(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                VK_ACCESS_NONE, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                commandBuffer[graph.getFrameInFlight()]);

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
  VkTimelineSemaphoreSubmitInfo timelineInfo = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &loadCounter,
  };

  auto semaphore = loadSemaphore.getSemaphore();
  VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .pNext = &timelineInfo,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &commandBuffer[graph.getFrameInFlight()].getCommandBuffer(),
                             .signalSemaphoreCount = 1,
                             .pSignalSemaphores = &semaphore};
  vkQueueSubmit(device.getQueue(vkb::QueueType::graphics), 1, &submitInfo, nullptr);

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