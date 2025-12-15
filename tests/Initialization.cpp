#include <gtest/gtest.h>
import Instance;
import Device;
import Window;
import Surface;
import Allocator;
import Buffer;
import Command;
import CommandPool;
import Shader;
import DescriptorBuffer;
import Pipeline;
import Swapchain;
import Sync;
import Texture;
import <algorithm>;
import <fstream>;

TEST(InstanceTest, CreateWithoutValidation) {
  RenderGraph::Instance instance("TestApp", false);
  EXPECT_FALSE(instance.isDebug());
  EXPECT_NE(instance.getInstance().instance, nullptr);
}

TEST(InstanceTest, CreateWithValidation) {
  RenderGraph::Instance instance("TestApp", true);
  EXPECT_TRUE(instance.isDebug());
}

TEST(WindowTest, CreateWithInitialization) {
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  EXPECT_NE(window.getWindow(), nullptr);
}

TEST(WindowTest, CreateWithoutInitialization) {
  RenderGraph::Window window({1920, 1080});
  EXPECT_EQ(window.getWindow(), nullptr);
}

TEST(WindowTest, ResizedFlag) {
  RenderGraph::Window window({1920, 1080});
  EXPECT_FALSE(window.getResized());
  window.setResized(true);
  EXPECT_TRUE(window.getResized());
}

TEST(SurfaceTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  EXPECT_NE(surface.getSurface(), nullptr);
}

TEST(DeviceTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  EXPECT_NE(device.getPhysicalDevice(), nullptr);
  EXPECT_NE(device.getDevice(), nullptr);
  EXPECT_NE(device.getLogicalDevice(), nullptr);
}

TEST(DeviceTest, SupportedFormatFeature) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  // Assuming VK_FORMAT_R8G8B8A8_UNORM with VK_IMAGE_TILING_LINEAR does not support VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
  EXPECT_TRUE(device.isFormatFeatureSupported(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
}

TEST(DeviceTest, QueueFamilyProperties) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  auto properties = device.getQueueFamilyProperties(vkb::QueueType::graphics);
  EXPECT_GT(properties.queueCount, 0);
}

TEST(DeviceTest, DeviceProperties) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  auto properties = device.getDeviceProperties();
  EXPECT_GT(properties.apiVersion, 0);
}

TEST(AllocatorTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  EXPECT_NE(allocator.getAllocator(), nullptr);
}

TEST(BufferTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, allocator);
  EXPECT_NE(buffer.getBuffer(), nullptr);
  EXPECT_EQ(buffer.getSize(), 1024);
  EXPECT_NE(buffer.getAllocation(), nullptr);
}

TEST(CommandTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);  
  EXPECT_NO_THROW(RenderGraph::CommandBuffer commandBuffer(commandPool, device));
  
}

TEST(CommandTest, BeginEnd) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  EXPECT_TRUE(commandBuffer.getActive());
  commandBuffer.endCommands();
  EXPECT_FALSE(commandBuffer.getActive());
}

TEST(BufferTest, SetDataCPU) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                             allocator);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  std::vector<std::byte> data(512, std::byte{1});
  EXPECT_NO_THROW(buffer.setData(data, commandBuffer));
  commandBuffer.endCommands();
}

TEST(BufferTest, SetDataPotentiallyStaging) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT,
                             allocator);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  std::vector<std::byte> data(512, std::byte{1});
  EXPECT_NO_THROW(buffer.setData(data, commandBuffer));
  commandBuffer.endCommands();
}

TEST(BufferTest, ShaderCreate) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::Shader shader(device);
  // #version 450
  // void main() {}
  const uint32_t minimalSPIRv[] = {0x07230203, 0x00010000, 0x000d0003, 0x00000006, 0x00000000, 0x20011,    0x00000001,
                                   0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
                                   0x00000000, 0x00000001, 0x0005000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000,
                                   0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
                                   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00050036, 0x00000002,
                                   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000100fd, 0x00010038};
  std::vector<char> spirvCode(reinterpret_cast<const char*>(minimalSPIRv),
                              reinterpret_cast<const char*>(minimalSPIRv) + sizeof(minimalSPIRv));
  EXPECT_NO_THROW(shader.add(spirvCode));
}

TEST(BufferTest, GetDeviceAddress) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                             allocator);
  EXPECT_NE(buffer.getDeviceAddress(device), 0);
}

TEST(DescriptorBufferTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::DescriptorSetLayout layout(device);
  std::vector<VkDescriptorSetLayoutBinding> layoutColor{{.binding = 0,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                         .descriptorCount = 1,
                                                         .stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                                         .pImmutableSamplers = nullptr}};
  layout.createCustom(layoutColor);
  RenderGraph::DescriptorBuffer descriptorBuffer({&layout}, allocator, device);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  EXPECT_THROW(descriptorBuffer.initialize(commandBuffer), std::runtime_error);
  EXPECT_EQ(descriptorBuffer.getOffsets().size(), 1);
  EXPECT_EQ(descriptorBuffer.getOffsets()[0], 0);
  EXPECT_GT(descriptorBuffer.getLayoutSize(), 0);
  EXPECT_GT(descriptorBuffer.getLayoutSize(), descriptorBuffer.getOffsets().back());
  commandBuffer.endCommands();
}

TEST(DescriptorBufferTest, BigDescriptorCount) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::DescriptorSetLayout layout(device);
  std::vector<VkDescriptorSetLayoutBinding> layoutColor{{.binding = 0,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                         .descriptorCount = 4,
                                                         .stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                                         .pImmutableSamplers = nullptr}};
  layout.createCustom(layoutColor);
  RenderGraph::DescriptorBuffer descriptorBuffer({&layout}, allocator, device);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  EXPECT_THROW(descriptorBuffer.initialize(commandBuffer), std::runtime_error);
  EXPECT_EQ(descriptorBuffer.getOffsets().size(), 4);
  EXPECT_GT(descriptorBuffer.getOffsets()[1], descriptorBuffer.getOffsets()[0]);
  EXPECT_GT(descriptorBuffer.getOffsets()[2], descriptorBuffer.getOffsets()[1]);
  EXPECT_GT(descriptorBuffer.getOffsets()[3], descriptorBuffer.getOffsets()[2]);
  commandBuffer.endCommands();
}

TEST(DescriptorBufferTest, DifferentDescriptors) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::DescriptorSetLayout layout(device);  
  std::vector<VkDescriptorSetLayoutBinding> layoutBinding{{.binding = 0,
                                                           .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                           .descriptorCount = 1,
                                                           .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                                                           .pImmutableSamplers = nullptr},
                                                          {.binding = 1,
                                                           .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                           .descriptorCount = 1,
                                                           .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                                           .pImmutableSamplers = nullptr}};
  layout.createCustom(layoutBinding);
  RenderGraph::DescriptorBuffer descriptorBuffer({&layout}, allocator, device);
  EXPECT_EQ(descriptorBuffer.getOffsets().size(), 2);
  bool offset = false;
  if ((descriptorBuffer.getOffsets()[0] == 0 && descriptorBuffer.getOffsets()[1] != 0) ||
      (descriptorBuffer.getOffsets()[1] == 0 && descriptorBuffer.getOffsets()[0] != 0))
    offset = true;
  EXPECT_EQ(offset, true);
}

TEST(DescriptorBufferTest, Update) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Buffer buffer(1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                             allocator);  
  RenderGraph::DescriptorSetLayout layout(device);
  std::vector<VkDescriptorSetLayoutBinding> layoutColor{{.binding = 0,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                         .descriptorCount = 1,
                                                         .stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                                         .pImmutableSamplers = nullptr}};
  layout.createCustom(layoutColor);
  RenderGraph::DescriptorBuffer descriptorBuffer({&layout}, allocator, device);
  descriptorBuffer.add(VkDescriptorAddressInfoEXT{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                                  .pNext = nullptr,
                                                  .address = buffer.getDeviceAddress(device),
                                                  .range = buffer.getSize(),
                                                  .format = VK_FORMAT_UNDEFINED},
                       VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  descriptorBuffer.initialize(commandBuffer);
  EXPECT_THROW(descriptorBuffer.initialize(commandBuffer), std::runtime_error);
  EXPECT_NE(descriptorBuffer.getBuffer()->getDeviceAddress(device), 0);
  EXPECT_THROW(descriptorBuffer.add(VkDescriptorAddressInfoEXT{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                                               .pNext = nullptr,
                                                               .address = buffer.getDeviceAddress(device),
                                                               .range = buffer.getSize(),
                                                               .format = VK_FORMAT_UNDEFINED},
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), std::runtime_error);
  commandBuffer.endCommands();
}

TEST(PipelineTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::Shader shader(device);
  // #version 450
  // void main() {}
  const uint32_t minimalSPIRv[] = {0x07230203, 0x00010000, 0x000d0003, 0x00000006, 0x00000000, 0x20011,    0x00000001,
                                   0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
                                   0x00000000, 0x00000001, 0x0005000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000,
                                   0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
                                   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00050036, 0x00000002,
                                   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000100fd, 0x00010038};
  std::vector<char> spirvCode(reinterpret_cast<const char*>(minimalSPIRv),
                              reinterpret_cast<const char*>(minimalSPIRv) + sizeof(minimalSPIRv));
  shader.add(spirvCode);

  RenderGraph::PipelineGraphic pipelineGraphic;
  RenderGraph::Pipeline pipeline(device);
  
  RenderGraph::DescriptorSetLayout layout(device);
  std::vector<VkDescriptorSetLayoutBinding> layoutColor{{.binding = 0,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                         .descriptorCount = 1,
                                                         .stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                                         .pImmutableSamplers = nullptr}};
  layout.createCustom(layoutColor);
  VkVertexInputBindingDescription bindingDescription{.binding = 0,
                                                     .stride = sizeof(glm::vec4),
                                                     .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
  // custom fields
  std::vector<std::tuple<VkFormat, uint32_t>> fields{{VK_FORMAT_R32G32B32_SFLOAT, 0}};
  std::vector<VkVertexInputAttributeDescription> attributeDescriptions(fields.size());
  for (int i = 0; i < fields.size(); i++) {
    attributeDescriptions[i] = {.location = static_cast<uint32_t>(i),
                                .binding = 0,
                                .format = std::get<0>(fields[i]),
                                .offset = std::get<1>(fields[i])};
  }

  std::vector<std::pair<std::string, RenderGraph::DescriptorSetLayout*>> descriptorSetLayouts;
  descriptorSetLayouts.emplace_back("test", &layout);

  // validation error is expected because minimal shader does not have any input
  EXPECT_NO_THROW(pipeline.createGraphic(
      pipelineGraphic,
      shader.getShaderStageInfo(),
      descriptorSetLayouts,
      {}, bindingDescription,
      attributeDescriptions));
}

TEST(SwapchainTest, CreateWithoutInitialization) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(allocator, device);
  EXPECT_NE(swapchain.getSwapchain(), nullptr);
  EXPECT_EQ(swapchain.getImageCount(), 0);
  EXPECT_EQ(swapchain.getImageViews().size(), 0);
}

TEST(SwapchainTest, CreateWithInitialization) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(allocator, device);
  RenderGraph::CommandPool commandPool(vkb::QueueType::graphics, device);
  RenderGraph::CommandBuffer commandBuffer(commandPool, device);
  commandBuffer.beginCommands();
  swapchain.initialize(commandBuffer);
  commandBuffer.endCommands();
  EXPECT_GE(swapchain.getImageCount(), 2);
  EXPECT_GE(swapchain.getImageViews().size(), 2);
}

TEST(SyncTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::Semaphore semaphoreTimeline(VK_SEMAPHORE_TYPE_TIMELINE, device);
  RenderGraph::Semaphore semaphoreBinary(VK_SEMAPHORE_TYPE_BINARY, device);
  EXPECT_NE(semaphoreTimeline.getSemaphore(), nullptr);
  EXPECT_NE(semaphoreBinary.getSemaphore(), nullptr);
}

TEST(ImageTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Image image(allocator);  
  image.createImage(VK_FORMAT_R8G8B8A8_UNORM, {512, 512}, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  EXPECT_NE(image.getImage(), nullptr);
  EXPECT_EQ(image.getFormat(), VK_FORMAT_R8G8B8A8_UNORM);
  EXPECT_EQ(image.getResolution(), glm::ivec2(512, 512));
  EXPECT_EQ(image.getMipMapNumber(), 1);
  EXPECT_EQ(image.getLayerNumber(), 1);
  EXPECT_EQ(image.getImageLayout(), VK_IMAGE_LAYOUT_UNDEFINED);
}

TEST(ImageViewTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  std::unique_ptr<RenderGraph::Image> image = std::make_unique<RenderGraph::Image>(allocator);
  image->createImage(VK_FORMAT_R8G8B8A8_UNORM, {512, 512}, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  RenderGraph::ImageView imageView(device);
  imageView.createImageView(std::move(image), VK_IMAGE_VIEW_TYPE_2D,
                            VK_IMAGE_ASPECT_COLOR_BIT, 0, 0);
  EXPECT_NE(imageView.getImageView(), nullptr);
  EXPECT_EQ(imageView.getImage().getFormat(), VK_FORMAT_R8G8B8A8_UNORM);
  EXPECT_EQ(imageView.getImage().getResolution(), glm::ivec2(512, 512));
  EXPECT_EQ(imageView.getImage().getMipMapNumber(), 1);
  EXPECT_EQ(imageView.getImage().getLayerNumber(), 1);
}

TEST(ImageViewHolderTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  std::unique_ptr<RenderGraph::Image> image1 = std::make_unique<RenderGraph::Image>(allocator);
  image1->createImage(VK_FORMAT_R8G8B8A8_UNORM, {512, 512}, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  std::shared_ptr<RenderGraph::ImageView> imageView1 = std::make_shared<RenderGraph::ImageView>(device);
  imageView1->createImageView(std::move(image1), VK_IMAGE_VIEW_TYPE_2D,
                             VK_IMAGE_ASPECT_COLOR_BIT, 0, 0);
  std::unique_ptr<RenderGraph::Image> image2 = std::make_unique<RenderGraph::Image>(allocator);
  image2->createImage(VK_FORMAT_R8G8B8A8_UNORM, {512, 512}, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  std::shared_ptr<RenderGraph::ImageView> imageView2 = std::make_shared<RenderGraph::ImageView>(device);
  imageView2->createImageView(std::move(image2), VK_IMAGE_VIEW_TYPE_2D,
                             VK_IMAGE_ASPECT_COLOR_BIT, 0, 0);
  std::vector<std::shared_ptr<RenderGraph::ImageView>> imageViews{imageView1, imageView2};
  int index = 0;
  RenderGraph::ImageViewHolder holder(imageViews, [&index]() { return index; });
  EXPECT_EQ(&holder.getImageView(), imageViews[0].get());
  index = 1;
  EXPECT_EQ(&holder.getImageView(), imageViews[1].get());
  auto containedViews = holder.getImageViews();
  EXPECT_EQ(containedViews.size(), imageViews.size());
  EXPECT_TRUE(holder.contains(imageViews));
}

TEST(SamplerTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::Sampler sampler(device);
  sampler.createSampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, 1, 4, VK_FILTER_LINEAR);
  EXPECT_NE(sampler.getSampler(), nullptr);
}

TEST(TextureTest, Create) {
  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::MemoryAllocator allocator(device, instance);
  std::unique_ptr<RenderGraph::Image> image = std::make_unique<RenderGraph::Image>(allocator);
  image->createImage(VK_FORMAT_R8G8B8A8_UNORM, {512, 512}, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  std::shared_ptr<RenderGraph::ImageView> imageView = std::make_shared<RenderGraph::ImageView>(device);
  imageView->createImageView(std::move(image), VK_IMAGE_VIEW_TYPE_2D,
                            VK_IMAGE_ASPECT_COLOR_BIT, 0, 0);
  std::shared_ptr<RenderGraph::Sampler> sampler = std::make_shared<RenderGraph::Sampler>(device);
  sampler->createSampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, 1, 4, VK_FILTER_LINEAR);
  RenderGraph::Texture texture(imageView, sampler);
  EXPECT_EQ(&texture.getImageView(), imageView.get());
  EXPECT_EQ(&texture.getSampler(), sampler.get());
}

template <typename T>
constexpr bool IsValidImageCPU = requires { typename RenderGraph::ImageCPU<T>; };

TEST(ImageCPUTest, AcceptsArithmeticTypes) {
  EXPECT_TRUE(IsValidImageCPU<int>);
  EXPECT_TRUE(IsValidImageCPU<float>);
}

TEST(ImageCPUTest, RejectsNonArithmeticTypes) {
  EXPECT_FALSE(IsValidImageCPU<std::string>);
  EXPECT_FALSE(IsValidImageCPU<void*>);
}

TEST(ImageCPUTest, WithoutDeleter) {
  std::vector<float> pixels(256 * 256, 0.5f);
  {
    RenderGraph::ImageCPU<float> imageCPU;    
    imageCPU.setData(pixels.data(), [](float* data) {});
    auto data = imageCPU.getData();
    for (int i = 0; i < 256 * 256; i++) {
      EXPECT_EQ(data[i], 0.5f);
    }
  }
  for (int i = 0; i < 256 * 256; i++) {
    EXPECT_EQ(pixels[i], 0.5f);
  }
}

TEST(ImageCPUTest, WithDeleter) {
  std::vector<float> pixels(256 * 256, 0.5f);
  bool deleterCalled = false;
  {
    RenderGraph::ImageCPU<float> imageCPU;    
    imageCPU.setData(pixels.data(), [&deleterCalled](float* data) { deleterCalled = true; });
    auto data = imageCPU.getData();
    for (int i = 0; i < 256 * 256; i++) {
      EXPECT_EQ(data[i], 0.5f);
    }
  }
  EXPECT_TRUE(deleterCalled);
}

TEST(ShaderTest, Reflection) {
  auto readFileDesktop = [&](const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file " + filename);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
  };
  auto vertexSpirv = readFileDesktop("../resources/vertex.spv");
  auto fragmentSpirv = readFileDesktop("../resources/fragment.spv");

  RenderGraph::Instance instance("TestApp", false);
  RenderGraph::Window window({1920, 1080});
  window.initialize();
  RenderGraph::Surface surface(window, instance);
  RenderGraph::Device device(surface, instance);
  RenderGraph::Shader shader(device);
  shader.add(fragmentSpirv);
  auto resultFragment = shader.getDescriptorSetLayoutBindings();
  EXPECT_EQ(resultFragment.size(), 1);
  EXPECT_EQ(resultFragment[0].binding, 1);
  EXPECT_EQ(resultFragment[0].stageFlags, VK_SHADER_STAGE_FRAGMENT_BIT);
  shader.add(vertexSpirv);
  auto resultVertex = shader.getDescriptorSetLayoutBindings();
  EXPECT_EQ(resultVertex.size(), 2);
  EXPECT_EQ(resultVertex[0].binding, 0);
  EXPECT_EQ(resultVertex[1].binding, 1);
  EXPECT_EQ(resultVertex[0].stageFlags, VK_SHADER_STAGE_VERTEX_BIT);
  EXPECT_EQ(resultVertex[1].stageFlags, VK_SHADER_STAGE_FRAGMENT_BIT);
  auto vertexInfo = shader.getVertexInputInfo();
  auto vertexAttributes = vertexInfo->pVertexAttributeDescriptions;
  EXPECT_EQ(vertexAttributes[0].binding, 0);
  EXPECT_EQ(vertexAttributes[0].location, 0);
  EXPECT_EQ(vertexAttributes[0].offset, 0);
  EXPECT_EQ(vertexAttributes[0].format, VK_FORMAT_R32G32_SFLOAT);
  EXPECT_EQ(vertexAttributes[1].binding, 0);
  EXPECT_EQ(vertexAttributes[1].location, 1);
  EXPECT_EQ(vertexAttributes[1].offset, 8);
  EXPECT_EQ(vertexAttributes[1].format, VK_FORMAT_R32G32_SFLOAT);
  EXPECT_EQ(vertexAttributes[2].binding, 0);
  EXPECT_EQ(vertexAttributes[2].location, 2);
  EXPECT_EQ(vertexAttributes[2].offset, 16);
  EXPECT_EQ(vertexAttributes[2].format, VK_FORMAT_R32G32B32A32_SFLOAT);
  auto bindingDescription = vertexInfo->pVertexBindingDescriptions;
  EXPECT_GE(bindingDescription->stride, vertexAttributes[2].offset);
}